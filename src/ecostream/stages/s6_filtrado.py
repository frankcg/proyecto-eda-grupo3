"""Etapa 6 [FIJA] - Filtrado probabilístico (S7): Bloom Filter + Count-Min Sketch.

Definiciones exigidas por el enunciado:
* **Bloom Filter** sobre la lista negra de *payment_token* y *device_id*
  señalados (n esperado = 50 000, p = 0.1 %).  Un positivo significa
  "posiblemente señalado" -> se **verifica** contra la lista negra exacta
  (hash table de la etapa 3) antes de bloquear; si no está, es un falso
  positivo y la orden continúa (se contabiliza).
* **Count-Min Sketch con ventana deslizante**:
    - clave: ``payment_token`` y ``device_id`` de cada intento de pago;
    - ventana: 5 minutos en 5 buckets de 60 s;
    - umbral: > 15 intentos en la ventana => alerta de velocidad
      (*card testing*).  La alerta (i) se re-inyecta en la cola de
      prioridades con prioridad 0, (ii) agrega token y dispositivo al Bloom y
      a la lista negra exacta (los siguientes intentos se bloquean) y
      (iii) se muestra en el panel de alertas.
  Un segundo CMS (ventana 1 h) cuenta compras por producto -> tendencias
  (heavy hitters) para la etapa 9.
Salida -> órdenes aprobadas/bloqueadas: solo las aprobadas actualizan los
índices de similitud (etapa 7) y el grafo de co-compra (etapa 8).
"""
from __future__ import annotations

from ..structures.bloom_filter import BloomFilter
from ..structures.count_min_sketch import WindowedCountMin
from .base import StageMetrics, timed


class FiltradoProbabilistico:
    def __init__(self, blacklist_tokens, storage, threshold: int = 15,
                 window_s: float = 300, tracer=None):
        self.bloom = BloomFilter(50_000, 0.001)
        self.storage = storage
        for t in blacklist_tokens:
            k = ("tok", int(t))
            self.bloom.add(str(k))
            storage.blacklist.put(str(k), "lista_inicial")
        self.velocity = WindowedCountMin(window_s, 5, epsilon=0.0005, delta=0.001)
        self.trending = WindowedCountMin(3600, 6, epsilon=0.0005, delta=0.001)
        self.threshold = threshold
        self.alerts: list[dict] = []
        self.metrics = StageMetrics("E6 Bloom Filter + Count-Min Sketch")
        self.tracer = tracer
        self._alerted: set = set()

    def _flag(self, key: str, reason: str):
        self.bloom.add(key)
        self.storage.blacklist.put(key, reason)

    def process(self, ev: dict, heap) -> dict:
        t = ev["event_type"]
        m = self.metrics
        if t == "purchase" and ev.get("product_id") is not None:
            self.trending.update(ev["product_id"], ev["event_time"])
            o = ev.get("_order")
            if o is not None and o["status"] == "pendiente":
                pay = self.storage.payments.get(o["order_id"])
                if pay is not None:
                    o["status"] = pay
            if self.tracer and self.tracer.watching(ev) and o is not None:
                self.tracer(ev, "E6", f"orden en estado '{o['status'].upper()}' (su intento de pago pasó por Bloom + "
                            f"verificación exacta y CMS); compra contada en el CMS de tendencias "
                            f"(~{self.trending.estimate(ev['product_id'])} en la última hora)")
            return ev
        if t != "payment_attempt":
            return ev
        with timed(m, 1):
            tok = str(("tok", ev.get("payment_token")))
            dev = str(("dev", ev.get("device_id")))
            status, why = "aprobado", ""
            # 1) Bloom: posiblemente señalado -> verificar contra la lista exacta
            for key in (tok, dev):
                if key in self.bloom:
                    m.counters["bloom_positivos"] += 1
                    if self.storage.blacklist.get(key) is not None:
                        m.counters["bloom_verificados"] += 1
                        kind = "token de pago" if key.startswith("('tok'") else "dispositivo"
                        status, why = "bloqueado", f"{kind} en lista negra (positivo del Bloom verificado)"
                        break
                    m.counters["bloom_falsos_positivos"] += 1
            # 2) CMS: intensidad en ventana de 5 min
            est_tok = self.velocity.update(tok, ev["event_time"])
            est_dev = self.velocity.update(dev, ev["event_time"])
            est = max(est_tok, est_dev)
            if est > self.threshold:
                vel = f"velocidad: ~{est} intentos en 5 min (umbral {self.threshold})"
                why = vel if status == "aprobado" else f"{vel}; además {why}"
                status = "bloqueado"
                if dev not in self._alerted:
                    self._alerted.add(dev)
                    self._flag(tok, "velocidad"); self._flag(dev, "velocidad")
                    alert = {"event_id": -(len(self.alerts) + 1), "event_type": "fraud_alert",
                             "event_time": ev["event_time"], "device_id": ev.get("device_id"),
                             "payment_token": ev.get("payment_token"), "estimate": est,
                             "reason": why, "source_event": ev["event_id"]}
                    self.alerts.append(alert)
                    heap.push_alert(alert)          # re-inyección con prioridad 0
                    m.counters["alertas_velocidad"] += 1
            ev["_payment_status"] = status
            if ev.get("order_id") is not None:
                self.storage.payments.put(ev["order_id"], status)
                o = self.storage.orders.get(ev["order_id"])
                if o is not None:
                    o["status"] = status
            m.counters["pagos_" + status] += 1
        m.items_out += 1
        if self.tracer and self.tracer.watching(ev):
            self.tracer(ev, "E6", f"Bloom/CMS: estimación ventana={est}, decisión={status.upper()} {why}")
        return ev

    def handle_alert(self, alert: dict) -> None:
        self.metrics.counters["alertas_atendidas"] += 1
