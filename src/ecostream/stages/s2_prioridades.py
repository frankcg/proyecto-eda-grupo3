"""Etapa 2 - Cola con prioridades (S3): min-heap binario.

Prioridad (menor = antes), desempate por tiempo del evento (FIFO):
    0 alerta de fraude/pago (re-inyectada por la etapa 6)
    1 intento de pago            2 compra express o cliente premium
    3 compra estándar            4 compra programada
    5 devolución                 6 carrito
    7 reseña                     8 vista / búsqueda
El consumidor extrae como máximo ``capacity`` eventos por tick (capacidad de
servicio); ante picos, los eventos de baja prioridad esperan en el heap y
los críticos no.  Se mide la espera (en ticks) por clase.
Salida -> eventos en orden de prioridad hacia la etapa 3.
"""
from __future__ import annotations

from collections import defaultdict

from ..structures.heap import BinaryHeap
from .base import StageMetrics, timed

PRIORITY_NAMES = {0: "alerta", 1: "pago", 2: "compra_express/premium", 3: "compra_estandar",
                  4: "compra_programada", 5: "devolucion", 6: "carrito", 7: "resena", 8: "vista/busqueda"}


def priority_of(ev: dict) -> int:
    t = ev["event_type"]
    if t == "fraud_alert":
        return 0
    if t == "payment_attempt":
        return 1
    if t == "purchase":
        if ev.get("shipping") == "express" or ev.get("premium"):
            return 2
        return 3 if ev.get("shipping") != "scheduled" else 4
    return {"return": 5, "cart": 6, "remove_from_cart": 6, "review": 7}.get(t, 8)


class ColaPrioridades:
    def __init__(self, capacity_per_tick: int = 5000, tracer=None):
        self.heap = BinaryHeap()
        self.capacity = capacity_per_tick
        self.tick = 0
        self.metrics = StageMetrics("E2 Cola de prioridades (min-heap)")
        self.wait_sum = defaultdict(float)
        self.wait_n = defaultdict(int)
        self.max_backlog = 0
        self.tracer = tracer

    def push_batch(self, batch: list) -> None:
        with timed(self.metrics, len(batch)):
            for ev in batch:
                p = priority_of(ev)
                ev["_prio"] = p
                ev["_tick_in"] = self.tick
                self.heap.push((p, ev["event_time"]), ev)
        if self.tracer:
            for ev in batch:
                if self.tracer.watching(ev):
                    self.tracer(ev, "E2", f"insertado en heap con prioridad {ev['_prio']} "
                                f"({PRIORITY_NAMES[ev['_prio']]}); heap={len(self.heap)}")
        self.max_backlog = max(self.max_backlog, len(self.heap))

    def push_alert(self, alert: dict) -> None:
        alert["_prio"] = 0
        alert["_tick_in"] = self.tick
        self.heap.push((0, alert["event_time"]), alert)

    def pop(self):
        """Extrae el siguiente evento si queda capacidad en el tick."""
        if not len(self.heap):
            return None
        with timed(self.metrics):
            _, ev = self.heap.pop()
        p = ev["_prio"]
        self.wait_sum[p] += self.tick - ev["_tick_in"]
        self.wait_n[p] += 1
        self.metrics.items_out += 1
        if self.tracer and self.tracer.watching(ev):
            self.tracer(ev, "E2", f"extraído del heap tras esperar {self.tick - ev['_tick_in']} ticks")
        return ev

    def end_tick(self):
        self.tick += 1

    def avg_wait(self) -> dict:
        return {PRIORITY_NAMES[p]: round(self.wait_sum[p] / self.wait_n[p], 3)
                for p in sorted(self.wait_n) if self.wait_n[p]}
