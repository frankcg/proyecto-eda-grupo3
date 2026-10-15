"""Etapa 1 - Preprocesamiento por lotes (S2): Merge Sort + búsqueda binaria.

Por micro-lote:
1. **Validación y normalización**: tipo de evento en minúsculas y sin
   espacios, precio con coma decimal -> float, categoría en minúsculas y
   validada por **búsqueda binaria** en el arreglo ordenado de categorías.
   Se descartan precios negativos y tipos desconocidos; el precio faltante se
   marca para que la etapa 3 lo complete desde el catálogo.
2. **Deduplicación intra-lote**: Merge Sort (estable) por event_id y
   eliminación de adyacentes iguales: O(B log B).
3. **Deduplicación entre lotes**: búsqueda binaria de cada event_id en la
   ventana ordenada de ids recientes (O(log W) por evento); la ventana se
   actualiza mezclando listas ordenadas en O(B + W).
4. Re-orden por event_time (Merge Sort) para corregir el desorden de llegada.
Salida -> lote limpio y ordenado por tiempo para la etapa 2.
"""
from __future__ import annotations

from ..data.generator import CATEGORIES, EVENT_TYPES
from ..structures.sorting import binary_search, dedup_sorted, merge_sort, merge_sorted_unique
from .base import StageMetrics, timed

VALID_TYPES = set(EVENT_TYPES) | {"fraud_alert"}


class Preprocesamiento:
    def __init__(self, dedup_window: int = 300_000, tracer=None):
        self.categories = merge_sort([c for c, _ in CATEGORIES])
        self.recent_ids: list[int] = []        # ordenado
        self.window = dedup_window
        self.metrics = StageMetrics("E1 Preprocesamiento (Merge Sort + búsqueda binaria)")
        self.tracer = tracer

    def _normalize(self, ev: dict):
        t = ev.get("event_type")
        if not isinstance(t, str):
            return None, "tipo_invalido"
        t2 = t.strip().lower()
        if t2 not in VALID_TYPES:
            return None, "tipo_invalido"
        if t2 != t:
            self.metrics.counters["normalizado_tipo"] += 1
        ev["event_type"] = t2
        p = ev.get("price")
        if isinstance(p, str):
            try:
                p = float(p.replace(",", "."))
                self.metrics.counters["normalizado_precio"] += 1
            except ValueError:
                p = None
        if p is not None and p < 0:
            return None, "precio_negativo"
        ev["price"] = p
        if p is None:
            ev["_price_missing"] = True
            self.metrics.counters["precio_faltante"] += 1
        c = ev.get("category_code")
        if isinstance(c, str):
            c2 = c.strip().lower()
            if c2 != c:
                self.metrics.counters["normalizado_categoria"] += 1
            if not binary_search(self.categories, c2):
                return None, "categoria_invalida"
            ev["category_code"] = c2
        if ev.get("event_time") is None or ev.get("event_id") is None:
            return None, "campos_faltantes"
        return ev, None

    def process(self, batch: list) -> list:
        m = self.metrics
        with timed(m, len(batch)):
            clean = []
            for ev in batch:
                ev2, err = self._normalize(ev)
                if err:
                    m.counters["rechazado_" + err] += 1
                    if self.tracer:
                        self.tracer(ev, "E1", f"rechazado: {err}")
                    continue
                clean.append(ev2)
            # dedup intra-lote
            by_id = merge_sort(clean, key=lambda e: e["event_id"])
            uniq = dedup_sorted(by_id, key=lambda e: e["event_id"])
            m.counters["duplicado_intra_lote"] += len(by_id) - len(uniq)
            if self.tracer and len(by_id) != len(uniq):
                kept = {id(e) for e in uniq}
                for e in by_id:
                    if id(e) not in kept and self.tracer.watching(e):
                        self.tracer(e, "E1", "copia duplicada eliminada: Merge Sort por event_id + "
                                    "descarte de adyacentes iguales (O(b log b))")
            # dedup entre lotes por búsqueda binaria
            out = []
            recent = self.recent_ids
            for ev in uniq:
                if binary_search(recent, ev["event_id"]):
                    m.counters["duplicado_entre_lotes"] += 1
                    if self.tracer:
                        self.tracer(ev, "E1", "descartado: duplicado detectado por búsqueda binaria")
                    continue
                out.append(ev)
            ids = [e["event_id"] for e in out]          # ya ordenados
            self.recent_ids = merge_sorted_unique(recent, ids)[-self.window:]
            out = merge_sort(out, key=lambda e: e["event_time"])
        m.items_out += len(out)
        if self.tracer:
            for ev in out:
                if self.tracer.watching(ev):
                    self.tracer(ev, "E1", "validado/normalizado, único (búsqueda binaria en "
                                f"{len(recent)} ids recientes), reordenado por tiempo")
        return out
