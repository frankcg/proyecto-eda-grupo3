"""Etapa 9 - Servicio y visualización (transversal).

* **Autocompletado**: Trie con top-K cacheado por nodo; peso = popularidad
  (ventas aprobadas) x influencia (PageRank de la etapa 8).
* **Recomendaciones Top-K** para un usuario: fusión de candidatos
    - PPR desde su carrito/compras (etapa 8),
    - similares por comportamiento MinHash+LSH (etapa 7),
    - similares por contenido HNSW (etapa 7),
    - vecinos por atributos KD-Tree (etapa 5),
  y selección con ``top_k`` (min-heap de tamaño K, O(n log K)).  Resultados
  cacheados en LRU (etapa 3).
* **Tendencias**: heavy hitters del Count-Min Sketch de compras (etapa 6).
* **Consulta por rango de precio**: B+Tree de precios (etapa 4).
* **Panel**: ventas, actividad, alertas y métricas por etapa (dashboard.py).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..structures.heap import top_k
from ..structures.lru_cache import LRUCache
from ..structures.trie import Trie
from .base import StageMetrics, timed


class Servicio:
    def __init__(self, catalog, storage, index, kd, sim, graph, filt, tracer=None):
        self.catalog = catalog
        self.storage, self.index, self.kd, self.sim, self.graph, self.filt = storage, index, kd, sim, graph, filt
        self.trie = Trie(k=8)
        self.sales = Counter()
        self.revenue = 0.0
        for pid in range(catalog.n):
            self.trie.insert(catalog.names[pid], pid, 0.0)
        self.cache = LRUCache(10_000)
        self.version = 0
        self.metrics = StageMetrics("E9 Servicio: Trie, Top-K, panel")
        self.tracer = tracer
        self.event_counts = Counter()

    # ------------------------------------------------------------------
    def process(self, ev: dict) -> dict:
        self.event_counts[ev["event_type"]] += 1
        if ev["event_type"] == "purchase" and ev.get("_order", {}).get("status") == "aprobado":
            self.sales[ev["product_id"]] += 1
            self.revenue += ev.get("price") or 0.0
        if self.tracer and self.tracer.watching(ev):
            uid = ev.get("user_id")
            if uid is not None:
                recs = self.recommend(uid, 5)
                self.tracer(ev, "E9", "Top-5 recomendado al usuario: " +
                            ", ".join(f"{self.catalog.names[p]} ({s:.2f})" for s, p in recs))
            else:
                self.tracer(ev, "E9", "reflejado en el panel de alertas/actividad")
        return ev

    def refresh(self) -> None:
        """Actualiza pesos del Trie con ventas x PageRank (salida de la etapa 8)."""
        with timed(self.metrics):
            pr = self.graph.pr
            prn = pr / pr.max() if pr.max() > 0 else pr
            for pid, cnt in self.sales.items():
                self.trie.update_weight(self.catalog.names[pid], float(cnt * (0.5 + prn[pid])))
            self.version += 1

    # ------------------------------------------------------------------
    def autocomplete(self, prefix: str, k: int = 5):
        key = ("ac", prefix.lower(), self.version)
        res = self.cache.get(key)
        if res is None:
            with timed(self.metrics, 1):
                res = [(round(w, 2), name, pid) for w, name, pid in self.trie.autocomplete(prefix, k)]
            self.cache.put(key, res)
        return res

    def recommend(self, uid: int, k: int = 10, explain: bool = False):
        key = ("rec", uid, k, self.version, explain)
        res = self.cache.get(key)
        if res is not None:
            return res
        with timed(self.metrics, 1):
            prof = self.storage.user_profile(uid)
            seeds = list(dict.fromkeys(list(prof["cart"]) + prof["purchased"][-5:] + prof["recent"][-3:]))
            owned = set(prof["purchased"])
            scores: dict[int, float] = {}
            why: dict[int, list] = {}

            def add(p, s, tag):
                if p in owned or p in seeds:
                    return
                scores[p] = scores.get(p, 0.0) + s
                why.setdefault(p, []).append(tag)

            if seeds:
                ppr = self.graph.ppr(seeds)
                cand = np.argpartition(-ppr, min(50, len(ppr) - 1))[:50]
                mx = ppr[cand].max() or 1.0
                for p in cand:
                    if ppr[p] > 0:
                        add(int(p), 0.45 * ppr[p] / mx, "PPR")
                for s in seeds[:5]:
                    for p, j in self.sim.similar_behavior(s, 10):
                        add(p, 0.30 * j, "LSH")
                    for p, c in self.sim.similar_content(s, 5):
                        add(p, 0.15 * max(c, 0), "HNSW")
            for p in prof["kd_candidates"][-10:]:
                add(p, 0.03, "KD")
            prn = self.graph.pr / (self.graph.pr.max() or 1)
            for p in list(scores):
                scores[p] += 0.07 * prn[p]
            if not scores:      # arranque en frío: más influyentes globalmente
                best = np.argsort(-self.graph.pr)[:k]
                res = [(float(prn[p]), int(p)) for p in best]
            else:
                res = top_k(((s, p) for p, s in scores.items()), k)
            if explain:
                res = [(s, p, why.get(p, ["PR"])) for s, p in res]
        self.cache.put(key, res)
        return res

    def trending(self, k: int = 10):
        cands = list(self.sales.keys())
        return top_k(((self.filt.trending.estimate(p), p) for p in cands), k)

    def price_range(self, lo: float, hi: float, category_code: str | None = None, k: int = 10):
        pids = self.index.products_in_price_range(lo, hi)
        if category_code:
            pids = [p for p in pids if self.storage.products.get(p)["category_code"] == category_code]
        return top_k(((float(self.graph.pr[p]), p) for p in pids), k)
