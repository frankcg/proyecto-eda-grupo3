"""Etapa 5 - Índice multidimensional y KNN exacto (S6): KD-Tree.

Espacio 4-D de atributos del producto (log precio, rating, log popularidad,
descuento; estandarizados).  Para cada vista, carrito o compra se consultan los
k = 5 productos más parecidos en atributos ("alternativas de precio y
calidad similares").  Los resultados se cachean en un LRU (los productos
calientes se repiten por la ley de Zipf).  KNN exacto por fuerza bruta se
mantiene como línea base y verificación.
Salida -> candidatos por atributos en el perfil del usuario (etapa 9).
"""
from __future__ import annotations

from ..structures.kd_tree import KDTree, knn_brute_force
from ..structures.lru_cache import LRUCache
from .base import StageMetrics, timed


class IndiceMultidimensional:
    def __init__(self, catalog, k: int = 5, tracer=None):
        self.X = catalog.attributes()
        self.kd = KDTree(self.X, leaf_size=32)
        self.k = k
        self.cache = LRUCache(5000)
        self.metrics = StageMetrics("E5 KD-Tree (KNN exacto)")
        self.tracer = tracer

    def neighbors(self, pid: int) -> list:
        res = self.cache.get(pid)
        if res is None:
            _, idx = self.kd.query(self.X[pid], self.k + 1)
            res = [int(i) for i in idx if i != pid][: self.k]
            self.cache.put(pid, res)
        return res

    def verify(self, pid: int) -> bool:
        _, a = self.kd.query(self.X[pid], self.k + 1)
        _, b = knn_brute_force(self.X, self.X[pid], self.k + 1)
        return set(a.tolist()) == set(b.tolist())

    def process(self, ev: dict, users) -> dict:
        if ev["event_type"] not in ("view", "cart", "purchase") or ev.get("product_id") is None:
            return ev
        with timed(self.metrics, 1):
            nb = self.neighbors(ev["product_id"])
            if ev.get("user_id") is not None:
                prof = users.user_profile(ev["user_id"])
                prof["kd_candidates"] = (prof["kd_candidates"] + nb)[-20:]
        self.metrics.items_out += 1
        if self.tracer and self.tracer.watching(ev):
            self.tracer(ev, "E5", f"KD-Tree: {self.k} productos con atributos más cercanos {nb}")
        return ev
