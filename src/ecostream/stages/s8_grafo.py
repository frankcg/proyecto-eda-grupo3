"""Etapa 8 [FIJA] - Grafo de dominio (S8-S9): co-compra + PageRank + Louvain.

Cada ``refresh`` lotes:
1. Range scan del B+Tree de órdenes (etapa 4) en la ventana [t - 7 días, t].
2. Solo órdenes **aprobadas** por la etapa 6 (las bloqueadas por fraude no
   contaminan el grafo).
3. Grafo ponderado de co-compra: w_ij = nº de órdenes con i y j.
4. PageRank (influencia global) y Louvain (comunidades de consumo).
   Personalized PageRank bajo demanda con semillas = carrito/compras del
   usuario (etapa 9).
Salida -> puntajes PR, comunidad por producto y PPR para la etapa 9.
"""
from __future__ import annotations

import numpy as np

from ..graph.communities import community_stats, louvain, modularity
from ..graph.graph import WeightedGraph
from ..graph.pagerank import pagerank, personalized_pagerank
from .base import StageMetrics, timed


class GrafoDominio:
    def __init__(self, n_products: int, window_days: float = 7.0, tracer=None):
        self.n = n_products
        self.window = window_days * 86400
        self.g = None
        self.pr = np.full(n_products, 1.0 / n_products)
        self.comm = np.zeros(n_products, dtype=np.int64)
        self.q = 0.0
        self.metrics = StageMetrics("E8 Grafo co-compra: PageRank + Louvain")
        self.tracer = tracer
        self.last_orders = 0

    def rebuild(self, index, now: float) -> None:
        with timed(self.metrics):
            baskets = [o["items"] for o in index.orders_in_window(now - self.window, now)
                       if o["status"] == "aprobado" and len(o["items"]) >= 1]
            self.last_orders = len(baskets)
            g = WeightedGraph.from_baskets(self.n, baskets)
            if g.m == 0:
                return
            self.g = g
            self.pr, _ = pagerank(g)
            self.comm, _ = louvain(g, seed=0)
            self.q = modularity(g, self.comm)
        st = community_stats(self.comm[g.degree > 0]) if (g.degree > 0).any() else {"n_communities": 0}
        self.metrics.counters["reconstrucciones"] += 1
        self.metrics.counters["ordenes_en_ventana"] = len(baskets)
        self.metrics.counters["aristas"] = g.m
        self.metrics.counters["modularidad"] = round(self.q, 4)
        self.metrics.counters["comunidades"] = st["n_communities"]

    def ppr(self, seeds) -> np.ndarray:
        seeds = [s for s in seeds if s is not None]
        if self.g is None or not seeds:
            return self.pr
        r, _ = personalized_pagerank(self.g, seeds)
        return r

    def process(self, ev: dict) -> dict:
        if self.tracer and self.tracer.watching(ev) and ev["event_type"] == "purchase":
            o = ev.get("_order", {})
            if o.get("status") == "aprobado":
                self.tracer(ev, "E8", f"orden {o.get('order_id')} entra al grafo de co-compra en la "
                            f"siguiente reconstrucción (ventana 7 días vía range scan del B+Tree)")
            else:
                self.tracer(ev, "E8", f"orden con estado '{o.get('status')}': excluida del grafo")
        return ev
