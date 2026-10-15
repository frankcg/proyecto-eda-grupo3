"""Etapa 8 [FIJA] - PageRank y Personalized PageRank (Brin & Page, 1998).

Iteración de potencias sobre la matriz de transición ponderada:
    r_{t+1} = d · Pᵀ r_t + (d · masa_colgante + (1 - d)) · v
con P_{ij} = w_ij / s_i (s_i = fuerza del nodo), d = 0.85 y v = vector de
teletransporte (uniforme -> PageRank; concentrado en semillas -> PPR).
Cada iteración es O(m) (producto matriz dispersa-vector implementado con
``np.bincount``); converge geométricamente con razón d, por lo que
~log(tol)/log(d) ≈ 85 iteraciones para tol = 1e-6 en el peor caso.

PPR con semillas = productos del carrito del usuario => afinidad de cada
producto con ese carrito (usado por el servicio de recomendación).
Línea base: ranking por grado ponderado (fuerza).
"""
from __future__ import annotations

import numpy as np

from .graph import WeightedGraph


def pagerank(g: WeightedGraph, damping: float = 0.85, tol: float = 1e-9,
             max_iter: int = 200, personalization: np.ndarray | None = None):
    n = g.n
    if personalization is None:
        v = np.full(n, 1.0 / n)
    else:
        v = np.asarray(personalization, dtype=np.float64)
        v = v / v.sum()
    s = g.strength
    dangling = s == 0
    inv_s = np.zeros(n)
    inv_s[~dangling] = 1.0 / s[~dangling]
    edge_p = g.wt * inv_s[g.src]            # P_{src -> nbr}
    r = v.copy()
    it = 0
    for it in range(1, max_iter + 1):
        contrib = np.bincount(g.nbr, weights=edge_p * r[g.src], minlength=n)
        dmass = r[dangling].sum()
        r_new = damping * contrib + (damping * dmass + (1 - damping)) * v
        err = np.abs(r_new - r).sum()
        r = r_new
        if err < tol:
            break
    return r, it


def personalized_pagerank(g: WeightedGraph, seeds, damping: float = 0.85, **kw):
    v = np.zeros(g.n)
    v[np.asarray(list(seeds), dtype=np.int64)] = 1.0
    return pagerank(g, damping, personalization=v, **kw)


def degree_ranking(g: WeightedGraph) -> np.ndarray:
    return g.strength.astype(np.float64)
