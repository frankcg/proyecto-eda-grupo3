"""Etapa 8 [FIJA] (S8-S9) - Grafo de co-compra ponderado.

Nodos = productos.  Arista {i, j} con peso w_ij = número de órdenes (no
bloqueadas por fraude) que contienen a i y a j dentro de la ventana temporal
W (por defecto 7 días).  Representación: lista de aristas no dirigidas
(u < v, w) + CSR simétrico (indptr, nbr, wt) para recorridos O(grado).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


class WeightedGraph:
    def __init__(self, n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray):
        """u, v, w: aristas no dirigidas, sin duplicados, u != v."""
        self.n = int(n)
        self.u = np.asarray(u, dtype=np.int64)
        self.v = np.asarray(v, dtype=np.int64)
        self.w = np.asarray(w, dtype=np.float64)
        self.m = len(self.u)
        src = np.concatenate([self.u, self.v])
        dst = np.concatenate([self.v, self.u])
        ww = np.concatenate([self.w, self.w])
        order = np.argsort(src, kind="stable")
        self.src = src[order]
        self.nbr = dst[order]
        self.wt = ww[order]
        self.indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(self.indptr, self.src + 1, 1)
        np.cumsum(self.indptr, out=self.indptr)
        self.strength = np.bincount(self.src, weights=self.wt, minlength=self.n)
        self.degree = np.diff(self.indptr)

    @classmethod
    def from_edge_list(cls, n: int, u, v, w=None):
        """Agrega aristas repetidas (suma pesos) y elimina lazos."""
        u = np.asarray(u, dtype=np.int64); v = np.asarray(v, dtype=np.int64)
        w = np.ones(len(u)) if w is None else np.asarray(w, dtype=np.float64)
        keep = u != v
        u, v, w = u[keep], v[keep], w[keep]
        a, b = np.minimum(u, v), np.maximum(u, v)
        key = a * n + b
        uniq, inv = np.unique(key, return_inverse=True)
        ws = np.bincount(inv, weights=w)
        return cls(n, uniq // n, uniq % n, ws)

    @classmethod
    def from_baskets(cls, n: int, baskets):
        """Construye el grafo de co-compra: todos los pares de cada orden."""
        us, vs = [], []
        for items in baskets:
            its = sorted(set(items))
            for a, b in combinations(its, 2):
                us.append(a); vs.append(b)
        return cls.from_edge_list(n, us, vs)

    def neighbors(self, i: int):
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.nbr[s:e], self.wt[s:e]

    def adjacency_lists(self):
        """Listas de Python (nodo -> [(vecino, peso)]) para algoritmos iterativos."""
        nb = self.nbr.tolist(); wt = self.wt.tolist(); ip = self.indptr.tolist()
        return [list(zip(nb[ip[i]:ip[i + 1]], wt[ip[i]:ip[i + 1]])) for i in range(self.n)]

    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in (self.u, self.v, self.w, self.src, self.nbr, self.wt, self.indptr))


def planted_partition_graph(n_nodes: int, n_edges: int, n_comms: int | None = None,
                            p_in: float = 0.8, zipf_a: float = 1.3, seed: int = 42):
    """Grafo sintético con comunidades plantadas y grados de cola pesada.

    Imita un grafo de co-compra: categorías = comunidades; popularidad de
    producto ~ Zipf.  Devuelve (WeightedGraph, etiquetas_verdaderas).
    """
    rng = np.random.default_rng(seed)
    if n_comms is None:
        n_comms = max(2, int(np.sqrt(n_nodes) / 2))
    sizes = rng.pareto(1.5, n_comms) + 1
    sizes = np.maximum(1, (sizes / sizes.sum() * n_nodes).astype(np.int64))
    sizes[-1] += n_nodes - sizes.sum()
    if sizes[-1] < 1:
        sizes = np.full(n_comms, n_nodes // n_comms); sizes[-1] += n_nodes - sizes.sum()
    labels = np.repeat(np.arange(n_comms), sizes)[:n_nodes]
    rng.shuffle(labels)
    # popularidad intra-comunidad (Zipf truncado)
    pop = 1.0 / (rng.permutation(n_nodes) + 1.0) ** (1 / zipf_a)
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    starts = np.searchsorted(lab_sorted, np.arange(n_comms))
    ends = np.searchsorted(lab_sorted, np.arange(n_comms), side="right")
    cum = {}
    for c in range(n_comms):
        idx = order[starts[c]:ends[c]]
        p = pop[idx]; cum[c] = (idx, np.cumsum(p) / p.sum())
    glob = np.cumsum(pop) / pop.sum()
    # extremo u por popularidad global
    u = np.searchsorted(glob, rng.random(n_edges))
    u = np.minimum(u, n_nodes - 1)
    intra = rng.random(n_edges) < p_in
    v = np.searchsorted(glob, rng.random(n_edges))
    v = np.minimum(v, n_nodes - 1)
    cu = labels[u]
    for c in np.unique(cu[intra]):
        sel = np.where(intra & (cu == c))[0]
        idx, cp = cum[c]
        v[sel] = idx[np.minimum(np.searchsorted(cp, rng.random(len(sel))), len(idx) - 1)]
    g = WeightedGraph.from_edge_list(n_nodes, u, v)
    return g, labels
