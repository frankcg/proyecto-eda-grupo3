"""Etapa 7 [FIJA] (S7) - HNSW: Hierarchical Navigable Small World
(Malkov & Yashunin, 2020).

Grafo de proximidad multicapa.  Cada elemento recibe un nivel máximo
l = floor(-ln(U)·m_L) con m_L = 1/ln(M) (distribución geométrica): las capas
superiores son "autopistas" con pocos nodos y la capa 0 contiene todos.
* Inserción: búsqueda voraz desde el punto de entrada en las capas > l; en
  las capas <= l, ``search_layer`` con ef_construction candidatos y selección
  de M vecinos con la **heurística de diversidad** (Algoritmo 4 del paper):
  un candidato e se acepta si está más cerca de q que de cualquier vecino ya
  elegido, lo que preserva conexiones entre clústeres.
* Consulta: descenso voraz + ``search_layer`` en capa 0 con ef_search >= k.
Complejidad empírica: construcción O(n log n), consulta ~O(log n);
memoria O(n·M) aristas + los vectores.

Detalle de implementación: las distancias de un nodo a sus vecinos se
calculan vectorizadas con NumPy; las colas de prioridad internas usan
``heapq`` (implementación en C, verificada contra nuestro BinaryHeap en los
tests) porque esta es la ruta crítica.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


class HNSW:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100,
                 ef_search: int = 50, max_elements: int = 1024, seed: int = 42):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.efC = ef_construction
        self.ef = ef_search
        self.mL = 1 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.data = np.empty((max_elements, dim), dtype=np.float32)
        self.labels: list = []
        self.graph: list[dict[int, list[int]]] = []   # graph[capa][nodo] -> vecinos
        self.entry = -1
        self.max_level = -1
        self.n = 0
        self.dist_evals = 0

    # ------------------------------------------------------------------
    def _dists(self, q: np.ndarray, ids) -> np.ndarray:
        diff = self.data[ids] - q
        self.dist_evals += len(ids)
        return np.einsum("ij,ij->i", diff, diff)

    def _search_layer(self, q, entry_points, ef: int, layer: int):
        """Devuelve lista [(dist, id)] de los ef más cercanos encontrados."""
        g = self.graph[layer]
        visited = set(entry_points)
        d0 = self._dists(q, list(entry_points))
        cand = [(float(d), e) for d, e in zip(d0, entry_points)]      # min-heap
        heapq.heapify(cand)
        res = [(-d, e) for d, e in cand]                              # max-heap
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -res[0][0]:
                break
            nbrs = [v for v in g.get(c, ()) if v not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            ds = self._dists(q, nbrs)
            worst = -res[0][0]
            for dv, v in zip(ds.tolist(), nbrs):
                if len(res) < ef or dv < worst:
                    heapq.heappush(cand, (dv, v))
                    heapq.heappush(res, (-dv, v))
                    if len(res) > ef:
                        heapq.heappop(res)
                    worst = -res[0][0]
        return sorted((-d, e) for d, e in res)

    def _select_heuristic(self, cands, M: int):
        """Algoritmo 4 (heurística) con keepPrunedConnections."""
        if len(cands) <= M:
            return [e for _, e in cands]
        selected: list[int] = []
        pruned = []
        for d, e in cands:                 # cands ordenado por distancia asc.
            if len(selected) >= M:
                break
            if selected:
                de = self._dists(self.data[e], selected)
                if (de < d).any():
                    pruned.append(e)
                    continue
            selected.append(e)
        for e in pruned:
            if len(selected) >= M:
                break
            selected.append(e)
        return selected

    def _random_level(self) -> int:
        return int(-math.log(1.0 - self.rng.random()) * self.mL)

    # ------------------------------------------------------------------
    def add(self, x: np.ndarray, label=None) -> int:
        if self.n >= len(self.data):
            self.data = np.concatenate([self.data, np.empty_like(self.data)])
        i = self.n
        self.data[i] = x
        self.labels.append(i if label is None else label)
        self.n += 1
        lvl = self._random_level()
        while len(self.graph) <= lvl:
            self.graph.append({})
        for lc in range(lvl + 1):
            self.graph[lc][i] = []
        if self.entry < 0:
            self.entry, self.max_level = i, lvl
            return i
        q = self.data[i]
        ep = [self.entry]
        for lc in range(self.max_level, lvl, -1):
            ep = [self._search_layer(q, ep, 1, lc)[0][1]]
        for lc in range(min(lvl, self.max_level), -1, -1):
            W = self._search_layer(q, ep, self.efC, lc)
            Mmax = self.M0 if lc == 0 else self.M
            nbrs = self._select_heuristic(W, self.M)
            g = self.graph[lc]
            g[i] = nbrs
            for e in nbrs:
                ne = g[e]
                ne.append(i)
                if len(ne) > Mmax:
                    de = self._dists(self.data[e], ne)
                    order = np.argsort(de)
                    g[e] = self._select_heuristic([(float(de[o]), ne[o]) for o in order], Mmax)
            ep = [e for _, e in W]
        if lvl > self.max_level:
            self.entry, self.max_level = i, lvl
        return i

    def add_items(self, X: np.ndarray, labels=None) -> None:
        if self.n + len(X) > len(self.data):
            extra = self.n + len(X) - len(self.data)
            self.data = np.concatenate([self.data, np.empty((extra, self.dim), np.float32)])
        for t, x in enumerate(X):
            self.add(x, None if labels is None else labels[t])

    def knn_query(self, q: np.ndarray, k: int = 10, ef: int | None = None):
        q = np.asarray(q, dtype=np.float32)
        ef = max(ef or self.ef, k)
        ep = [self.entry]
        for lc in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, lc)[0][1]]
        W = self._search_layer(q, ep, ef, 0)[:k]
        return (np.sqrt([d for d, _ in W]),
                np.array([self.labels[e] for _, e in W], dtype=np.int64))

    def memory_bytes(self) -> int:
        edges = sum(len(v) for g in self.graph for v in g.values())
        return self.n * self.dim * 4 + edges * 8
