"""Etapa 5 (S6) - KD-Tree para KNN exacto en baja dimensión + línea base.

Construcción: partición por la mediana (``np.argpartition``, O(n) por nivel)
alternando la dimensión de mayor dispersión => O(n log n).  Las hojas guardan
``leaf_size`` puntos contiguos para calcular distancias vectorizadas.
Consulta KNN: descenso al lado más cercano y poda con la bola del k-ésimo
vecino actual (max-heap de tamaño k).  Esperado O(log n) en d bajo; tiende a
O(n) cuando d crece (maldición de la dimensionalidad), por eso en la etapa 7
(32 dimensiones) se usa HNSW.

Se elige KD-Tree frente a R-Tree porque los productos son **puntos** (no
rectángulos) y el índice es mayormente estático (se reconstruye por lotes).
"""
from __future__ import annotations

import numpy as np

from .heap import BinaryHeap


class KDTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 32):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.n, self.d = X.shape
        self.leaf_size = leaf_size
        self.perm = np.arange(self.n)
        # nodos en arreglos paralelos
        self.dim: list[int] = []
        self.val: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.start: list[int] = []
        self.end: list[int] = []
        self._X = X
        self._build()
        self.points = X[self.perm]          # puntos reordenados por hoja
        del self._X
        self.nodes_visited = 0

    def _new(self, d, v, s, e):
        self.dim.append(d); self.val.append(v)
        self.left.append(-1); self.right.append(-1)
        self.start.append(s); self.end.append(e)
        return len(self.dim) - 1

    def _build(self):
        X, perm = self._X, self.perm
        root = self._new(-1, 0.0, 0, self.n)
        stack = [root]
        while stack:
            nd = stack.pop()
            s, e = self.start[nd], self.end[nd]
            if e - s <= self.leaf_size:
                continue
            idx = perm[s:e]
            pts = X[idx]
            d = int(np.argmax(pts.max(0) - pts.min(0)))
            m = (e - s) // 2
            part = np.argpartition(pts[:, d], m)
            perm[s:e] = idx[part]
            self.dim[nd] = d
            self.val[nd] = float(X[perm[s + m], d])
            l = self._new(-1, 0.0, s, s + m)
            r = self._new(-1, 0.0, s + m, e)
            self.left[nd], self.right[nd] = l, r
            stack.append(l); stack.append(r)

    def query(self, q: np.ndarray, k: int = 10):
        """Devuelve (distancias, índices originales) de los k vecinos más cercanos."""
        q = np.asarray(q, dtype=np.float32)
        best = BinaryHeap()          # max-heap vía prioridad negativa: (-dist2, idx)
        dim, val, left, right = self.dim, self.val, self.left, self.right
        P = self.points
        stack = [(0, 0.0)]
        visited = 0
        while stack:
            nd, bound = stack.pop()
            if len(best) == k and bound >= -best.peek()[0]:
                continue
            visited += 1
            if left[nd] == -1:          # hoja
                s, e = self.start[nd], self.end[nd]
                diff = P[s:e] - q
                d2 = np.einsum("ij,ij->i", diff, diff)
                for off in np.argsort(d2)[:k]:
                    dd = float(d2[off])
                    if len(best) < k:
                        best.push(-dd, s + int(off))
                    elif dd < -best.peek()[0]:
                        best.pop(); best.push(-dd, s + int(off))
                    else:
                        break
                continue
            dd = float(q[dim[nd]]) - val[nd]
            near, far = (left[nd], right[nd]) if dd < 0 else (right[nd], left[nd])
            stack.append((far, max(bound, dd * dd)))   # se explora después
            stack.append((near, bound))
        self.nodes_visited += visited
        out = []
        while len(best):
            nd2, pos = best.pop()
            out.append((-nd2, pos))
        out.reverse()
        dists = np.sqrt(np.array([d for d, _ in out], dtype=np.float64))
        idx = self.perm[np.array([p for _, p in out], dtype=np.int64)]
        return dists, idx

    def memory_bytes(self) -> int:
        return self.points.nbytes + self.perm.nbytes + len(self.dim) * 6 * 8


def knn_brute_force(X: np.ndarray, q: np.ndarray, k: int = 10):
    """KNN exacto por fuerza bruta: O(n·d) por consulta (línea base)."""
    diff = X - np.asarray(q, dtype=X.dtype)
    d2 = np.einsum("ij,ij->i", diff, diff)
    if k < len(d2):
        idx = np.argpartition(d2, k)[:k]
    else:
        idx = np.arange(len(d2))
    idx = idx[np.argsort(d2[idx])]
    return np.sqrt(d2[idx]), idx
