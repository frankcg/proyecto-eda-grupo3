"""Etapa 3 (S4) - Matriz dispersa usuario x producto.

Formato DOK (dict de dicts) para actualizaciones incrementales O(1) desde el
stream y conversión a CSR (indptr, indices, data) en O(nnz) para recorridos
por fila/columna vectorizados (etapas 7 y 8).  Con 10^5 usuarios x 10^4
productos, una matriz densa ocuparía 8 GB; la dispersa ocupa O(nnz).
"""
from __future__ import annotations

import numpy as np


class SparseMatrix:
    def __init__(self, n_rows: int, n_cols: int):
        self.shape = (n_rows, n_cols)
        self._rows: dict[int, dict[int, float]] = {}
        self.nnz = 0

    def add(self, i: int, j: int, v: float = 1.0) -> None:
        row = self._rows.get(i)
        if row is None:
            row = self._rows[i] = {}
        if j in row:
            row[j] += v
        else:
            row[j] = v
            self.nnz += 1

    def get(self, i: int, j: int, default: float = 0.0) -> float:
        row = self._rows.get(i)
        return default if row is None else row.get(j, default)

    def row(self, i: int) -> dict:
        return self._rows.get(i, {})

    def to_csr(self):
        n = self.shape[0]
        indptr = np.zeros(n + 1, dtype=np.int64)
        for i, row in self._rows.items():
            indptr[i + 1] = len(row)
        np.cumsum(indptr, out=indptr)
        indices = np.empty(self.nnz, dtype=np.int32)
        data = np.empty(self.nnz, dtype=np.float32)
        for i, row in self._rows.items():
            s = indptr[i]
            ks = sorted(row)
            indices[s:s + len(ks)] = ks
            data[s:s + len(ks)] = [row[k] for k in ks]
        return indptr, indices, data

    def column_sets(self) -> dict[int, list[int]]:
        """Conjunto de filas (usuarios) por columna (producto): entrada de MinHash."""
        cols: dict[int, list[int]] = {}
        for i, row in self._rows.items():
            for j in row:
                cols.setdefault(j, []).append(i)
        return cols

    def memory_bytes_dense(self) -> int:
        return self.shape[0] * self.shape[1] * 8
