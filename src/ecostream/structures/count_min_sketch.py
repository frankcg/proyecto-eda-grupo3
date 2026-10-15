"""Etapa 6 [FIJA] (S7) - Count-Min Sketch (Cormode & Muthukrishnan, 2005).

Matriz d x w de contadores.  update(x, c): para cada fila i suma c en
``C[i, h_i(x)]``.  estimate(x) = min_i C[i, h_i(x)].

Garantía: con w = ceil(e/ε) y d = ceil(ln(1/δ)),
    f(x) <= f̂(x) <= f(x) + ε·N   con probabilidad >= 1 - δ
(N = suma de todas las cuentas).  Nunca subestima; puede sobreestimar por
colisiones.  Tiempo O(d) por operación, memoria O(w·d) independiente del
número de claves distintas.

``WindowedCountMin``: ventana deslizante por *buckets* de tiempo.  Se mantiene
un anillo de B sub-sketches (uno por bucket de ``bucket_seconds``); la
estimación en la ventana suma los B buckets (la cota pasa a ε·N_ventana por
bucket).  Al avanzar el tiempo, el bucket más antiguo se pone a cero: memoria
O(B·w·d) y expiración O(w·d) por bucket.
"""
from __future__ import annotations

import math

import numpy as np

from .hashing import key_to_u64, splitmix64, splitmix64_scalar


class CountMinSketch:
    def __init__(self, epsilon: float = 0.001, delta: float = 0.01,
                 width: int | None = None, depth: int | None = None, seed: int = 7):
        self.w = width or math.ceil(math.e / epsilon)
        self.d = depth or math.ceil(math.log(1 / delta))
        self.epsilon = math.e / self.w
        self.delta = math.exp(-self.d)
        self.table = np.zeros((self.d, self.w), dtype=np.int64)
        self.seed = seed
        self.total = 0

    def _cols(self, x: int):
        h1 = splitmix64_scalar(x, self.seed)
        h2 = splitmix64_scalar(x, self.seed + 1) | 1
        w = self.w
        return [((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % w for i in range(self.d)]

    def update(self, key, count: int = 1) -> None:
        x = key_to_u64(key)
        t = self.table
        for i, c in enumerate(self._cols(x)):
            t[i, c] += count
        self.total += count

    def estimate(self, key) -> int:
        x = key_to_u64(key)
        t = self.table
        return int(min(t[i, c] for i, c in enumerate(self._cols(x))))

    # ---- vectorizado ----
    def _cols_many(self, keys_u64):
        h1 = splitmix64(keys_u64, self.seed)
        h2 = splitmix64(keys_u64, self.seed + 1) | np.uint64(1)
        i = np.arange(self.d, dtype=np.uint64)
        with np.errstate(over="ignore"):
            return ((h1[None, :] + i[:, None] * h2[None, :]) % np.uint64(self.w)).astype(np.int64)

    def update_many(self, keys_u64: np.ndarray, counts=None, chunk: int = 1 << 20) -> None:
        keys_u64 = np.asarray(keys_u64, dtype=np.uint64)
        for s in range(0, len(keys_u64), chunk):
            k = keys_u64[s:s + chunk]
            cols = self._cols_many(k)
            c = 1 if counts is None else np.asarray(counts[s:s + chunk], dtype=np.int64)
            flat = (cols + (np.arange(self.d, dtype=np.int64) * self.w)[:, None]).ravel()
            if counts is None:
                self.table.ravel()[:] += np.bincount(flat, minlength=self.d * self.w)
            else:
                np.add.at(self.table.ravel(), flat, np.tile(c, self.d))
        self.total += len(keys_u64) if counts is None else int(np.sum(counts))

    def estimate_many(self, keys_u64: np.ndarray) -> np.ndarray:
        cols = self._cols_many(np.asarray(keys_u64, dtype=np.uint64))
        rows = np.arange(self.d)[:, None]
        return self.table[rows, cols].min(axis=0)

    def merge(self, other: "CountMinSketch") -> None:
        self.table += other.table
        self.total += other.total

    def error_bound(self) -> float:
        return self.epsilon * self.total

    def memory_bytes(self) -> int:
        return self.table.nbytes


class WindowedCountMin:
    """Count-Min sobre ventana deslizante [t - window, t] con buckets."""

    def __init__(self, window_seconds: float = 300, n_buckets: int = 5,
                 epsilon: float = 0.001, delta: float = 0.01):
        self.window = window_seconds
        self.B = n_buckets
        self.bucket_seconds = window_seconds / n_buckets
        self.sketches = [CountMinSketch(epsilon, delta) for _ in range(n_buckets)]
        self.current = None       # índice absoluto del bucket actual

    def _advance(self, ts: float) -> CountMinSketch:
        b = int(ts // self.bucket_seconds)
        if self.current is None:
            self.current = b
        if b > self.current:
            # expira buckets que salen de la ventana
            for old in range(self.current + 1, min(b, self.current + self.B) + 1):
                sk = self.sketches[old % self.B]
                sk.table.fill(0)
                sk.total = 0
            self.current = b
        return self.sketches[b % self.B]

    def update(self, key, ts: float, count: int = 1) -> int:
        """Actualiza y devuelve la estimación en la ventana para ``key``."""
        self._advance(ts).update(key, count)
        return self.estimate(key)

    def estimate(self, key) -> int:
        return sum(sk.estimate(key) for sk in self.sketches)

    def memory_bytes(self) -> int:
        return sum(s.memory_bytes() for s in self.sketches)
