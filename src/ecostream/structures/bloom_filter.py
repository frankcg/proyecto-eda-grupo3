"""Etapa 6 [FIJA] (S7) - Bloom Filter (Bloom, 1970).

Arreglo de m bits y k funciones hash.  ``add`` enciende k bits; ``contains``
responde *"posiblemente observado"* si los k bits están en 1 y *"definitivamente
no observado"* si alguno está en 0.  No hay falsos negativos.

Dimensionamiento óptimo para n elementos y tasa de falsos positivos p:
    m = -n ln p / (ln 2)^2        k = (m/n) ln 2
    FPR teórica tras n inserciones: (1 - e^{-kn/m})^k
Tiempo O(k) por operación (independiente de n); memoria m bits
(~9.6 bits/elemento para p = 1%), frente a decenas de bytes por elemento de un
``set`` exacto.

En el pipeline, un positivo NO bloquea directamente: se verifica contra la
lista negra exacta (tabla hash de la etapa 3) antes de bloquear (decisión
crítica), tal como exige el enunciado.
"""
from __future__ import annotations

import math

import numpy as np

from .hashing import key_to_u64, splitmix64, splitmix64_scalar


class BloomFilter:
    def __init__(self, n_expected: int, fp_rate: float = 0.01):
        self.n_expected = max(1, n_expected)
        self.fp_rate = fp_rate
        m = math.ceil(-self.n_expected * math.log(fp_rate) / (math.log(2) ** 2))
        self.m = max(8, m)
        self.k = max(1, round(self.m / self.n_expected * math.log(2)))
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        self.count = 0

    # ---------------- escalar (stream evento a evento) ----------------
    def _positions(self, key):
        x = key_to_u64(key)
        h1 = splitmix64_scalar(x, 0)
        h2 = splitmix64_scalar(x, 1) | 1
        m = self.m
        return [((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % m for i in range(self.k)]

    def add(self, key) -> None:
        b = self.bits
        for p in self._positions(key):
            b[p >> 3] |= (1 << (p & 7))
        self.count += 1

    def __contains__(self, key) -> bool:
        b = self.bits
        for p in self._positions(key):
            if not (b[p >> 3] >> (p & 7)) & 1:
                return False
        return True

    contains = __contains__

    # ---------------- vectorizado (lotes) ----------------
    def _positions_many(self, keys_u64: np.ndarray) -> np.ndarray:
        h1 = splitmix64(keys_u64, 0)
        h2 = splitmix64(keys_u64, 1) | np.uint64(1)
        i = np.arange(self.k, dtype=np.uint64)
        with np.errstate(over="ignore"):
            pos = (h1[:, None] + i[None, :] * h2[:, None]) % np.uint64(self.m)
        return pos

    def add_many(self, keys_u64: np.ndarray, chunk: int = 1 << 20) -> None:
        keys_u64 = np.asarray(keys_u64, dtype=np.uint64)
        for s in range(0, len(keys_u64), chunk):
            pos = self._positions_many(keys_u64[s:s + chunk]).ravel()
            np.bitwise_or.at(self.bits, (pos >> np.uint64(3)).astype(np.int64),
                             (np.uint8(1) << (pos & np.uint64(7)).astype(np.uint8)))
        self.count += len(keys_u64)

    def contains_many(self, keys_u64: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
        keys_u64 = np.asarray(keys_u64, dtype=np.uint64)
        out = np.empty(len(keys_u64), dtype=bool)
        for s in range(0, len(keys_u64), chunk):
            pos = self._positions_many(keys_u64[s:s + chunk])
            byte = self.bits[(pos >> np.uint64(3)).astype(np.int64)]
            bit = (byte >> (pos & np.uint64(7)).astype(np.uint8)) & 1
            out[s:s + chunk] = bit.all(axis=1)
        return out

    # ---------------- métricas ----------------
    def theoretical_fpr(self, n: int | None = None) -> float:
        n = self.count if n is None else n
        return (1 - math.exp(-self.k * n / self.m)) ** self.k

    def memory_bytes(self) -> int:
        return self.bits.nbytes

    def fill_ratio(self) -> float:
        return float(np.unpackbits(self.bits).sum()) / self.m
