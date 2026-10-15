"""Funciones hash compartidas por las estructuras probabilísticas.

* ``key_to_u64``: cualquier clave (str/int/bytes) -> entero de 64 bits con
  BLAKE2b (determinista entre ejecuciones, a diferencia de ``hash()`` de Python
  para strings, que usa una semilla aleatoria por proceso).
* ``splitmix64``: mezclador de 64 bits (Steele et al., 2014) vectorizado en
  NumPy; se usa para derivar h1, h2 en *double hashing* (Kirsch & Mitzenmacher,
  2006): g_i(x) = h1(x) + i·h2(x) mod m, que conserva la tasa de falsos
  positivos asintótica del Bloom Filter con solo dos hashes.
"""
from __future__ import annotations

import hashlib

import numpy as np

_M64 = (1 << 64) - 1


def key_to_u64(key) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key) & _M64
    if isinstance(key, str):
        key = key.encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def keys_to_u64(keys) -> np.ndarray:
    arr = np.asarray(keys)
    if arr.dtype.kind in "iu":
        return arr.astype(np.uint64, copy=False)
    return np.fromiter((key_to_u64(k) for k in keys), dtype=np.uint64, count=len(keys))


def splitmix64(x: np.ndarray, seed: int = 0) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64, copy=True) + np.uint64((0x9E3779B97F4A7C15 * (seed + 1)) & _M64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def splitmix64_scalar(x: int, seed: int = 0) -> int:
    z = (x + 0x9E3779B97F4A7C15 * (seed + 1)) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)
