"""Etapa 3 (S4) - Tabla hash con direccionamiento abierto (sondeo lineal).

* Capacidad potencia de 2, índice = hash(k) & (cap-1).
* Factor de carga máximo 0.70 -> redimensiona x2 (rehash amortizado O(1)).
* Borrado con lápidas (tombstones) para no romper las cadenas de sondeo.
Complejidad esperada: get/put/delete O(1) promedio, O(n) peor caso.
Memoria: O(cap) con cap <= n/0.35 tras redimensionar.

Se eligió sondeo lineal frente a encadenamiento por su localidad de caché
(los arreglos de claves contiguos) y porque no requiere asignar nodos.
"""
from __future__ import annotations

_EMPTY = object()
_TOMB = object()


class HashTable:
    __slots__ = ("_keys", "_vals", "_cap", "_size", "_used", "probes")

    MAX_LOAD = 0.70

    def __init__(self, capacity: int = 16):
        cap = 8
        while cap < capacity:
            cap <<= 1
        self._cap = cap
        self._keys = [_EMPTY] * cap
        self._vals = [None] * cap
        self._size = 0      # claves vivas
        self._used = 0      # vivas + lápidas
        self.probes = 0     # métrica: sondeos totales

    def __len__(self):
        return self._size

    def _find_slot(self, key):
        """Devuelve (índice, encontrado)."""
        mask = self._cap - 1
        i = hash(key) & mask
        keys = self._keys
        first_tomb = -1
        while True:
            k = keys[i]
            if k is _EMPTY:
                return (first_tomb if first_tomb >= 0 else i), False
            if k is _TOMB:
                if first_tomb < 0:
                    first_tomb = i
            elif k == key:
                return i, True
            i = (i + 1) & mask
            self.probes += 1

    def _resize(self, new_cap: int):
        old_k, old_v = self._keys, self._vals
        self._cap = new_cap
        self._keys = [_EMPTY] * new_cap
        self._vals = [None] * new_cap
        self._size = self._used = 0
        for k, v in zip(old_k, old_v):
            if k is not _EMPTY and k is not _TOMB:
                self.put(k, v)

    def put(self, key, value) -> None:
        if (self._used + 1) > self.MAX_LOAD * self._cap:
            self._resize(self._cap * 2 if self._size * 2 >= self._used else self._cap)
        i, found = self._find_slot(key)
        if found:
            self._vals[i] = value
            return
        if self._keys[i] is _EMPTY:
            self._used += 1
        self._keys[i] = key
        self._vals[i] = value
        self._size += 1

    __setitem__ = put

    def get(self, key, default=None):
        i, found = self._find_slot(key)
        return self._vals[i] if found else default

    def __getitem__(self, key):
        i, found = self._find_slot(key)
        if not found:
            raise KeyError(key)
        return self._vals[i]

    def __contains__(self, key) -> bool:
        return self._find_slot(key)[1]

    def delete(self, key) -> bool:
        i, found = self._find_slot(key)
        if not found:
            return False
        self._keys[i] = _TOMB
        self._vals[i] = None
        self._size -= 1
        return True

    def items(self):
        for k, v in zip(self._keys, self._vals):
            if k is not _EMPTY and k is not _TOMB:
                yield k, v

    def keys(self):
        for k, _ in self.items():
            yield k

    @property
    def load_factor(self) -> float:
        return self._used / self._cap
