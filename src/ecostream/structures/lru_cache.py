"""Etapa 3 (S4) - Caché LRU: tabla hash + lista doblemente enlazada.

get/put O(1): la tabla hash localiza el nodo y la lista mantiene el orden de
uso (cabeza = más reciente, cola = víctima de expulsión).  Se usa para
mantener disponibles los productos/consultas usados recientemente y como
*buffer pool* de páginas del B+Tree (etapa 4).
"""
from __future__ import annotations


class _DNode:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int, on_evict=None):
        if capacity <= 0:
            raise ValueError("capacity > 0")
        self.capacity = capacity
        self.on_evict = on_evict          # callback(key, val) p.ej. escribir página sucia
        self._map: dict = {}
        self._head = _DNode()             # centinelas
        self._tail = _DNode()
        self._head.next = self._tail
        self._tail.prev = self._head
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._map)

    def __contains__(self, key):
        return key in self._map

    def _unlink(self, n: _DNode):
        n.prev.next = n.next
        n.next.prev = n.prev

    def _push_front(self, n: _DNode):
        n.next = self._head.next
        n.prev = self._head
        self._head.next.prev = n
        self._head.next = n

    def get(self, key, default=None):
        n = self._map.get(key)
        if n is None:
            self.misses += 1
            return default
        self.hits += 1
        self._unlink(n)
        self._push_front(n)
        return n.val

    def put(self, key, val) -> None:
        n = self._map.get(key)
        if n is not None:
            n.val = val
            self._unlink(n)
            self._push_front(n)
            return
        if len(self._map) >= self.capacity:
            victim = self._tail.prev
            self._unlink(victim)
            del self._map[victim.key]
            if self.on_evict is not None:
                self.on_evict(victim.key, victim.val)
        n = _DNode(key, val)
        self._map[key] = n
        self._push_front(n)

    def items(self):
        n = self._head.next
        while n is not self._tail:
            yield n.key, n.val
            n = n.next

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0
