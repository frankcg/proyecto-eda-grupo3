"""Etapa 2 (S3) - Min-heap binario para cola de prioridades.

Implementación en arreglo (hijo izq = 2i+1, der = 2i+2).
    push / pop   O(log n)
    peek         O(1)
    heapify      O(n)   (Floyd)
Las entradas son tuplas ``(prioridad, secuencia, item)``; ``secuencia`` es un
contador monotónico que (i) desempata por orden de llegada (FIFO dentro de la
misma prioridad, evita inanición dentro de una clase) y (ii) evita comparar
``item`` directamente.

También se incluye ``top_k``: selección de los K mayores en O(n log K) con un
min-heap de tamaño K, usado por la etapa 9 (servicio).
"""
from __future__ import annotations

import itertools
from typing import Iterable


class BinaryHeap:
    __slots__ = ("_a", "_seq")

    def __init__(self, items: Iterable | None = None):
        self._seq = itertools.count()
        self._a = []
        if items is not None:
            self._a = [(p, next(self._seq), x) for p, x in items]
            self._heapify()

    def __len__(self):
        return len(self._a)

    def _heapify(self):
        a = self._a
        for i in range(len(a) // 2 - 1, -1, -1):
            self._sift_down(i)

    def push(self, priority, item) -> None:
        a = self._a
        a.append((priority, next(self._seq), item))
        # sift-up
        i = len(a) - 1
        e = a[i]
        while i > 0:
            p = (i - 1) >> 1
            if e < a[p]:
                a[i] = a[p]
                i = p
            else:
                break
        a[i] = e

    def _sift_down(self, i: int) -> None:
        a = self._a
        n = len(a)
        e = a[i]
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            r = c + 1
            if r < n and a[r] < a[c]:
                c = r
            if a[c] < e:
                a[i] = a[c]
                i = c
            else:
                break
        a[i] = e

    def pop(self):
        """Extrae (prioridad, item) con la menor prioridad."""
        a = self._a
        if not a:
            raise IndexError("pop de un heap vacío")
        last = a.pop()
        if not a:
            return last[0], last[2]
        top = a[0]
        a[0] = last
        self._sift_down(0)
        return top[0], top[2]

    def peek(self):
        if not self._a:
            raise IndexError("peek de un heap vacío")
        return self._a[0][0], self._a[0][2]


def top_k(pairs: Iterable, k: int) -> list:
    """Devuelve los k pares (score, item) de mayor score, ordenados desc.

    Usa un min-heap de tamaño k: O(n log k) tiempo, O(k) memoria.
    """
    h = BinaryHeap()
    for score, item in pairs:
        if len(h) < k:
            h.push(score, item)
        elif score > h.peek()[0]:
            h.pop()
            h.push(score, item)
    out = []
    while len(h):
        out.append(h.pop())
    out.reverse()
    return out
