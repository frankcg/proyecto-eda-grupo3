"""Etapa 1 (S2) - Ordenación eficiente y búsqueda binaria.

* ``merge_sort``: Merge Sort estable bottom-up (iterativo).  Tiempo
  Θ(n log n) en todos los casos, memoria auxiliar O(n).  Se elige frente a
  Quicksort porque es **estable** (preserva el orden de llegada de eventos con
  igual clave, necesario para deduplicar conservando la primera ocurrencia) y
  su peor caso no se degrada con lotes casi ordenados (los streams llegan casi
  ordenados por tiempo).
* ``bisect_left`` / ``binary_search``: búsqueda binaria O(log n).
* ``dedup_sorted``: elimina duplicados adyacentes en O(n) tras ordenar.
"""
from __future__ import annotations

from typing import Callable, Sequence


def merge_sort(items: list, key: Callable | None = None) -> list:
    """Devuelve una nueva lista ordenada (estable) usando Merge Sort bottom-up."""
    n = len(items)
    if n <= 1:
        return list(items)
    if key is None:
        keys = list(items)
    else:
        keys = [key(x) for x in items]
    # Ordenamos índices para no recalcular claves; estabilidad: en empate
    # tomamos primero el de la mitad izquierda (<=).
    src = list(range(n))
    dst = [0] * n
    width = 1
    # Optimización: runs iniciales de 16 con inserción (como Timsort/introsort)
    RUN = 16
    for lo in range(0, n, RUN):
        hi = min(lo + RUN, n)
        for i in range(lo + 1, hi):
            cur = src[i]
            kc = keys[cur]
            j = i - 1
            while j >= lo and keys[src[j]] > kc:
                src[j + 1] = src[j]
                j -= 1
            src[j + 1] = cur
    width = RUN
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                if keys[src[i]] <= keys[src[j]]:
                    dst[k] = src[i]
                    i += 1
                else:
                    dst[k] = src[j]
                    j += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
        src, dst = dst, src
        width *= 2
    return [items[i] for i in src]


def bisect_left(a: Sequence, x, lo: int = 0, hi: int | None = None) -> int:
    """Primer índice i tal que a[i] >= x (a ordenado). O(log n)."""
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search(a: Sequence, x) -> bool:
    """True si x está en la secuencia ordenada a."""
    i = bisect_left(a, x)
    return i < len(a) and a[i] == x


def dedup_sorted(items: list, key: Callable) -> list:
    """Elimina duplicados adyacentes de una lista ya ordenada por ``key``."""
    out = []
    last = object()
    for it in items:
        k = key(it)
        if k != last:
            out.append(it)
            last = k
    return out


def merge_sorted_unique(a: list, b: list) -> list:
    """Mezcla dos listas ordenadas sin duplicados (O(|a|+|b|))."""
    i = j = 0
    out = []
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] > b[j]:
            out.append(b[j]); j += 1
        else:
            out.append(a[i]); i += 1; j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
