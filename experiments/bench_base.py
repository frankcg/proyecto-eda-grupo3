"""Experimentos de estructuras base (etapas 0-5 y 9).

    python experiments/bench_base.py [--max=1e7] [--only=fifo,heap,...]
"""
from __future__ import annotations

import bisect
import heapq
import random
from collections import OrderedDict, deque

import numpy as np

from harness import Recorder, memory, save_platform, sizes_from_argv, timeit, SEED
from memrss import memory_rss

from ecostream.structures.bplus_tree import BPlusTree
from ecostream.structures.hash_table import HashTable
from ecostream.structures.heap import BinaryHeap
from ecostream.structures.kd_tree import KDTree, knn_brute_force
from ecostream.structures.linked_queue import LinkedQueue
from ecostream.structures.lru_cache import LRUCache
from ecostream.structures.sorting import bisect_left, merge_sort
from ecostream.structures.trie import Trie, linear_autocomplete

REC = Recorder("base")


def mem(setup, run, n, rss_from=10_000_000):
    return memory_rss(setup, run) if n >= rss_from else memory(setup, run)


def record(estructura, operacion, n, setup, run, ops=None, extra=None, do_mem=True, max_reps=100, rss_from=10_000_000):
    if do_mem and n >= rss_from:      # medir memoria primero: no mantener dos estructuras vivas
        cur, peak = mem(setup, run, n, rss_from)
    mean, std, reps, res = timeit(setup, run, n, max_reps=max_reps)
    if not (do_mem and n >= rss_from):
        cur, peak = mem(setup, run, n, rss_from) if do_mem else (float("nan"), float("nan"))
    ops = ops or n
    row = dict(estructura=estructura, operacion=operacion, n=n, reps=reps, media_s=round(mean, 6),
               std_s=round(std, 6), us_por_op=round(1e6 * mean / ops, 4),
               mem_actual_mb=round(cur, 3), mem_pico_mb=round(peak, 3))
    if extra:
        row.update(extra(res) if callable(extra) else extra)
    REC.add(**row)
    return res


def ints(n, seed=SEED):
    return np.random.default_rng(seed + n).integers(0, 2**62, n).tolist()


# --------------------------------------------------------------------- E0
def bench_fifo(n):
    data = list(range(n))
    def own(_):
        q = LinkedQueue()
        for x in data:
            q.enqueue(x)
        while len(q):
            q.dequeue()
        return q
    def own_keep(_):            # memoria con n elementos encolados
        q = LinkedQueue()
        for x in data:
            q.enqueue(x)
        return q
    record("LinkedQueue (propia)", "enqueue+dequeue", n, lambda: None, own, ops=2 * n, do_mem=False)
    record("LinkedQueue (propia)", "n encolados (memoria)", n, lambda: None, own_keep, do_mem=True)
    def dq(_):
        q = deque()
        for x in data:
            q.append(x)
        while q:
            q.popleft()
        return q
    record("collections.deque (ref.)", "enqueue+dequeue", n, lambda: None, dq, ops=2 * n, do_mem=False)
    if n <= 100_000:
        def lst(_):
            q = []
            for x in data:
                q.append(x)
            while q:
                q.pop(0)
            return q
        record("list.pop(0) (ingenua)", "enqueue+dequeue", n, lambda: None, lst, ops=2 * n, do_mem=False)


# --------------------------------------------------------------------- E1
def bench_sort(n):
    data = ints(n)
    record("Merge Sort (propio)", "ordenar", n, lambda: list(data), lambda d: merge_sort(d))
    record("sorted/Timsort (ref. C)", "ordenar", n, lambda: list(data), lambda d: sorted(d))
    a = sorted(data)
    rng = random.Random(n)
    qs = [a[rng.randrange(n)] if rng.random() < 0.5 else rng.randrange(2**62) for _ in range(min(n, 1_000_000))]
    def own(_):
        return sum(1 for x in qs if (i := bisect_left(a, x)) < n and a[i] == x)
    def ref(_):
        return sum(1 for x in qs if (i := bisect.bisect_left(a, x)) < n and a[i] == x)
    record("Búsqueda binaria (propia)", "consultas", n, lambda: None, own, ops=len(qs), do_mem=False)
    record("bisect (ref. C)", "consultas", n, lambda: None, ref, ops=len(qs), do_mem=False)
    if n <= 100_000:
        def lin(_):
            s = 0
            for x in qs[:1000]:
                s += x in a          # búsqueda lineal O(n)
            return s
        record("Búsqueda lineal (ingenua)", "consultas", n, lambda: None, lin, ops=1000, do_mem=False, max_reps=10)


# --------------------------------------------------------------------- E2
def bench_heap(n):
    pr = np.random.default_rng(SEED + n).integers(0, 9, n).tolist()
    def own(_):
        h = BinaryHeap()
        for i, p in enumerate(pr):
            h.push(p, i)
        while len(h):
            h.pop()
        return h
    def own_keep(_):
        h = BinaryHeap()
        for i, p in enumerate(pr):
            h.push(p, i)
        return h
    record("BinaryHeap (propio)", "push+pop", n, lambda: None, own, ops=2 * n, do_mem=False)
    record("BinaryHeap (propio)", "n en cola (memoria)", n, lambda: None, own_keep)
    def ref(_):
        h = []
        for i, p in enumerate(pr):
            heapq.heappush(h, (p, i))
        while h:
            heapq.heappop(h)
        return h
    record("heapq (ref. C)", "push+pop", n, lambda: None, ref, ops=2 * n, do_mem=False)
    if n <= 100_000:
        def ins(_):
            h = []
            for i, p in enumerate(pr):
                bisect.insort(h, (-p, -i))   # lista ordenada: inserción O(n)
            while h:
                h.pop()
            return h
        record("Lista ordenada (insort)", "push+pop", n, lambda: None, ins, ops=2 * n, do_mem=False, max_reps=10)


# --------------------------------------------------------------------- E3
def bench_hash(n):
    keys = ints(n)
    def own(_):
        t = HashTable()
        for k in keys:
            t.put(k, 1)
        s = 0
        for k in keys:
            s += t.get(k)
        return t
    def ref(_):
        t = {}
        for k in keys:
            t[k] = 1
        s = 0
        for k in keys:
            s += t[k]
        return t
    record("HashTable sondeo lineal (propia)", "put+get", n, lambda: None, own, ops=2 * n,
           extra=lambda t: {"factor_carga": round(t.load_factor, 3), "sondeos_por_op": round(t.probes / (2 * n), 3)})
    record("dict (ref. C)", "put+get", n, lambda: None, ref, ops=2 * n)


def bench_lru(n):
    rng = np.random.default_rng(SEED + n)
    keys = (rng.zipf(1.2, n) % max(n, 10)).tolist()
    cap = max(10, n // 100)
    def own(_):
        c = LRUCache(cap)
        for k in keys:
            if c.get(k) is None:
                c.put(k, k)
        return c
    def ref(_):
        c = OrderedDict()
        for k in keys:
            if k in c:
                c.move_to_end(k)
            else:
                c[k] = k
                if len(c) > cap:
                    c.popitem(last=False)
        return c
    record("LRUCache (propia)", "get/put Zipf", n, lambda: None, own,
           extra=lambda c: {"hit_rate": round(c.hit_rate, 4), "capacidad": cap})
    record("OrderedDict LRU (ref.)", "get/put Zipf", n, lambda: None, ref)


# --------------------------------------------------------------------- E4
def bench_bptree(n):
    keys = ints(n)
    rng = random.Random(n)
    def build(_):
        t = BPlusTree(pool_pages=1024)
        for i, k in enumerate(keys):
            t.insert(k, i)
        return t
    t = record("B+Tree (propio, páginas 4 KB)", "insertar", n, lambda: None, build,
               extra=lambda t: {"altura": t.height, "paginas": t.pager.n_pages, "escrituras_disco": t.pager.disk_writes},
               max_reps=20)
    t.pager.flush()
    pq = [keys[rng.randrange(n)] for _ in range(10_000)]
    r0 = t.pager.disk_reads
    pool = t.pager.pool
    def point(_):
        return sum(t.search(k) is not None for k in pq)
    m, s, reps, _ = timeit(lambda: None, point, n)
    REC.add(estructura="B+Tree (propio, páginas 4 KB)", operacion="busqueda puntual", n=n, reps=reps,
            media_s=round(m, 6), std_s=round(s, 6), us_por_op=round(1e6 * m / len(pq), 4),
            lecturas_disco_por_consulta=round((t.pager.disk_reads - r0) / (reps * len(pq)), 4),
            accesos_pagina_por_consulta=t.height, pool_hit_rate=round(pool.hit_rate, 4))
    sk = sorted(keys)
    width = max(1, n // 1000)
    rq = [rng.randrange(max(1, n - width)) for _ in range(100)]
    def rng_q(_):
        tot = 0
        for i in rq:
            tot += sum(1 for _ in t.range(sk[i], sk[i + width - 1]))
        return tot
    m, s, reps, tot = timeit(lambda: None, rng_q, n)
    REC.add(estructura="B+Tree (propio, páginas 4 KB)", operacion=f"rango (~{width} claves)", n=n, reps=reps,
            media_s=round(m, 6), std_s=round(s, 6), us_por_op=round(1e6 * m / 100, 4), resultados=tot // 100)
    def bulk(_):
        b = BPlusTree(pool_pages=1024)
        b.bulk_load((k, i) for i, k in enumerate(sk))
        return b
    b = record("B+Tree (propio, páginas 4 KB)", "bulk load ordenado", n, lambda: None, bulk, do_mem=False,
               extra=lambda b: {"altura": b.height, "paginas": b.pager.n_pages})
    b.close()
    t.close()
    if n <= 100_000:
        def ins(_):
            a = []
            for k in keys:
                bisect.insort(a, k)
            return a
        record("Arreglo ordenado + insort", "insertar", n, lambda: None, ins, max_reps=10)


# --------------------------------------------------------------------- E5
def kd_data(n, d=4):
    rng = np.random.default_rng(SEED + n)
    c = rng.normal(0, 3, (40, d))
    return (c[rng.integers(0, 40, n)] + rng.normal(0, 1, (n, d))).astype(np.float32)


def bench_kdtree(n):
    X = kd_data(n)
    kd = record("KD-Tree 4D (propio)", "construir", n, lambda: None, lambda _: KDTree(X, 32),
                max_reps=20 if n < 10**6 else 3)
    Q = X[np.random.default_rng(1).integers(0, n, 100)] + 0.01
    def q_kd(_):
        return [kd.query(q, 10)[1] for q in Q]
    def q_bf(_):
        return [knn_brute_force(X, q, 10)[1] for q in Q]
    kd.nodes_visited = 0
    m, s, reps, r1 = timeit(lambda: None, q_kd, n)
    visited = kd.nodes_visited / ((reps + 1) * len(Q))
    m2, s2, reps2, r2 = timeit(lambda: None, q_bf, n)
    exact = np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(r1, r2)])
    REC.add(estructura="KD-Tree 4D (propio)", operacion="KNN k=10", n=n, reps=reps, media_s=round(m, 6),
            std_s=round(s, 6), us_por_op=round(1e6 * m / len(Q), 3), nodos_visitados=round(visited, 1),
            recall_vs_exacto=round(exact, 4))
    REC.add(estructura="KNN fuerza bruta (NumPy)", operacion="KNN k=10", n=n, reps=reps2, media_s=round(m2, 6),
            std_s=round(s2, 6), us_por_op=round(1e6 * m2 / len(Q), 3))


# --------------------------------------------------------------------- E9
SYL = ["sam", "lap", "smart", "pro", "max", "tel", "vi", "son", "ex", "ul", "tra", "mi", "ni", "air", "gal", "zo", "ra", "fit"]


def names(n):
    rng = np.random.default_rng(SEED + n)
    idx = rng.integers(0, len(SYL), (n, 3))
    return [f"{SYL[a]}{SYL[b]} {SYL[c]} {i}" for i, (a, b, c) in enumerate(idx.tolist())]


def bench_trie(n):
    nm = names(n)
    w = np.random.default_rng(3).random(n).tolist()
    def build(_):
        t = Trie(k=8)
        for s, x in zip(nm, w):
            t.insert(s, None, x)
        return t
    t = record("Trie top-K cacheado (propio)", "insertar", n, lambda: None, build,
               extra=lambda t: {"nodos": t.n_nodes}, max_reps=10, rss_from=1_000_000)
    rng = random.Random(1)
    pref = [nm[rng.randrange(n)][:rng.randint(2, 6)] for _ in range(1000)]
    m, s, reps, _ = timeit(lambda: None, lambda _: [t.autocomplete(p) for p in pref], n)
    REC.add(estructura="Trie top-K cacheado (propio)", operacion="autocompletar k=8", n=n, reps=reps,
            media_s=round(m, 6), std_s=round(s, 6), us_por_op=round(1e6 * m / 1000, 3))
    terms = list(zip(nm, w))
    lp = pref[:20] if n >= 1_000_000 else pref[:100]
    m, s, reps, _ = timeit(lambda: None, lambda _: [linear_autocomplete(terms, p) for p in lp], n, max_reps=10)
    REC.add(estructura="Escaneo lineal (ingenuo)", operacion="autocompletar k=8", n=n, reps=reps,
            media_s=round(m, 6), std_s=round(s, 6), us_por_op=round(1e6 * m / len(lp), 3))


BENCHES = {"fifo": bench_fifo, "sort": bench_sort, "heap": bench_heap, "hash": bench_hash, "lru": bench_lru,
           "bptree": bench_bptree, "kdtree": bench_kdtree, "trie": bench_trie}

if __name__ == "__main__":
    save_platform()
    sizes, only = sizes_from_argv()
    for name, fn in BENCHES.items():
        if only and name not in only:
            continue
        for n in sizes:
            print(f"[{name}] n={n:,}", flush=True)
            fn(n)
