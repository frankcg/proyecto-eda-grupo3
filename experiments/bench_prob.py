"""Etapa 6 [FIJA]: Bloom Filter y Count-Min Sketch vs alternativas exactas.

    python experiments/bench_prob.py [--max=1e7]
Métricas: tiempo (media, std), memoria actual/pico (tracemalloc), tasa de
falsos positivos empírica vs teórica, error de frecuencia del CMS.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from harness import Recorder, memory, save_platform, sizes_from_argv, timeit, SEED

from ecostream.structures.bloom_filter import BloomFilter
from ecostream.structures.count_min_sketch import CountMinSketch

REC = Recorder("probabilisticas")


def keys(n, seed):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 2**63, int(n * 2.05), dtype=np.uint64)
    k = np.unique(k)[: 2 * n]
    rng.shuffle(k)
    return k[:n], k[n:2 * n]          # presentes, ausentes (disjuntos)


def bench_bloom(n, p=0.01):
    ins, absent = keys(n, SEED + n)
    queries = np.concatenate([ins, absent])
    def b_build(_):
        bf = BloomFilter(n, p); bf.add_many(ins); return bf
    mean, std, reps, bf = timeit(lambda: None, b_build, n)
    cur, peak = memory(lambda: None, b_build)
    mq, sq, rq, hits = timeit(lambda: None, lambda _: bf.contains_many(queries), n)
    fn = int((~hits[:n]).sum())
    fpr = float(hits[n:].mean())
    REC.add(estructura="Bloom Filter (vectorizado)" + ("" if p == 0.01 else f" p={p:g}"), operacion="insertar", n=n, p_objetivo=p, reps=reps,
            media_s=round(mean, 6), std_s=round(std, 6), us_por_op=round(1e6 * mean / n, 4),
            mem_actual_mb=round(cur, 4), mem_pico_mb=round(peak, 4), bits_por_elemento=round(bf.m / n, 2), k=bf.k)
    REC.add(estructura="Bloom Filter (vectorizado)" + ("" if p == 0.01 else f" p={p:g}"), operacion="consultar 2n", n=n, p_objetivo=p, reps=rq,
            media_s=round(mq, 6), std_s=round(sq, 6), us_por_op=round(1e6 * mq / (2 * n), 4),
            fpr_empirica=round(fpr, 6), fpr_teorica=round(bf.theoretical_fpr(), 6), falsos_negativos=fn)
    if p == 0.01:
        insl, ql = ins.tolist(), queries.tolist()
        def s_build(_):
            return set(insl)
        mean, std, reps, st = timeit(lambda: None, s_build, n)
        cur, peak = memory(lambda: None, lambda _: set(ins.tolist()))   # incluye los objetos int
        mq, sq, rq, _ = timeit(lambda: None, lambda _: [x in st for x in ql], n)
        REC.add(estructura="set de Python (exacto)", operacion="insertar", n=n, p_objetivo=0, reps=reps,
                media_s=round(mean, 6), std_s=round(std, 6), us_por_op=round(1e6 * mean / n, 4),
                mem_actual_mb=round(cur, 4), mem_pico_mb=round(peak, 4))
        REC.add(estructura="set de Python (exacto)", operacion="consultar 2n", n=n, p_objetivo=0, reps=rq,
                media_s=round(mq, 6), std_s=round(sq, 6), us_por_op=round(1e6 * mq / (2 * n), 4),
                fpr_empirica=0.0, falsos_negativos=0)
        if n <= 1_000_000:          # modo escalar evento a evento (como en el pipeline)
            def sc(_):
                bf2 = BloomFilter(n, p)
                for x in insl:
                    bf2.add(x)
                return bf2
            mean, std, reps, bf2 = timeit(lambda: None, sc, n, max_reps=20)
            REC.add(estructura="Bloom Filter (escalar, Python)", operacion="insertar", n=n, p_objetivo=p,
                    reps=reps, media_s=round(mean, 6), std_s=round(std, 6), us_por_op=round(1e6 * mean / n, 4))


def bench_cms(n, eps=0.001, delta=0.01):
    rng = np.random.default_rng(SEED + n)
    U = max(100, n // 10)
    stream = ((rng.zipf(1.1, n) - 1) % U).astype(np.uint64)
    stream = (stream * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(12345)      # claves de 64 bits
    def c_build(_):
        c = CountMinSketch(eps, delta); c.update_many(stream); return c
    mean, std, reps, cms = timeit(lambda: None, c_build, n)
    cur, peak = memory(lambda: None, c_build)
    sl = stream.tolist()
    def d_build(_):
        return Counter(sl)
    mean2, std2, reps2, exact = timeit(lambda: None, d_build, n)
    cur2, peak2 = memory(lambda: None, lambda _: Counter(stream.tolist()))
    ks = np.fromiter(exact.keys(), dtype=np.uint64, count=len(exact))
    tv = np.fromiter(exact.values(), dtype=np.int64, count=len(exact))
    est = cms.estimate_many(ks)
    err = est - tv
    bound = cms.epsilon * n
    top = np.argsort(-tv)[:100]
    REC.add(estructura="Count-Min Sketch (vectorizado)", operacion="actualizar stream", n=n, reps=reps,
            media_s=round(mean, 6), std_s=round(std, 6), us_por_op=round(1e6 * mean / n, 4),
            mem_actual_mb=round(cur, 4), mem_pico_mb=round(peak, 4), w=cms.w, d=cms.d, claves_distintas=len(ks),
            subestimaciones=int((err < 0).sum()), error_abs_medio=round(float(err.mean()), 3),
            error_max=int(err.max()), cota_eps_N=round(bound, 1), frac_dentro_cota=round(float((err <= bound).mean()), 5),
            error_rel_top100=round(float((err[top] / tv[top]).mean()), 5),
            error_rel_medio=round(float((err / tv).mean()), 4))
    REC.add(estructura="Counter/dict (exacto)", operacion="actualizar stream", n=n, reps=reps2,
            media_s=round(mean2, 6), std_s=round(std2, 6), us_por_op=round(1e6 * mean2 / n, 4),
            mem_actual_mb=round(cur2, 4), mem_pico_mb=round(peak2, 4), claves_distintas=len(ks))
    if n <= 1_000_000:
        def sc(_):
            c = CountMinSketch(eps, delta)
            for x in sl:
                c.update(x)
            return c
        m3, s3, r3, _ = timeit(lambda: None, sc, n, max_reps=20)
        REC.add(estructura="Count-Min Sketch (escalar, Python)", operacion="actualizar stream", n=n, reps=r3,
                media_s=round(m3, 6), std_s=round(s3, 6), us_por_op=round(1e6 * m3 / n, 4))


if __name__ == "__main__":
    save_platform()
    sizes, only = sizes_from_argv()
    for n in sizes:
        print(f"[bloom] n={n:,}", flush=True); bench_bloom(n)
        print(f"[cms] n={n:,}", flush=True); bench_cms(n)
    if 1_000_000 in sizes:
        for p in (0.1, 0.001):
            print(f"[bloom p={p}]", flush=True); bench_bloom(1_000_000, p)
