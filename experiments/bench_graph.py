"""Etapa 8 [FIJA]: PageRank vs ranking por grado; Louvain vs Label Propagation.

    python experiments/bench_graph.py [--max=1e7] [--only=pagerank,communities]
El tamaño n es el número de interacciones de co-compra (aristas antes de
agregar duplicados) de un grafo con comunidades plantadas y grados Zipf;
nodos = n / 8.
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np

from harness import Recorder, memory, save_platform, sizes_from_argv, timeit, SEED
from memrss import memory_rss

from ecostream.graph.communities import community_stats, label_propagation, louvain, modularity, nmi
from ecostream.graph.graph import planted_partition_graph
from ecostream.graph.pagerank import degree_ranking, pagerank

REC = Recorder("grafo")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def graph(n):
    return planted_partition_graph(max(50, n // 8), n, p_in=0.7, seed=SEED + n)


def bench_pagerank(n):
    g, truth = graph(n)
    mean, std, reps, (pr, iters) = timeit(lambda: None, lambda _: pagerank(g), n)
    md, sd, rd, deg = timeit(lambda: None, lambda _: degree_ranking(g), n)
    mem = memory_rss if n >= 10_000_000 else memory
    cur, peak = mem(lambda: None, lambda _: pagerank(g))
    K = min(100, g.n // 10)
    tp, td = set(np.argsort(-pr)[:K].tolist()), set(np.argsort(-deg)[:K].tolist())
    # ¿PageRank premia nodos "puente"? fracción de vecinos fuera de su comunidad en el top-K
    def bridging(top):
        out = []
        for i in top:
            nb, _ = g.neighbors(i)
            if len(nb):
                out.append(float((truth[nb] != truth[i]).mean()))
        return float(np.mean(out)) if out else 0.0
    REC.add(algoritmo="PageRank (iteración de potencias)", n=n, nodos=g.n, aristas=g.m, reps=reps,
            media_s=round(mean, 6), std_s=round(std, 6), iteraciones=iters, us_por_arista_iter=round(1e6 * mean / (g.m * iters), 4),
            mem_actual_mb=round(cur, 3), mem_pico_mb=round(peak, 3), spearman_vs_grado=round(spearman(pr, deg), 4),
            solapamiento_topK=round(len(tp & td) / K, 3), K=K, puente_topK=round(bridging(tp), 4))
    REC.add(algoritmo="Ranking por grado ponderado", n=n, nodos=g.n, aristas=g.m, reps=rd, media_s=round(md, 6),
            std_s=round(sd, 6), spearman_vs_grado=1.0, solapamiento_topK=1.0, K=K, puente_topK=round(bridging(td), 4))


def bench_communities(n):
    g, truth = graph(n)
    seeds = 10 if n <= 100_000 else (5 if n <= 1_000_000 else 2)
    act = g.degree > 0
    for name, fn in [("Louvain", lambda s: louvain(g, seed=s)[0]), ("Label Propagation", lambda s: label_propagation(g, seed=s)[0])]:
        times, labs = [], []
        for s in range(seeds):
            t0 = time.perf_counter()
            lab = fn(s)
            times.append(time.perf_counter() - t0)
            labs.append(lab)
        mem = memory_rss if n >= 1_000_000 else memory
        cur, peak = mem(lambda: None, lambda _: fn(0))
        qs = [modularity(g, l) for l in labs]
        st = community_stats(labs[0][act])
        pair = [nmi(a[act], b[act]) for a, b in itertools.combinations(labs, 2)]
        sizes = st["sizes"]
        REC.add(algoritmo=name, n=n, nodos=g.n, aristas=g.m, reps=seeds, media_s=round(float(np.mean(times)), 5),
                std_s=round(float(np.std(times, ddof=1)), 5), modularidad=round(float(np.mean(qs)), 5),
                modularidad_std=round(float(np.std(qs, ddof=1)), 5), comunidades=st["n_communities"],
                comunidades_std=round(float(np.std([community_stats(l[act])["n_communities"] for l in labs], ddof=1)), 2),
                mayor=st["largest"], mediana=st["median_size"], singletons=st["singletons"],
                frac_mayor=round(st["largest"] / act.sum(), 4), estabilidad_nmi=round(float(np.mean(pair)), 4),
                nmi_vs_plantadas=round(float(np.mean([nmi(l[act], truth[act]) for l in labs])), 4),
                mem_actual_mb=round(cur, 2), mem_pico_mb=round(peak, 2),
                tamanos_top10=[int(x) for x in sizes[:10]])


if __name__ == "__main__":
    save_platform()
    sizes, only = sizes_from_argv()
    cmax = int(float(os.environ.get("COMM_MAX", 10_000_000)))
    for n in sizes:
        if not only or "pagerank" in only:
            print(f"[pagerank] n={n:,}", flush=True); bench_pagerank(n)
        if (not only or "communities" in only) and n <= cmax:
            print(f"[communities] n={n:,}", flush=True); bench_communities(n)
