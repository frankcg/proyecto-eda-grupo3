"""Pipeline completo de punta a punta: throughput y costo por etapa vs n eventos.

    python experiments/bench_pipeline.py [--max=1e7]
Además compara Louvain vs Label Propagation sobre el grafo de co-compra REAL
producido por el pipeline (mismo grafo para ambos).
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from harness import RESULTS, Recorder, save_platform, sizes_from_argv
from memrss import peak_rss_mb

from ecostream.graph.communities import community_stats, label_propagation, louvain, modularity, nmi
from ecostream.pipeline import EcoStreamPipeline

REC = Recorder("pipeline")


def run(n):
    B = 5000 if n <= 1_000_000 else 10_000
    refresh = max(5, n // (B * 20))
    t0 = time.perf_counter()
    p = EcoStreamPipeline(n_events=n, n_products=20_000, batch_size=B, refresh_every=refresh, verbose=n >= 1_000_000)
    p.run()
    total = time.perf_counter() - t0
    s = p.summary()
    rss = peak_rss_mb()
    row = dict(n=n, eventos_procesados=s["eventos_procesados"], reps=1, setup_s=round(p.setup_seconds, 2),
               run_s=round(p.run_seconds, 2), total_s=round(total, 2), throughput_ev_s=s["throughput_ev_s"],
               rss_pico_mb=round(rss, 1), ordenes_aprobadas=s["ordenes"].get("aprobado", 0),
               ordenes_bloqueadas=s["ordenes"].get("bloqueado", 0), alertas=s["alertas_fraude"],
               refrescos=p.batches // refresh + 1, micro_lote=B)
    for st in p.stage_table():
        key = st["etapa"].split(" ")[0]
        row[f"{key}_s"] = st["segundos"]
        row[f"{key}_us"] = st.get("us_por_item", "")
    row["espera_ticks"] = s["espera_promedio_ticks"]
    REC.add(**row)
    if n == 1_000_000 or (n < 1_000_000 and n == max(sizes)):
        g = p.s8.g
        act = g.degree > 0
        out = {"nodos_activos": int(act.sum()), "aristas": int(g.m), "ordenes": p.s8.last_orders}
        for name, fn in [("Louvain", lambda sd: louvain(g, seed=sd)[0]), ("Label Propagation", lambda sd: label_propagation(g, seed=sd)[0])]:
            labs, ts = [], []
            for sd in range(10):
                t = time.perf_counter(); labs.append(fn(sd)); ts.append(time.perf_counter() - t)
            st = community_stats(labs[0][act])
            out[name] = {"modularidad": float(np.mean([modularity(g, l) for l in labs])),
                         "modularidad_std": float(np.std([modularity(g, l) for l in labs], ddof=1)),
                         "tiempo_s": float(np.mean(ts)), "tiempo_std": float(np.std(ts, ddof=1)),
                         "comunidades": st["n_communities"], "mayor": st["largest"], "mediana": st["median_size"],
                         "singletons": st["singletons"],
                         "estabilidad_nmi": float(np.mean([nmi(a[act], b[act]) for a, b in itertools.combinations(labs, 2)])),
                         "tamanos_top10": [int(x) for x in st["sizes"][:10]]}
        with open(RESULTS / "comunidades_grafo_pipeline.json", "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    save_platform()
    sizes, _ = sizes_from_argv()
    for n in sizes:
        if n < 10_000:
            continue
        print(f"[pipeline] n={n:,}", flush=True)
        run(n)
