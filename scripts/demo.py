"""Demo end-to-end (3-4 min en la presentación).

    python scripts/demo.py --events 300000

Muestra: (1) recorrido de eventos concretos por las 10 etapas, (2)
prioridades, (3) filtrado probabilístico con verificación, (4) similitud
aproximada vs método exacto, (5) análisis del grafo y (6) servicio al usuario
final (autocompletado, recomendaciones Top-K, rango de precios, panel HTML).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecostream.dashboard import build_dashboard  # noqa: E402
from ecostream.data.generator import ET, StreamConfig, generate_stream, make_catalog  # noqa: E402
from ecostream.graph.communities import community_stats, label_propagation, louvain, modularity  # noqa: E402
from ecostream.graph.pagerank import degree_ranking  # noqa: E402
from ecostream.pipeline import EcoStreamPipeline  # noqa: E402


def pick_watch(n_events, n_products, seed):
    """Elige eventos representativos para trazar (misma semilla que el pipeline)."""
    cat = make_catalog(n_products, seed=seed)
    st = generate_stream(cat, StreamConfig(n_events=n_events, seed=seed))
    c = st.cols
    n = st.n
    mid = slice(n // 2, n)
    def first(mask):
        idx = np.where(mask)[0]
        idx = idx[idx >= n // 2] if (idx >= n // 2).any() else idx
        return int(c["event_id"][idx[0]]) if len(idx) else None
    bl = set(st.blacklist_tokens.tolist())
    express = first((c["event_type"] == ET["purchase"]) & (c["shipping"] == 1) & (c["dirty"] == 0)
                    & ~np.isin(c["payment_token"], list(bl)))
    view = first((c["event_type"] == ET["view"]) & (c["dirty"] == 0))
    # intento de pago nº 20 de una ráfaga (supera el umbral del CMS)
    burst = np.where(c["device_id"] >= 10**7)[0]
    fraud = None
    if len(burst):
        dev = c["device_id"][burst[len(burst) // 2]]
        same = burst[c["device_id"][burst] == dev]
        fraud = int(c["event_id"][same[min(19, len(same) - 1)]])
    dirty = first(c["dirty"] == 4)
    _, cnt = np.unique(c["event_id"], return_counts=True)
    dup_ids = np.unique(c["event_id"])[cnt > 1]
    dup = int(dup_ids[len(dup_ids) // 2]) if len(dup_ids) else None
    return {"compra_express": express, "vista": view, "pago_fraudulento": fraud,
            "registro_sucio": dirty, "duplicado": dup}


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=300_000)
    ap.add_argument("--products", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "docs"))
    a = ap.parse_args()

    watch = pick_watch(a.events, a.products, a.seed)
    section("0. Configuración")
    print(f"eventos={a.events:,}  productos={a.products:,}  micro-lote={a.batch}  semilla={a.seed}")
    print("eventos trazados:", watch)

    p = EcoStreamPipeline(a.events, a.products, a.batch, refresh_every=20, seed=a.seed,
                          watch={v for v in watch.values() if v is not None})
    print(f"inicialización (catálogo, hash, B+Tree bulk load, KD-Tree, HNSW, Trie): {p.setup_seconds:.1f}s")
    p.run()

    section("1. Recorrido de eventos por las 10 etapas")
    for name, eid in watch.items():
        if eid is None:
            continue
        print(f"\n▶ {name} (event_id={eid})")
        for stage, msg in p.tracer.log.get(eid, []):
            print(f"   [{stage}] {msg}")

    s = p.summary()
    section("2. Prioridades (E2): espera promedio en ticks por clase")
    for k, v in s["espera_promedio_ticks"].items():
        print(f"   {k:<24} {v:>7.3f}")
    print(f"   backlog máximo del heap: {s['max_backlog_heap']:,} eventos")

    section("3. Filtrado probabilístico (E6)")
    m = p.s6.metrics.counters
    print(f"   Bloom: m={p.s6.bloom.m:,} bits, k={p.s6.bloom.k}, {p.s6.bloom.memory_bytes()/1024:.1f} KB; "
          f"positivos={int(m['bloom_positivos'])}, verificados={int(m['bloom_verificados'])}, "
          f"falsos positivos={int(m['bloom_falsos_positivos'])} (dejados pasar tras verificar)")
    print(f"   CMS ventana 5 min: {p.s6.velocity.memory_bytes()/1024:.0f} KB; alertas de velocidad={len(p.s6.alerts)}")
    print(f"   Órdenes: {s['ordenes']}")
    for al in p.s6.alerts[:5]:
        print(f"     alerta → dispositivo {al['device_id']}: {al['reason']}")

    section("4. Similitud aproximada vs exacta (E7)")
    pid = int(max(p.s9.sales, key=p.s9.sales.get))
    print(f"   Producto consulta: {p.catalog.names[pid]} (id {pid})")
    t = time.perf_counter(); appr = p.s7.similar_behavior(pid, 10); ta = time.perf_counter() - t
    t = time.perf_counter(); exact = p.s7.similar_behavior(pid, 10, exact=True); te = time.perf_counter() - t
    inter = len({x for x, _ in appr} & {x for x, _ in exact})
    rel = {x for x, j in exact if j >= 0.2}
    print(f"   MinHash+LSH (b=32, r=2, umbral≈0.18): {ta*1e3:.2f} ms vs fuerza bruta Jaccard: {te*1e3:.2f} ms; "
          f"recall vecinos J≥0.2 = {len(rel & {x for x, _ in appr})}/{len(rel)}; recall@10 total = {inter}/{max(1, len(exact))} "
          f"(los demás tienen J<0.2, bajo el umbral de diseño)")
    for (x, j) in appr[:5]:
        print(f"      LSH  J={j:.2f}  {p.catalog.names[x]}")
    t = time.perf_counter(); ca = p.s7.similar_content(pid, 10); ta = time.perf_counter() - t
    t = time.perf_counter(); ce = p.s7.similar_content(pid, 10, exact=True); te = time.perf_counter() - t
    inter = len({x for x, _ in ca} & {x for x, _ in ce})
    print(f"   HNSW: {ta*1e3:.2f} ms vs KNN exacto: {te*1e3:.2f} ms; recall@10 = {inter}/10")
    for (x, c) in ca[:5]:
        print(f"      HNSW cos={c:.2f}  {p.catalog.names[x]}")
    print(f"   KD-Tree (E5) verificado contra fuerza bruta: {p.s5.verify(pid)}")

    section("5. Grafo de co-compra (E8)")
    g = p.s8.g
    print(f"   ventana 7 días: {p.s8.last_orders:,} órdenes aprobadas → {g.n:,} nodos, {g.m:,} aristas")
    deg = degree_ranking(g)
    top_pr = np.argsort(-p.s8.pr)[:10]; top_dg = np.argsort(-deg)[:10]
    print(f"   Top-10 PageRank vs Top-10 grado: {len(set(top_pr) & set(top_dg))}/10 en común")
    for i, q in enumerate(top_pr[:5]):
        print(f"      #{i+1} PR={p.s8.pr[q]:.5f} grado={int(deg[q])}  {p.catalog.names[q]}")
    t = time.perf_counter(); lv, _ = louvain(g, seed=0); tl = time.perf_counter() - t
    t = time.perf_counter(); lp, _ = label_propagation(g, seed=0); tp = time.perf_counter() - t
    act = g.degree > 0
    print(f"   Louvain: Q={modularity(g, lv):.4f}, {community_stats(lv[act])['n_communities']} comunidades, {tl:.2f}s")
    print(f"   Label Propagation: Q={modularity(g, lp):.4f}, {community_stats(lp[act])['n_communities']} comunidades, {tp:.2f}s")
    from ecostream.data.generator import CATEGORIES
    sizes = np.bincount(lv, weights=act.astype(float))
    print("   Comunidades Louvain más grandes (categorías dominantes):")
    for c in np.argsort(-sizes)[:5]:
        mem = np.where((lv == c) & act)[0]
        cats = np.bincount(p.catalog.category[mem], minlength=len(CATEGORIES))
        top = ", ".join(f"{CATEGORIES[i][1]} ({cats[i]})" for i in np.argsort(-cats)[:4] if cats[i] > 0)
        print(f"      C{c}: {len(mem)} productos → {top}")

    section("6. Servicio al usuario final (E9)")
    for pref in ["smart", "lap", "zapa"]:
        print(f"   autocompletar '{pref}': " + " | ".join(n for _, n, _ in p.s9.autocomplete(pref, 4)))
    buyers = [u for u, prof in p.s3.users.items() if len(prof["purchased"]) >= 2]
    uid = buyers[0]
    prof = p.s3.user_profile(uid)
    print(f"   Usuario {uid}: compró {[p.catalog.names[x] for x in prof['purchased'][-3:]]}")
    for sc, q, why in p.s9.recommend(uid, 5, explain=True):
        print(f"      rec {sc:.3f} [{'+'.join(sorted(set(why)))}] {p.catalog.names[q]}")
    print("   Laptops S/ 2000-3000 por influencia (B+Tree rango + PR): " +
          ", ".join(p.catalog.names[q] for _, q in p.s9.price_range(2000, 3000, "computers.notebook", 3)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dash = build_dashboard(p, str(out / "dashboard.html"))
    print(f"   Panel HTML: {dash}")

    section("7. Resumen")
    print(json.dumps({k: v for k, v in s.items() if k not in ("generador",)}, indent=1, ensure_ascii=False, default=str))
    for r in p.stage_table():
        print("  ", r)
    with open(out / "demo_resumen.json", "w", encoding="utf-8") as f:
        json.dump({"summary": s, "stages": p.stage_table(), "watch": watch,
                   "traces": {str(k): v for k, v in p.tracer.log.items()}}, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
