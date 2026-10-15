"""Gráficas log-log de escalabilidad y tabla de pendientes ajustadas.

    python experiments/plots.py
Lee experiments/results/*.csv y escribe experiments/figures/*.png y
experiments/results/pendientes.csv (exponente b de t ~ a·n^b por ajuste de
mínimos cuadrados en log-log, para contrastar con la complejidad esperada).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)

# Paleta categórica validada (orden fijo): azul, naranja, aqua, amarillo, magenta, verde
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARK = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 2,
                     "lines.markersize": 5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

SLOPES = []


def load(name):
    p = R / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in df.columns:
        if c not in ("estructura", "operacion", "algoritmo", "tamanos_top10", "espera_ticks"):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= df[c].notna().sum() * 0.5:
                df[c] = conv
    return df


def fit(n, t):
    n, t = np.asarray(n, float), np.asarray(t, float)
    m = (n > 0) & (t > 0) & np.isfinite(t)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log10(n[m]), np.log10(t[m]), 1)[0])


def series(ax, df, items, ycol="media_s", yerr="std_s", slope_label=True):
    for i, (label, sel, expected) in enumerate(items):
        d = df[sel].sort_values("n")
        if d.empty:
            continue
        y = d[ycol].astype(float)
        err = d[yerr].astype(float) if yerr in d else None
        ax.errorbar(d["n"], y, yerr=err, color=C[i % 6], marker=MARK[i % 6], capsize=2, elinewidth=1,
                    label=f"{label}" + (f"  (pendiente {fit(d['n'], y):.2f})" if slope_label else ""))
        if expected:
            SLOPES.append({"estructura/operación": label, "metrica": ycol, "pendiente_ajustada": round(fit(d["n"], y), 3),
                           "esperada": expected, "n_min": int(d["n"].min()), "n_max": int(d["n"].max())})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (elementos)")


def ref_lines(ax, n0, t0, kinds=("n", "nlogn")):
    xs = np.array([1e3, 1e7])
    for k in kinds:
        if k == "n":
            ys = t0 * xs / n0
            ax.plot(xs, ys, ls=":", lw=1, color=MUTED)
            ax.text(xs[-1], ys[-1], " O(n)", color=MUTED, fontsize=7, va="center")
        if k == "nlogn":
            ys = t0 * xs * np.log2(xs) / (n0 * np.log2(n0))
            ax.plot(xs, ys, ls="--", lw=1, color=MUTED)
            ax.text(xs[-1], ys[-1], " O(n log n)", color=MUTED, fontsize=7, va="center")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_base():
    df = load("base")
    if df.empty:
        return
    E = df["estructura"]; O = df["operacion"]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2))
    ax = axs[0, 0]
    series(ax, df, [("LinkedQueue propia", (E == "LinkedQueue (propia)") & (O == "enqueue+dequeue"), "1 (O(n) total, O(1)/op)"),
                    ("deque (C)", (E == "collections.deque (ref.)"), "1"),
                    ("list.pop(0)", E == "list.pop(0) (ingenua)", "2 (O(n²))")])
    ax.set_title("E0 · FIFO: n enqueue + n dequeue"); ax.set_ylabel("tiempo total (s)")
    ax = axs[0, 1]
    series(ax, df, [("Merge Sort propio", E == "Merge Sort (propio)", "≈1.05-1.1 (n log n)"),
                    ("Timsort (C)", E == "sorted/Timsort (ref. C)", "≈1.05-1.1 (n log n)")])
    ax.set_title("E1 · Ordenación"); ax.set_ylabel("tiempo (s)")
    ax = axs[0, 2]
    series(ax, df, [("BinaryHeap propio", (E == "BinaryHeap (propio)") & (O == "push+pop"), "≈1.05-1.1 (n log n)"),
                    ("heapq (C)", E == "heapq (ref. C)", "≈1.05-1.1"),
                    ("Lista ordenada insort", E == "Lista ordenada (insort)", "2 (O(n²) memmove)")])
    ax.set_title("E2 · Heap: n push + n pop"); ax.set_ylabel("tiempo total (s)")
    ax = axs[1, 0]
    series(ax, df, [("HashTable propia", E == "HashTable sondeo lineal (propia)", "1 (O(1)/op)"),
                    ("dict (C)", E == "dict (ref. C)", "1"),
                    ("LRU propia (Zipf)", E == "LRUCache (propia)", "1")])
    ax.set_title("E3 · Hash table y LRU: n operaciones"); ax.set_ylabel("tiempo total (s)")
    ax = axs[1, 1]
    series(ax, df, [("B+Tree insertar", (E.str.startswith("B+Tree")) & (O == "insertar"), "≈1.05-1.1 (n log_B n)"),
                    ("B+Tree bulk load", (E.str.startswith("B+Tree")) & (O == "bulk load ordenado"), "1 (O(n))"),
                    ("Arreglo + insort", E == "Arreglo ordenado + insort", "2 (O(n²))")])
    ax.set_title("E4 · B+Tree (páginas 4 KB)"); ax.set_ylabel("tiempo total (s)")
    ax = axs[1, 2]
    series(ax, df, [("KD-Tree 4D consulta", (E == "KD-Tree 4D (propio)") & (O == "KNN k=10"), "≈0-0.3 (O(log n) esperado)"),
                    ("Fuerza bruta consulta", E == "KNN fuerza bruta (NumPy)", "1 (O(n))"),
                    ("Trie autocompletar", (E == "Trie top-K cacheado (propio)") & (O == "autocompletar k=8"), "0 (O(|p|+K))"),
                    ("Escaneo lineal autocompletar", E == "Escaneo lineal (ingenuo)", "1")], ycol="us_por_op", yerr="__")
    ax.set_title("E5/E9 · Latencia por consulta"); ax.set_ylabel("µs por consulta")
    for a in axs.ravel():
        a.legend(loc="best")
    save(fig, "fig_base_loglog.png")
    # memoria
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    items = [("LinkedQueue", (E == "LinkedQueue (propia)") & (O == "n encolados (memoria)")),
             ("BinaryHeap", (E == "BinaryHeap (propio)") & (O == "n en cola (memoria)")),
             ("HashTable", E == "HashTable sondeo lineal (propia)"),
             ("B+Tree (pool 1024 pág.)", (E.str.startswith("B+Tree")) & (O == "insertar")),
             ("KD-Tree 4D", (E == "KD-Tree 4D (propio)") & (O == "construir")),
             ("Trie", (E == "Trie top-K cacheado (propio)") & (O == "insertar"))]
    for i, (lab, sel) in enumerate(items):
        d = df[sel].sort_values("n")
        if not d.empty:
            ax.plot(d["n"], d["mem_pico_mb"].astype(float), color=C[i], marker=MARK[i], label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (elementos)"); ax.set_ylabel("memoria pico (MB)")
    ax.set_title("Memoria pico (tracemalloc ≤1M; RSS en 10M)"); ax.legend()
    save(fig, "fig_base_memoria.png")


def fig_prob():
    df = load("probabilisticas")
    if df.empty:
        return
    E, O = df["estructura"], df["operacion"]
    p1 = (df["p_objetivo"].astype(float).isin([0.01, 0.0])) if "p_objetivo" in df else True
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax = axs[0]
    series(ax, df, [("Bloom insertar", (E == "Bloom Filter (vectorizado)") & (O == "insertar") & p1, "1"),
                    ("set insertar", (E == "set de Python (exacto)") & (O == "insertar"), "1"),
                    ("Bloom consultar 2n", (E == "Bloom Filter (vectorizado)") & (O == "consultar 2n") & p1, "1"),
                    ("set consultar 2n", (E == "set de Python (exacto)") & (O == "consultar 2n"), "1")])
    ax.set_title("Bloom vs set: tiempo"); ax.set_ylabel("tiempo (s)"); ax.legend(loc="upper left")
    ax = axs[1]
    items = [("Bloom (p=1 %)", (E == "Bloom Filter (vectorizado)") & (O == "insertar") & p1),
             ("set exacto", (E == "set de Python (exacto)") & (O == "insertar")),
             ("CMS (ε=0.001, δ=0.01)", E == "Count-Min Sketch (vectorizado)"),
             ("Counter exacto", E == "Counter/dict (exacto)")]
    for i, (lab, sel) in enumerate(items):
        d = df[sel].sort_values("n")
        ax.plot(d["n"], d["mem_actual_mb"].astype(float), color=C[i], marker=MARK[i], label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (elementos)"); ax.set_ylabel("memoria retenida (MB)")
    ax.set_title("Memoria: probabilística vs exacta"); ax.legend(loc="upper left")
    ax = axs[2]
    d = df[E.str.startswith("Bloom Filter (vectorizado)") & (O == "consultar 2n")].copy()
    d["p_objetivo"] = d["p_objetivo"].astype(float)
    for i, p in enumerate(sorted(d["p_objetivo"].unique(), reverse=True)):
        dd = d[d["p_objetivo"] == p].sort_values("n")
        ax.plot(dd["n"], dd["fpr_empirica"].astype(float), color=C[i], marker=MARK[i], label=f"empírica p={p:g}")
        ax.plot(dd["n"], dd["fpr_teorica"].astype(float), color=C[i], ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n (elementos)"); ax.set_ylabel("tasa de falsos positivos")
    ax.set_title("Bloom: FPR empírica (línea) vs teórica (punteada)"); ax.legend()
    save(fig, "fig_probabilisticas.png")


def fig_similarity():
    df = load("similitud")
    if df.empty:
        return
    E, O = df["estructura"], df["operacion"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax = axs[0]
    series(ax, df, [("MinHash+LSH consulta", E.str.startswith("MinHash") & (O == "consulta top-10"), "≈0 (sublineal)"),
                    ("Fuerza bruta Jaccard", E.str.startswith("Fuerza bruta"), "1 (O(n))"),
                    ("HNSW consulta (ef=50)", E.str.startswith("HNSW") & (O == "consulta k=10 ef=50"), "≈0 (O(log n))"),
                    ("KNN exacto fuerza bruta", E == "KNN exacto fuerza bruta (NumPy)", "1"),
                    ("KD-Tree 32D", E == "KD-Tree 32D (exacto)", "≈1 (maldición dimensional)")], ycol="us_por_op", yerr="__")
    ax.set_title("Latencia por consulta (top-10)"); ax.set_ylabel("µs por consulta"); ax.legend(loc="upper left")
    ax = axs[1]
    series(ax, df, [("MinHash+LSH construir", E.str.startswith("MinHash") & (O == "construir indice"), "1 (O(n·k))"),
                    ("HNSW construir", E.str.startswith("HNSW") & (O == "construir"), "≈1.1 (O(n log n))")])
    ax.set_title("Construcción del índice"); ax.set_ylabel("tiempo (s)"); ax.legend(loc="upper left")
    ax = axs[2]
    d = df[E.str.startswith("HNSW") & O.str.startswith("consulta") & (df["n"] == 100000)].copy()
    if not d.empty:
        d["ef"] = d["operacion"].str.extract(r"ef=(\d+)").astype(int)
        d = d.sort_values("ef")
        ax.plot(d["us_por_op"].astype(float), d["recall_10"].astype(float), color=C[0], marker="o", label="HNSW n=100K")
        for _, r in d.iterrows():
            ax.annotate(f"ef={r['ef']}", (float(r["us_por_op"]), float(r["recall_10"])), fontsize=7, color=MUTED,
                        xytext=(4, -9), textcoords="offset points")
        bf = df[(E == "KNN exacto fuerza bruta (NumPy)") & (df["n"] == 100000)]
        if not bf.empty:
            ax.scatter([float(bf["us_por_op"].iloc[0])], [1.0], color=C[1], marker="s", s=40, zorder=3, label="exacto (fuerza bruta)")
    cur = df[E.str.startswith("LSH curva")]
    ax.set_xscale("log"); ax.set_xlabel("µs por consulta"); ax.set_ylabel("recall@10")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("HNSW: recall vs latencia (barrido ef)"); ax.legend(loc="lower right")
    save(fig, "fig_similitud.png")
    if not cur.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        J = np.linspace(0, 1, 200)
        for i, (_, r) in enumerate(cur.iterrows()):
            b, rr = [int(x) for x in r["estructura"].replace("LSH curva b=", "").split(" r=")]
            ax.plot(J, 1 - (1 - J ** rr) ** b, color=C[i % 6], label=f"b={b}, r={rr}: recall(J≥0.5)={float(r['recall_J05']):.2f}, cand={float(r['candidatos_prom']):.0f}")
        ax.axvline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("similitud de Jaccard J"); ax.set_ylabel("P[ser candidato]")
        ax.set_title("Curva S de LSH: teoría y recall medido (n=100K)"); ax.legend(fontsize=6.5, loc="lower right")
        ax.set_xscale("linear"); ax.set_yscale("linear")
        save(fig, "fig_lsh_curva.png")


def fig_graph():
    df = load("grafo")
    if df.empty:
        return
    A = df["algoritmo"]
    df["n"] = df["aristas"].astype(float)       # escala real: aristas únicas
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax = axs[0]
    series(ax, df, [("PageRank", A.str.startswith("PageRank"), "1 (O(m) por iteración)"),
                    ("Louvain", A == "Louvain", "≈1-1.2 (O(m log n))"),
                    ("Label Propagation", A == "Label Propagation", "≈1 (O(m) por iteración)"),
                    ("Grado ponderado", A.str.startswith("Ranking"), "0-1 (precomputado)")])
    ax.set_xlabel("aristas m"); ax.set_title("Grafo: tiempo vs tamaño"); ax.set_ylabel("tiempo (s)"); ax.legend(loc="upper left")
    ax = axs[1]
    for i, name in enumerate(["Louvain", "Label Propagation"]):
        d = df[A == name].sort_values("n")
        ax.errorbar(d["n"], d["modularidad"].astype(float), yerr=d["modularidad_std"].astype(float), color=C[i + 1],
                    marker=MARK[i + 1], capsize=2, label=name)
    ax.set_xscale("log"); ax.set_xlabel("aristas m"); ax.set_ylabel("modularidad Q"); ax.set_title("Modularidad (media ± std)")
    ax.legend()
    ax = axs[2]
    for i, name in enumerate(["Louvain", "Label Propagation"]):
        d = df[A == name].sort_values("n")
        ax.plot(d["n"], d["estabilidad_nmi"].astype(float), color=C[i + 1], marker=MARK[i + 1], label=f"{name}: estabilidad")
        ax.plot(d["n"], d["nmi_vs_plantadas"].astype(float), color=C[i + 1], ls=":", marker=MARK[i + 1], label=f"{name}: NMI vs verdad")
    ax.set_xscale("log"); ax.set_xlabel("aristas m"); ax.set_ylabel("NMI"); ax.set_title("Estabilidad entre semillas y recuperación")
    ax.legend(fontsize=6.5)
    save(fig, "fig_grafo.png")


def fig_pipeline():
    df = load("pipeline")
    if df.empty:
        return
    df = df.sort_values("n")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2))
    ax = axs[0]
    ax.plot(df["n"], df["throughput_ev_s"].astype(float), color=C[0], marker="o")
    ax.set_xscale("log"); ax.set_xlabel("eventos en el stream"); ax.set_ylabel("eventos/s"); ax.set_title("Throughput de punta a punta")
    ax.set_ylim(bottom=0)
    ax = axs[1]
    last = df.iloc[-1]
    st = [c[:-3] for c in df.columns if c.endswith("_us") and str(last[c]) not in ("", "nan")]
    vals = [float(last[f"{s}_us"]) for s in st]
    ax.barh(st[::-1], vals[::-1], color=C[0], height=0.6)
    ax.set_xscale("log"); ax.set_xlabel("µs por elemento procesado"); ax.set_title(f"Costo por etapa (n={int(last['n']):,})")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_pipeline.png")



def fig_similarity_ab():
    df = load("similitud")
    if df.empty:
        return
    E, O = df["estructura"], df["operacion"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axs[0]
    series(ax, df, [("MinHash+LSH", E.str.startswith("MinHash") & (O == "consulta top-10"), None),
                    ("Fuerza bruta Jaccard", E.str.startswith("Fuerza bruta"), None),
                    ("HNSW (ef=50)", E.str.startswith("HNSW") & (O == "consulta k=10 ef=50"), None),
                    ("KNN exacto", E == "KNN exacto fuerza bruta (NumPy)", None),
                    ("KD-Tree 32D", E == "KD-Tree 32D (exacto)", None)], ycol="us_por_op", yerr="__", slope_label=False)
    ax.set_title("Latencia por consulta (top-10)"); ax.set_ylabel("µs por consulta"); ax.legend(loc="upper left", fontsize=7)
    ax = axs[1]
    d = df[E.str.startswith("HNSW") & O.str.startswith("consulta") & (df["n"] == 100000)].copy()
    d["ef"] = d["operacion"].str.extract(r"ef=(\d+)").astype(int)
    d = d.sort_values("ef")
    ax.plot(d["us_por_op"].astype(float), d["recall_10"].astype(float), color=C[0], marker="o", label="HNSW n=10^5")
    for _, r in d.iterrows():
        ax.annotate(f"ef={r['ef']}", (float(r["us_por_op"]), float(r["recall_10"])), fontsize=7, color=MUTED,
                    xytext=(4, -10), textcoords="offset points")
    bf = df[(E == "KNN exacto fuerza bruta (NumPy)") & (df["n"] == 100000)]
    ax.scatter([float(bf["us_por_op"].iloc[0])], [1.0], color=C[1], marker="s", s=40, zorder=3, label="KNN exacto")
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("µs por consulta"); ax.set_ylabel("recall@10"); ax.set_title("HNSW: recall vs latencia")
    ax.legend(loc="lower right")
    save(fig, "fig_similitud_ab.png")


if __name__ == "__main__":
    fig_base(); fig_prob(); fig_similarity(); fig_similarity_ab(); fig_graph(); fig_pipeline()
    pd.DataFrame(SLOPES).to_csv(R / "pendientes.csv", index=False)
    print(pd.DataFrame(SLOPES).to_string())
