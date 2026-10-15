"""Genera las tablas LaTeX del informe directamente desde experiments/results/*.csv."""
import json
import re
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parents[1] / "docs" / "informe"


def fmt_t(s):
    s = float(s)
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.0f} µs"


def fmt_mb(x):
    x = float(x)
    if x != x:
        return "--"
    if x >= 1024:
        return f"{x / 1024:.2f} GB"
    if x >= 1:
        return f"{x:.1f} MB"
    return f"{x * 1024:.0f} KB"


def n_lbl(n):
    n = int(n)
    return {10**3: "$10^3$", 10**4: "$10^4$", 10**5: "$10^5$", 10**6: "$10^6$", 10**7: "$10^7$"}.get(n, f"{n:,}")


def write(name, s):
    s = re.sub(r"(\d),(\d{3})", r"\1\\,\2", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\\,\2", s)
    (OUT / name).write_text(s, encoding="utf-8")
    print("ok", name)


# ---------------- pipeline
pl = pd.read_csv(R / "pipeline.csv").sort_values("n")
rows = []
for _, r in pl.iterrows():
    st = " & ".join(f"{float(r[f'E{i}_us']):.0f}" if str(r.get(f"E{i}_us", "")) not in ("", "nan") else "--" for i in [0, 1, 2, 3, 4, 5, 6])
    rows.append(f"{n_lbl(r['n'])} & {int(r['eventos_procesados']):,} & {float(r['run_s']):.0f} & {int(r['throughput_ev_s']):,} & "
                f"{float(r['rss_pico_mb']) / 1024:.1f} & {int(r['ordenes_aprobadas']):,} / {int(r['ordenes_bloqueadas'])} & {int(r['alertas'])} & {st} \\\\")
write("tabla_pipeline.tex", r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rrrrrrrrrrrrrr@{}}\toprule
Eventos & Procesados & s & ev/s & RSS GB & Órd. aprob./bloq. & Alertas & E0 & E1 & E2 & E3 & E4 & E5 & E6 \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}""")

# ---------------- probabilísticas
p = pd.read_csv(R / "probabilisticas.csv")
rows = []
for n in [10**5, 10**6, 10**7]:
    bi = p[(p.estructura == "Bloom Filter (vectorizado)") & (p.operacion == "insertar") & (p.n == n)]
    bq = p[(p.estructura == "Bloom Filter (vectorizado)") & (p.operacion == "consultar 2n") & (p.n == n)]
    si = p[(p.estructura == "set de Python (exacto)") & (p.operacion == "insertar") & (p.n == n)]
    sq = p[(p.estructura == "set de Python (exacto)") & (p.operacion == "consultar 2n") & (p.n == n)]
    if len(bi) and len(si):
        bi, bq, si, sq = bi.iloc[0], bq.iloc[0], si.iloc[0], sq.iloc[0]
        rows.append(f"Bloom ($p$=1\\,\\%) vs \\texttt{{set}} & {n_lbl(n)} & {fmt_t(bi.media_s)} / {fmt_t(si.media_s)} & "
                    f"{fmt_t(bq.media_s)} / {fmt_t(sq.media_s)} & {fmt_mb(bi.mem_actual_mb)} / {fmt_mb(si.mem_actual_mb)} & "
                    f"FPR {100 * float(bq.fpr_empirica):.3f}\\,\\% (teórica {100 * float(bq.fpr_teorica):.3f}\\,\\%), FN = 0 \\\\")
for n in [10**5, 10**6, 10**7]:
    c = p[(p.estructura == "Count-Min Sketch (vectorizado)") & (p.n == n)]
    d = p[(p.estructura == "Counter/dict (exacto)") & (p.n == n)]
    if len(c) and len(d):
        c, d = c.iloc[0], d.iloc[0]
        rows.append(f"CMS ($\\varepsilon$=0.001) vs \\texttt{{Counter}} & {n_lbl(n)} & {fmt_t(c.media_s)} / {fmt_t(d.media_s)} & -- & "
                    f"{fmt_mb(c.mem_actual_mb)} / {fmt_mb(d.mem_actual_mb)} & err. medio {float(c.error_abs_medio):.0f}, máx {int(float(c.error_max))} "
                    f"($\\varepsilon N$={float(c.cota_eps_N):.0f}); {100 * float(c.frac_dentro_cota):.0f}\\,\\% en cota; subestim. 0 \\\\")
write("tabla_prob.tex", r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llllll@{}}\toprule
Comparación & $n$ & Construir (aprox./exacto) & Consultar $2n$ & Memoria (aprox./exacto) & Calidad \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}""")

# ---------------- similitud
s = pd.read_csv(R / "similitud.csv")
rows = []
for n in [10**4, 10**5, 10**6, 10**7]:
    lq = s[s.estructura.str.startswith("MinHash") & (s.operacion == "consulta top-10") & (s.n == n)]
    lb = s[s.estructura.str.startswith("MinHash") & (s.operacion == "construir indice") & (s.n == n)]
    bf = s[s.estructura.str.startswith("Fuerza bruta") & (s.n == n)]
    if len(lq) and len(bf):
        lq, lb, bf = lq.iloc[0], lb.iloc[0], bf.iloc[0]
        rows.append(f"MinHash+LSH (32$\\times$2) & {n_lbl(n)} & {float(lq.us_por_op) / 1e3:.2f} ms & {float(bf.us_por_op) / 1e3:,.1f} ms & "
                    f"{float(bf.us_por_op) / float(lq.us_por_op):,.0f}$\\times$ & {float(lq.recall_J05):.2f} (J$\\ge$0.5) & {fmt_mb(lb.mem_actual_mb)} / {fmt_mb(bf.mem_actual_mb)} & {fmt_t(lb.media_s)} \\\\")
for n in [10**4, 10**5, 10**6]:
    hq = s[s.estructura.str.startswith("HNSW") & (s.operacion == "consulta k=10 ef=50") & (s.n == n)]
    hb = s[s.estructura.str.startswith("HNSW") & (s.operacion == "construir") & (s.n == n)]
    ex = s[(s.estructura == "KNN exacto fuerza bruta (NumPy)") & (s.n == n)]
    if len(hq) and len(ex):
        hq, hb, ex = hq.iloc[0], hb.iloc[0], ex.iloc[0]
        rows.append(f"HNSW ($M$=16, $ef$=50) & {n_lbl(n)} & {float(hq.us_por_op) / 1e3:.2f} ms & {float(ex.us_por_op) / 1e3:,.1f} ms & "
                    f"{float(ex.us_por_op) / float(hq.us_por_op):,.1f}$\\times$ & {float(hq.recall_10):.3f} (@10) & {fmt_mb(hb.mem_actual_mb)} / {fmt_mb(ex.mem_actual_mb)} & {fmt_t(hb.media_s)} \\\\")
write("tabla_sim.tex", (r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrlll@{}}\toprule
Técnica & $n$ & Consulta aprox. & Exacto & Aceleración & Recall & Memoria (índice/datos) & Construcción \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}"""))

# ---------------- comunidades
g = pd.read_csv(R / "grafo.csv")
rows = []
for n in [10**4, 10**5, 10**6, 10**7]:
    for alg, ab in [("Louvain", "Louvain"), ("Label Propagation", "LPA")]:
        r = g[(g.algoritmo == alg) & (g.n == n)]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append(f"{n_lbl(n)} ({int(r.aristas):,} ar.) & {ab} & {float(r.modularidad):.4f} $\\pm$ {float(r.modularidad_std):.4f} & {fmt_t(r.media_s)} & "
                    f"{int(r.comunidades):,} $\\pm$ {float(r.comunidades_std):.1f} & {int(r.mayor):,} / {float(r.mediana):.0f} & {float(r.estabilidad_nmi):.3f} & {float(r.nmi_vs_plantadas):.3f} \\\\")
cj = json.load(open(R / "comunidades_grafo_pipeline.json"))
for alg, ab in [("Louvain", "Louvain"), ("Label Propagation", "LPA")]:
    r = cj[alg]
    rows.append(f"Co-compra real ({cj['aristas']:,} ar.) & {ab} & {r['modularidad']:.4f} $\\pm$ {r['modularidad_std']:.4f} & {fmt_t(r['tiempo_s'])} & "
                f"{r['comunidades']:,} & {r['mayor']:,} / {r['mediana']:.0f} & {r['estabilidad_nmi']:.3f} & -- \\\\")
write("tabla_com.tex", (r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llllllll@{}}\toprule
Grafo ($n$ interacciones) & Alg. & Modularidad $Q$ & Tiempo & Nº comunidades & Mayor / mediana & Estabilidad (NMI) & NMI vs verdad \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}"""))

# ---------------- pendientes
sl = pd.read_csv(R / "pendientes.csv")
half = (len(sl) + 1) // 2
L = []
for i in range(half):
    a = sl.iloc[i]
    b = sl.iloc[i + half] if i + half < len(sl) else None
    left = f"{a['estructura/operación']} & {a['pendiente_ajustada']:.2f} & {a['esperada'].split(' ')[0]}"
    right = f"{b['estructura/operación']} & {b['pendiente_ajustada']:.2f} & {b['esperada'].split(' ')[0]}" if b is not None else "& &"
    L.append(left + " & " + right + r" \\")
write("tabla_pendientes.tex", r"""\setlength{\tabcolsep}{3.5pt}\footnotesize
\begin{tabular}{@{}lrl|lrl@{}}\toprule
Estructura / operación & $b$ & Esperada & Estructura / operación & $b$ & Esperada \\\midrule
""" + "\n".join(L).replace("≈", "$\\approx$").replace("²", "$^2$") + r"""
\bottomrule\end{tabular}""")
