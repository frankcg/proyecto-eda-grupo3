"""Etapa 9 - Panel de ventas, actividad, alertas y métricas del pipeline (HTML estático)."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data.generator import CATEGORIES

INK, MUTED, ACCENT, ALERT, GRID = "#1f2937", "#6b7280", "#2563eb", "#dc2626", "#e5e7eb"
plt.rcParams.update({"font.size": 11})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


def build_dashboard(p, path: str) -> str:
    cat = p.catalog
    s = p.summary()
    # ventas por categoría
    rev = np.zeros(len(CATEGORIES))
    for pid, cnt in p.s9.sales.items():
        rev[cat.category[pid]] += cnt * cat.price[pid]
    top = np.argsort(-rev)[:10][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.barh([CATEGORIES[i][1] for i in top], rev[top] / 1000, color=ACCENT)
    ax.set_xlabel("Ventas aprobadas (miles de S/)", color=MUTED, fontsize=10); _style(ax)
    img_sales = _fig_to_b64(fig)
    # latencia por etapa
    rows = [r for r in p.stage_table() if r.get("entradas")]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    names = [r["etapa"].split(" ")[0] for r in rows][::-1]
    ax.barh(names, [r.get("us_por_item", 0) for r in rows][::-1], color=ACCENT)
    ax.set_xscale("log"); ax.set_xlabel("µs por elemento (escala log)", color=MUTED, fontsize=10); _style(ax)
    img_lat = _fig_to_b64(fig)
    # espera por prioridad
    w = s["espera_promedio_ticks"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.barh(list(w.keys())[::-1], list(w.values())[::-1], color=[ALERT if k in ("alerta", "pago") else ACCENT for k in w][::-1])
    ax.set_xlabel("Espera promedio en el heap (ticks)", color=MUTED, fontsize=10); _style(ax)
    img_wait = _fig_to_b64(fig)
    # tamaños de comunidades
    comm_html = ""
    if p.s8.g is not None:
        g = p.s8.g
        lab = p.s8.comm[g.degree > 0]
        sizes = np.sort(np.bincount(lab)[np.bincount(lab) > 0])[::-1]
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        ax.bar(range(1, len(sizes) + 1), sizes, color=ACCENT, width=1.0)
        ax.set_yscale("log"); ax.set_xlabel("Comunidad (ordenada por tamaño)", color=MUTED, fontsize=10)
        ax.set_ylabel("Productos", color=MUTED, fontsize=10); _style(ax); ax.grid(axis="y", color=GRID, lw=0.6)
        img_comm = _fig_to_b64(fig)
        # categoría dominante por comunidad grande
        items = []
        for c in np.argsort(-np.bincount(p.s8.comm, weights=(g.degree > 0)))[:6]:
            members = np.where((p.s8.comm == c) & (g.degree > 0))[0]
            if len(members) == 0:
                continue
            cats = np.bincount(cat.category[members], minlength=len(CATEGORIES))
            topc = [CATEGORIES[i][1] for i in np.argsort(-cats)[:3] if cats[i] > 0]
            items.append(f"<li><b>C{c}</b> · {len(members)} productos · {html.escape(', '.join(topc))}</li>")
        comm_html = f"<img src='data:image/png;base64,{img_comm}'><ul class='small'>{''.join(items)}</ul>"
    pr_top = np.argsort(-p.s8.pr)[:8]
    pr_rows = "".join(f"<tr><td>{i + 1}</td><td>{html.escape(cat.names[q])}</td><td>{p.s8.pr[q]:.5f}</td>"
                      f"<td>{p.s9.sales.get(int(q), 0)}</td></tr>" for i, q in enumerate(pr_top))
    trend = p.s9.trending(8)
    tr_rows = "".join(f"<tr><td>{html.escape(cat.names[q])}</td><td>{e}</td></tr>" for e, q in trend)
    al_rows = "".join(
        f"<tr><td>{datetime.fromtimestamp(a['event_time'], timezone.utc):%d/%m %H:%M}</td>"
        f"<td>{a['device_id']}</td><td>{a['estimate']}</td><td>{html.escape(a['reason'])}</td></tr>"
        for a in p.s6.alerts[-10:])
    st_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in
                      ("etapa", "entradas", "salidas", "segundos", "us_por_item")) + "</tr>" for r in p.stage_table())
    ac = p.s9.autocomplete("lap", 5)
    ac_html = "".join(f"<li>{html.escape(n)} <span class='muted'>(peso {w})</span></li>" for w, n, _ in ac)
    o = s["ordenes"]
    kpis = [("Eventos procesados", f"{s['eventos_procesados']:,}"), ("Throughput", f"{s['throughput_ev_s']:,} ev/s"),
            ("Ventas aprobadas", f"S/ {s['ventas_soles']:,.0f}"), ("Órdenes aprobadas", f"{o.get('aprobado', 0):,}"),
            ("Órdenes bloqueadas", f"{o.get('bloqueado', 0):,}"), ("Alertas de velocidad", f"{s['alertas_fraude']}")]
    kpi_html = "".join(f"<div class='kpi'><div class='muted'>{k}</div><div class='v'>{v}</div></div>" for k, v in kpis)
    doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Panel EcoStream</title>
<style>
:root{{--bg:#f8fafc;--card:#fff;--ink:{INK};--muted:{MUTED};--line:{GRID};--accent:{ACCENT};--alert:{ALERT}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:20px 16px}} h1{{font-size:20px;margin:0 0 4px}}
.muted{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-top:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}} .card h2{{font-size:14px;margin:0 0 8px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:12px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}} .kpi .v{{font-size:20px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:12px}} td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}}
img{{max-width:100%}} .small{{font-size:12px;padding-left:18px}} .scroll{{overflow-x:auto}}
</style></head><body><div class="wrap">
<h1>Panel EcoStream · ventas, actividad y alertas</h1>
<div class="muted">Grupo 3 · Pipeline de 10 etapas · {s['eventos_generados']:,} eventos generados · semilla {p.seed}</div>
<div class="kpis">{kpi_html}</div>
<div class="grid">
<div class="card"><h2>Ventas por categoría (E3 + E6)</h2><img src="data:image/png;base64,{img_sales}"></div>
<div class="card"><h2>Alertas de fraude (E6 · Count-Min Sketch, umbral 15 intentos / 5 min)</h2>
<div class="scroll"><table><tr><th>Hora UTC</th><th>Dispositivo</th><th>Estimación</th><th>Motivo</th></tr>{al_rows or '<tr><td colspan=4>Sin alertas</td></tr>'}</table></div></div>
<div class="card"><h2>Productos más influyentes (E8 · PageRank)</h2><div class="scroll"><table><tr><th>#</th><th>Producto</th><th>PR</th><th>Ventas</th></tr>{pr_rows}</table></div></div>
<div class="card"><h2>Tendencias última hora (E6 · CMS heavy hitters)</h2><div class="scroll"><table><tr><th>Producto</th><th>Compras estimadas</th></tr>{tr_rows}</table></div></div>
<div class="card"><h2>Comunidades de co-compra (E8 · Louvain, Q = {p.s8.q:.3f})</h2>{comm_html}</div>
<div class="card"><h2>Espera por clase de prioridad (E2 · heap)</h2><img src="data:image/png;base64,{img_wait}"></div>
<div class="card"><h2>Costo por etapa</h2><img src="data:image/png;base64,{img_lat}"></div>
<div class="card"><h2>Autocompletado "lap" (E9 · Trie)</h2><ol class="small">{ac_html}</ol>
<div class="muted">LRU productos: {s['lru_hit_rate_productos']:.1%} aciertos · LRU KD-Tree: {s['lru_hit_rate_kdtree']:.1%} ·
B+Tree órdenes: altura {s['bptree_io']['ordenes']['height']}, {s['bptree_io']['ordenes']['pages']} páginas</div></div>
</div>
<div class="card" style="margin-top:14px"><h2>Métricas del pipeline</h2><div class="scroll"><table>
<tr><th>Etapa</th><th>Entradas</th><th>Salidas</th><th>Segundos</th><th>µs/elemento</th></tr>{st_rows}</table></div></div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
