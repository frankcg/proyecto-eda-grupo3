"""Genera y ejecuta los notebooks del repositorio.

    python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


SETUP = """
import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd
pd.set_option('display.max_colwidth', 80); pd.set_option('display.width', 160)
"""

demo = [
    md("""# 01 · Demo del pipeline EcoStream (escenario B: comercio electrónico)
**Grupo 3** — Estructuras de Datos y Algoritmos 2026-2.

Este notebook ejecuta el pipeline de **10 etapas** sobre un stream de 200 000 eventos y muestra, para eventos
concretos, qué hace cada etapa y cómo su salida alimenta a la siguiente:
prioridades (E2), filtrado probabilístico (E6), similitud frente a métodos exactos (E7), análisis del grafo (E8)
y servicio al usuario final (E9)."""),
    code(SETUP),
    md("## 1. Generación del stream (esquema REES46 + componente sintético)"),
    code("""
from ecostream.data.generator import make_catalog, generate_stream, StreamConfig, iter_events, EVENT_TYPES
cat = make_catalog(20_000, seed=42)
st = generate_stream(cat, StreamConfig(n_events=200_000, seed=42))
print(st.meta)
pd.DataFrame(list(__import__('itertools').islice(iter_events(st, cat), 0, 8)))
"""),
    code("""
pd.Series(np.bincount(st.cols['event_type']), index=EVENT_TYPES, name='eventos').to_frame().T
"""),
    md("## 2. Ejecución de punta a punta con trazas"),
    code("""
import importlib.util
spec = importlib.util.spec_from_file_location('demo', ROOT / 'scripts' / 'demo.py')
demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
watch = demo.pick_watch(200_000, 20_000, 42)
from ecostream.pipeline import EcoStreamPipeline
p = EcoStreamPipeline(200_000, 20_000, 5000, refresh_every=20, seed=42,
                      watch={v for v in watch.values() if v is not None}, verbose=False).run()
watch
"""),
    code("""
for name, eid in watch.items():
    print(f'\\n▶ {name} (event_id={eid})')
    for stage, msg in p.tracer.log.get(eid, []):
        print(f'   [{stage}] {msg}')
"""),
    md("## 3. Métricas por etapa"),
    code("pd.DataFrame(p.stage_table())"),
    code("""
s = p.summary()
pd.Series(s['espera_promedio_ticks'], name='espera promedio (ticks)').to_frame()
"""),
    md("## 4. Filtrado probabilístico: Bloom (con verificación exacta) y Count-Min Sketch"),
    code("""
m = p.s6.metrics.counters
print(f"Bloom: m={p.s6.bloom.m:,} bits, k={p.s6.bloom.k}; positivos={int(m['bloom_positivos'])}, "
      f"verificados={int(m['bloom_verificados'])}, falsos positivos={int(m['bloom_falsos_positivos'])}")
print('Órdenes:', s['ordenes'])
pd.DataFrame(p.s6.alerts)[['event_time', 'device_id', 'estimate', 'reason']]
"""),
    md("## 5. Similitud aproximada vs exacta"),
    code("""
import time
pid = int(max(p.s9.sales, key=p.s9.sales.get))
rows = []
for name, fa, fe in [('MinHash+LSH vs Jaccard exacto', p.s7.similar_behavior, p.s7.similar_behavior),
                     ('HNSW vs KNN exacto', p.s7.similar_content, p.s7.similar_content)]:
    t = time.perf_counter(); a = fa(pid, 10); ta = time.perf_counter() - t
    t = time.perf_counter(); e = fe(pid, 10, exact=True); te = time.perf_counter() - t
    rows.append({'método': name, 'ms aprox': ta * 1e3, 'ms exacto': te * 1e3,
                 'recall@10': len({x for x, _ in a} & {x for x, _ in e}) / max(1, len(e))})
print('Producto:', cat.names[pid]); pd.DataFrame(rows)
"""),
    md("## 6. Grafo de co-compra: PageRank vs grado, Louvain vs Label Propagation (mismo grafo)"),
    code("""
from ecostream.graph.pagerank import degree_ranking
from ecostream.graph.communities import louvain, label_propagation, modularity, community_stats
g = p.s8.g; deg = degree_ranking(g)
top = np.argsort(-p.s8.pr)[:10]
display(pd.DataFrame({'producto': [cat.names[i] for i in top], 'PageRank': p.s8.pr[top], 'grado': deg[top]}))
act = g.degree > 0
res = []
for name, fn in [('Louvain', louvain), ('Label Propagation', label_propagation)]:
    t = time.perf_counter(); lab, _ = fn(g, seed=0); dt = time.perf_counter() - t
    res.append({'algoritmo': name, 'Q': modularity(g, lab), 'comunidades': community_stats(lab[act])['n_communities'], 's': dt})
pd.DataFrame(res)
"""),
    md("## 7. Servicio: autocompletado, recomendaciones Top-K, rango de precios y panel"),
    code("""
for pref in ['smart', 'lap', 'zapa']:
    print(pref, '→', [n for _, n, _ in p.s9.autocomplete(pref, 4)])
uid = next(u for u, pr in p.s3.users.items() if len(pr['purchased']) >= 2)
print('compró:', [cat.names[x] for x in p.s3.user_profile(uid)['purchased'][-3:]])
pd.DataFrame([{'score': sc, 'producto': cat.names[q], 'fuentes': '+'.join(sorted(set(w)))} for sc, q, w in p.s9.recommend(uid, 8, explain=True)])
"""),
    code("""
from ecostream.dashboard import build_dashboard
build_dashboard(p, str(ROOT / 'docs' / 'dashboard_notebook.html'))
"""),
]

exp = [
    md("""# 02 · Resultados experimentales
Lee los CSV de `experiments/results/` producidos por `bench_*.py` y regenera las figuras log-log.
Política: 100 repeticiones cuando es viable (presupuesto de 60 s por estructura y tamaño), menos en 1M/10M;
semilla 42; memoria con `tracemalloc` (≤1M) y RSS con `psutil` (10M)."""),
    code(SETUP),
    code("import json; json.load(open(ROOT / 'experiments/results/plataforma.json'))"),
    code("""
import subprocess
print(subprocess.run([sys.executable, str(ROOT / 'experiments/plots.py')], capture_output=True, text=True).stdout[-4000:])
"""),
    code("""
from IPython.display import Image, display
for f in ['fig_base_loglog', 'fig_base_memoria', 'fig_probabilisticas', 'fig_lsh_curva', 'fig_similitud', 'fig_grafo', 'fig_pipeline']:
    path = ROOT / 'experiments/figures' / f'{f}.png'
    if path.exists():
        print(f); display(Image(filename=str(path), width=900))
"""),
    md("## Tablas"),
    code("""
cols = ['estructura', 'operacion', 'n', 'reps', 'media_s', 'std_s', 'us_por_op', 'mem_actual_mb', 'mem_pico_mb']
pd.read_csv(ROOT / 'experiments/results/base.csv')[cols]
"""),
    code("pd.read_csv(ROOT / 'experiments/results/probabilisticas.csv')"),
    code("pd.read_csv(ROOT / 'experiments/results/similitud.csv')"),
    code("pd.read_csv(ROOT / 'experiments/results/grafo.csv')"),
    code("pd.read_csv(ROOT / 'experiments/results/pipeline.csv')"),
    code("pd.read_csv(ROOT / 'experiments/results/pendientes.csv')"),
]

for name, cells in [("01_demo_pipeline.ipynb", demo), ("02_resultados_experimentales.ipynb", exp)]:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbf.write(nb, HERE / name)
    print("ok", name)
