"""Imprime las cifras clave usadas en el informe y la presentación."""
import json
import pandas as pd
R = 'results/'
b = pd.read_csv(R + 'base.csv'); p = pd.read_csv(R + 'probabilisticas.csv'); s = pd.read_csv(R + 'similitud.csv')
g = pd.read_csv(R + 'grafo.csv'); pl = pd.read_csv(R + 'pipeline.csv')
def show(df, cols, q=None):
    d = df if q is None else df.query(q)
    print(d[[c for c in cols if c in d]].to_string(index=False)); print()
show(b, ['estructura','operacion','n','reps','media_s','std_s','us_por_op','mem_actual_mb','mem_pico_mb'], "n in (1000000, 10000000)")
show(b, ['estructura','operacion','n','altura','paginas','lecturas_disco_por_consulta','pool_hit_rate','escrituras_disco','nodos_visitados','hit_rate','factor_carga','sondeos_por_op','nodos'], "estructura.str.contains('B\\\\+Tree|KD|LRU|HashTable|Trie')", )
show(p, ['estructura','operacion','n','p_objetivo','reps','media_s','us_por_op','mem_actual_mb','mem_pico_mb','bits_por_elemento','k','fpr_empirica','fpr_teorica','falsos_negativos'], "estructura.str.contains('Bloom|set')")
show(p, ['estructura','n','reps','media_s','us_por_op','mem_actual_mb','mem_pico_mb','claves_distintas','subestimaciones','error_abs_medio','error_max','cota_eps_N','frac_dentro_cota','error_rel_top100','error_rel_medio'], "estructura.str.contains('Count')")
show(s, ['estructura','operacion','n','reps','media_s','us_por_op','mem_actual_mb','mem_pico_mb','recall_J05','recall_top10','candidatos_prom','recall_10','nodos_visitados','nodos_totales','construir_s','umbral'])
show(g, ['algoritmo','n','nodos','aristas','reps','media_s','std_s','iteraciones','modularidad','modularidad_std','comunidades','comunidades_std','mayor','mediana','estabilidad_nmi','nmi_vs_plantadas','spearman_vs_grado','solapamiento_topK','mem_pico_mb'])
print(pl.drop(columns=['espera_ticks']).to_string(index=False)); print(pl[['n','espera_ticks']].to_string(index=False))
print(json.dumps(json.load(open(R + 'comunidades_grafo_pipeline.json')), indent=0)[:1500])
print(pd.read_csv(R + 'pendientes.csv').to_string(index=False))
