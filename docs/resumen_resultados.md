# Resumen de resultados

Plataforma: Intel Xeon @ 2.30 GHz, 2 vCPU, 7.8 GB RAM, Python 3.13.16, NumPy 2.5.3, semilla 42. Repeticiones: 100 donde es viable (presupuesto de 60 s por estructura y tamaño), menos en 1M y 10M (cada fila de `experiments/results/*.csv` indica cuántas). Memoria con `tracemalloc` (≤1M) y RSS/VmHWM (10M). Cifras completas en `experiments/results/resumen_cifras.txt`.

## Pipeline de punta a punta

| Eventos | Eventos/s | Tiempo | Órdenes aprobadas / bloqueadas | Alertas | RAM pico |
|---|---:|---:|---:|---:|---:|
| 10 mil | 10 333 | 1 s | 348 / 2 | 3 | 0.1 GB |
| 100 mil | 11 551 | 9 s | 3 480 / 17 | 3 | 0.2 GB |
| 1 millón | 7 597 | 2.2 min | 34 808 / 146 | 20 | 1.1 GB |
| **10 millones** | **7 674** | **21.7 min** | **348 074 / 1 405** | **200** | **4.5 GB** |

Pagos, alertas y compras esperan 0 ticks en el heap en todos los tamaños; solo las vistas/búsquedas esperan en los picos (0.5–0.7 ticks).

## Etapas [FIJA]

| Comparación | Tamaño | Resultado |
|---|---|---|
| Bloom Filter vs `set` | 10M claves | 11 MB vs 594 MB (52× menos); FPR 1.008 % (teórica 1.004 %); 0 falsos negativos |
| Count-Min Sketch vs `Counter` | 10M eventos | 104 KB constantes vs 72 MB; nunca subestima; 100 % de estimaciones dentro de εN |
| MinHash+LSH (32×2) vs fuerza bruta | 10M conjuntos | 0.22 ms vs 2.06 s por consulta (≈9 200×); recall 1.0 para J ≥ 0.5; índice 2.4 GB |
| HNSW vs KNN exacto | 1M vectores 32D | 0.84 ms vs 45 ms (54×); recall@10 = 0.989 |
| PageRank vs grado | 8.6M aristas | 8.6 s, 38 iteraciones; Spearman 0.985 con el grado |
| Louvain vs Label Propagation | 8.6M aristas | Q 0.689 vs 0.688; estabilidad NMI 0.999 vs 0.945; 553 vs 4 722 comunidades; 152 s vs 78 s |
| Louvain vs Label Propagation | grafo real de co-compra (1M eventos) | Q 0.755 vs 0.630; 221 vs 848 comunidades |

## Estructuras base (10M)

| Estructura | Resultado |
|---|---|
| B+Tree (páginas 4 KB) | altura 4, 72 231 páginas; búsqueda puntual 23 µs y 1.7 lecturas de disco |
| KD-Tree 4D | KNN 0.19 ms vs 211 ms de fuerza bruta (1 100×), 94 nodos visitados |
| Tabla hash propia | 1.56 µs/op, 1.4 sondeos por operación, 256 MB (dict: 320 MB) |
| Trie (1M nombres) | autocompletar 1.8 µs vs 963 ms del escaneo lineal; 1.7 GB |

## Límites declarados

- HNSW se midió hasta 1M vectores: la construcción en Python puro tarda 55 min con 1M (≈10 h con 10M).
- El Trie se midió hasta 1M nombres: 1.7 GB con 1M; 10M (≈17 GB) no cabe en 7.8 GB de RAM.
- Las alternativas cuadráticas (`list.pop(0)`, `insort`, escaneo lineal) se midieron hasta 100K–1M.
