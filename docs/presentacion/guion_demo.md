# Guion de la demo (3-4 minutos)

Comando (preparar con antelación; tarda ~3-4 min, así que conviene ejecutarlo antes y mostrar la salida y el panel):

```powershell
python scripts/demo.py | Out-File -Encoding utf8 docs/demo_output.txt   # PowerShell
python scripts/demo.py | tee docs/demo_output.txt                        # bash
```

| Tiempo | Quién | Qué mostrar | Qué decir |
|---|---|---|---|
| 0:00–0:40 | Integrante 1 | Sección 1 de la salida: recorrido de la **compra express**, el **registro sucio** y el **duplicado** | La compra entra por la FIFO, E1 la valida y deduplica por búsqueda binaria, E2 la pone en prioridad 2 y sale sin esperar; el registro con precio negativo se rechaza en E1; el duplicado se elimina por Merge Sort + adyacentes. |
| 0:40–1:10 | Integrante 1 | Sección 2: espera por clase | Durante los picos solo esperan las vistas; pagos y compras se atienden en el mismo tick. |
| 1:10–1:50 | Integrante 3 | Recorrido del **pago fraudulento** y Sección 3 | El CMS estima más de 15 intentos en 5 minutos, genera la alerta, la re-inyecta en el heap con prioridad 0 y añade el dispositivo al Bloom; los positivos del Bloom se verifican en la tabla hash antes de bloquear (0 bloqueos erróneos). |
| 1:50–2:30 | Integrante 3 | Sección 4: LSH vs Jaccard exacto; HNSW vs KNN exacto | Mismo resultado (recall@10) con una fracción del trabajo; el KD-Tree de E5 se verifica contra fuerza bruta. |
| 2:30–3:10 | Integrante 4 | Sección 5: PageRank vs grado; Louvain vs Label Propagation en el mismo grafo | Grafo construido con un range scan del B+Tree (ventana de 7 días, solo órdenes aprobadas). Louvain obtiene mayor modularidad. |
| 3:10–3:50 | Integrante 4 (con Integrante 2) | Sección 6 y `docs/dashboard.html` | Autocompletado con Trie, recomendaciones Top-K explicadas (PPR+LSH+HNSW+KD), consulta por rango de precios en el B+Tree, panel de ventas, alertas y métricas por etapa. |

Plan B sin conexión o si falla la ejecución: abrir `docs/demo_output.txt` y `docs/dashboard.html` (generados por la misma ejecución).
