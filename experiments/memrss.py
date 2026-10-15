"""Medición de memoria para n = 10M (y estructuras muy grandes) con RSS del proceso.

``tracemalloc`` multiplica 3-5x el tiempo de las estructuras puras en Python
(registra cada asignación), lo que a 10M elementos supera el presupuesto de
cómputo.  Para esos casos se usa la herramienta equivalente del sistema
operativo, en el mismo proceso:
* se reinicia el pico de RSS (``echo 5 > /proc/self/clear_refs`` reinicia VmHWM),
* se construye la estructura,
* actual = VmRSS - RSS inicial; pico = VmHWM - RSS inicial.
(Una versión previa usaba un proceso hijo con fork; con el índice LSH de 10M
duplicaba la memoria y el hijo era terminado por el OOM killer.)
"""
from __future__ import annotations

import gc


def _status(key: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    return 0


def memory_rss(setup, run):
    gc.collect()
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass
    base = _status("VmRSS:")
    st = setup()
    res = run(st)
    gc.collect()
    cur = _status("VmRSS:") - base
    peak = _status("VmHWM:") - base
    del st, res
    gc.collect()
    return max(cur, 0) / 2**20, max(peak, cur, 0) / 2**20
