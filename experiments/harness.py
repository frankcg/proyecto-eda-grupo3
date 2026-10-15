"""Arnés de medición común a todos los experimentos.

Política de repeticiones (enunciado: 100 cuando sea viable):
* Se ejecuta una corrida de calentamiento y se estima su duración t.
* reps = min(100, max(r_min, floor(PRESUPUESTO / t)))  con
  PRESUPUESTO = 60 s por (estructura, tamaño); r_min = 3 hasta 1M y 1 en 10M.
  => 1K y 10K casi siempre llegan a 100 repeticiones; 1M y 10M usan menos por
  restricción de tiempo de cómputo (2 vCPU), lo que se reporta en cada fila.
* Tiempo: ``time.perf_counter``; se reportan media y desviación estándar.
* Memoria: corrida separada con ``tracemalloc`` (memoria actual retenida por la
  estructura y pico durante la operación); NumPy registra sus buffers en
  tracemalloc, por lo que se contabilizan.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SEED = 42
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
BUDGET = float(os.environ.get("BENCH_BUDGET", 60))


def platform_info() -> dict:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        ram = None
    return {"cpu": cpu, "nucleos": os.cpu_count(), "ram_gb": ram, "so": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "semilla": SEED}


def timeit(setup, run, n: int, r_min: int | None = None, max_reps: int = 100):
    """setup() -> estado; run(estado) -> resultado. Devuelve (media, std, reps, último resultado)."""
    if r_min is None:
        r_min = 1 if n >= 10_000_000 else 3
    st = setup()
    gc.collect()
    t0 = time.perf_counter()
    res = run(st)
    t1 = time.perf_counter() - t0
    reps = int(min(max_reps, max(r_min, math.floor(BUDGET / max(t1, 1e-9)))))
    if n >= 10_000_000:
        reps = min(reps, 3)
    times = [t1] if reps == 1 else []
    for _ in range(reps if reps > 1 else 0):
        st = setup()
        gc.collect()
        t0 = time.perf_counter()
        res = run(st)
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), len(arr), res


def memory(setup, run):
    """Memoria (MB) retenida por el resultado y pico durante setup+run."""
    gc.collect()
    tracemalloc.start()
    st = setup()
    res = run(st)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del st, res
    gc.collect()
    return cur / 2**20, peak / 2**20


class Recorder:
    def __init__(self, name: str):
        self.path = RESULTS / f"{name}.csv"
        self.rows = []
        if self.path.exists():
            with open(self.path) as f:
                self.rows = list(csv.DictReader(f))

    def add(self, **row):
        if self.path.exists():           # re-lee para fusionar con lo escrito por otros procesos
            with open(self.path) as f:
                self.rows = list(csv.DictReader(f))
        row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
        self.rows = [r for r in self.rows if not (r.get("estructura") == str(row.get("estructura"))
                                                  and r.get("n") == str(row.get("n"))
                                                  and r.get("operacion") == str(row.get("operacion")))]
        self.rows.append({k: str(v) for k, v in row.items()})
        self.save()
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    def save(self):
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


def save_platform():
    with open(RESULTS / "plataforma.json", "w") as f:
        json.dump(platform_info(), f, indent=1)


def sizes_from_argv(default_max: int = 10_000_000):
    mx = default_max
    for a in sys.argv[1:]:
        if a.startswith("--max="):
            mx = int(float(a.split("=")[1]))
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = set(a.split("=")[1].split(","))
    mn = 0
    for a in sys.argv[1:]:
        if a.startswith("--min="):
            mn = int(float(a.split("=")[1]))
    return [s for s in SIZES if mn <= s <= mx], only
