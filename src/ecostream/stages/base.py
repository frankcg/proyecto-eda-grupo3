"""Infraestructura común: métricas por etapa y trazas de eventos."""
from __future__ import annotations

import time
from collections import defaultdict


class StageMetrics:
    def __init__(self, name: str):
        self.name = name
        self.seconds = 0.0
        self.calls = 0
        self.items_in = 0
        self.items_out = 0
        self.counters: dict[str, float] = defaultdict(float)

    def as_dict(self) -> dict:
        d = {"etapa": self.name, "segundos": round(self.seconds, 4), "entradas": self.items_in,
             "salidas": self.items_out}
        if self.items_in:
            d["us_por_item"] = round(1e6 * self.seconds / self.items_in, 2)
        d.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.counters.items()})
        return d


class Tracer:
    """Registra lo que cada etapa hace con los eventos marcados (demo)."""

    def __init__(self, watch: set | None = None):
        self.watch = set(watch or ())
        self.log: dict[int, list] = defaultdict(list)

    def __call__(self, ev, stage: str, msg: str) -> None:
        if ev is not None and ev.get("event_id") in self.watch:
            self.log[ev["event_id"]].append((stage, msg))

    def watching(self, ev) -> bool:
        return ev is not None and ev.get("event_id") in self.watch


class timed:
    def __init__(self, m: StageMetrics, n_in: int = 0):
        self.m, self.n_in = m, n_in

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.m.seconds += time.perf_counter() - self.t
        self.m.calls += 1
        self.m.items_in += self.n_in
