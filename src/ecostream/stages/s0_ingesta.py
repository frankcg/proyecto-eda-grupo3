"""Etapa 0 - Ingesta y buffer (S1): cola FIFO con lista enlazada.

Recibe eventos de la fuente (generador sintético o REES46) y los almacena en
una ``LinkedQueue`` con capacidad máxima (backpressure).  El consumidor extrae
micro-lotes de tamaño B para la etapa 1.
Salida -> micro-lote FIFO para la etapa 1.
"""
from __future__ import annotations

from ..structures.linked_queue import LinkedQueue
from .base import StageMetrics, timed


class Ingesta:
    def __init__(self, source, capacity: int = 200_000, tracer=None):
        self.source = iter(source)
        self.queue = LinkedQueue(capacity)
        self.metrics = StageMetrics("E0 Ingesta (FIFO lista enlazada)")
        self.exhausted = False
        self.tracer = tracer

    def pump(self, n: int) -> int:
        """Lee hasta n eventos de la fuente y los encola."""
        got = 0
        with timed(self.metrics):
            for _ in range(n):
                try:
                    ev = next(self.source)
                except StopIteration:
                    self.exhausted = True
                    break
                if not self.queue.enqueue(ev):
                    break
                got += 1
                if self.tracer and self.tracer.watching(ev):
                    self.tracer(ev, "E0", f"encolado en FIFO (posición {len(self.queue)})")
        self.metrics.items_in += got
        return got

    def next_batch(self, b: int) -> list:
        with timed(self.metrics):
            batch = self.queue.dequeue_batch(b)
        self.metrics.items_out += len(batch)
        if self.tracer:
            for ev in batch:
                if self.tracer.watching(ev):
                    self.tracer(ev, "E0", f"desencolado en micro-lote de {len(batch)}")
        return batch
