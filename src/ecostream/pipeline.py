"""Orquestador del pipeline de 10 etapas (escenario B: comercio electrónico).

Flujo de un evento (cada salida alimenta a una etapa posterior):

  fuente -> E0 FIFO -> E1 limpieza/dedup -> E2 heap de prioridades
        -> E3 hash/LRU/matriz dispersa (enriquece) -> E4 B+Tree (indexa orden)
        -> E5 KD-Tree (candidatos por atributos) -> E6 Bloom+CMS (aprueba/bloquea,
           re-inyecta alertas en E2) -> E7 MinHash+LSH / HNSW (similares)
        -> E8 grafo co-compra (PageRank, Louvain) -> E9 servicio (Trie, Top-K, panel)

E7 y E8 se recalculan cada ``refresh_every`` micro-lotes con los datos que
ya pasaron el filtro (micro-batch, como en Spark Structured Streaming).
"""
from __future__ import annotations

import time

import numpy as np

from .data.generator import StreamConfig, generate_stream, iter_events, make_catalog
from .stages.base import Tracer
from .stages.s0_ingesta import Ingesta
from .stages.s1_preprocesamiento import Preprocesamiento
from .stages.s2_prioridades import ColaPrioridades
from .stages.s3_almacenamiento import Almacenamiento
from .stages.s4_indexacion import Indexacion
from .stages.s5_multidimensional import IndiceMultidimensional
from .stages.s6_filtrado import FiltradoProbabilistico
from .stages.s7_similitud import SimilitudAproximada
from .stages.s8_grafo import GrafoDominio
from .stages.s9_servicio import Servicio


class EcoStreamPipeline:
    def __init__(self, n_events: int = 200_000, n_products: int = 20_000, batch_size: int = 5000,
                 refresh_every: int = 20, seed: int = 42, watch: set | None = None,
                 source=None, verbose: bool = True):
        t0 = time.perf_counter()
        self.seed = seed
        self.batch_size = batch_size
        self.refresh_every = refresh_every
        self.verbose = verbose
        self.catalog = make_catalog(n_products, seed=seed)
        self.stream = generate_stream(self.catalog, StreamConfig(n_events=n_events, seed=seed))
        self.tracer = Tracer(watch)
        src = source if source is not None else iter_events(self.stream, self.catalog)
        n_users = int(self.stream.meta["n_users"]) + 1
        tr = self.tracer
        self.s0 = Ingesta(src, capacity=10 * batch_size, tracer=tr)
        self.s1 = Preprocesamiento(tracer=tr)
        self.s2 = ColaPrioridades(capacity_per_tick=batch_size, tracer=tr)
        self.s3 = Almacenamiento(self.catalog, n_users, tracer=tr)
        self.s4 = Indexacion(self.catalog, tracer=tr)
        self.s5 = IndiceMultidimensional(self.catalog, tracer=tr)
        self.s6 = FiltradoProbabilistico(self.stream.blacklist_tokens, self.s3, tracer=tr)
        self.s7 = SimilitudAproximada(self.catalog, tracer=tr)
        self.s8 = GrafoDominio(self.catalog.n, tracer=tr)
        self.s9 = Servicio(self.catalog, self.s3, self.s4, self.s5, self.s7, self.s8, self.s6, tracer=tr)
        self.setup_seconds = time.perf_counter() - t0
        self.processed = 0
        self.now = 0.0
        self.run_seconds = 0.0
        self.batches = 0

    @property
    def stages(self):
        return [self.s0, self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7, self.s8, self.s9]

    def _handle(self, ev: dict) -> None:
        if ev["event_type"] == "fraud_alert":
            self.s6.handle_alert(ev)
            if self.tracer.watching({"event_id": ev.get("source_event")}):
                self.tracer({"event_id": ev["source_event"]}, "E2",
                            "alerta re-inyectada con prioridad 0 y atendida antes que el resto")
            return
        ev = self.s3.process(ev)
        ev = self.s4.process(ev)
        ev = self.s5.process(ev, self.s3)
        ev = self.s6.process(ev, self.s2)
        ev = self.s7.process(ev)
        ev = self.s8.process(ev)
        self.s9.process(ev)
        self.processed += 1
        self.now = max(self.now, ev["event_time"])

    def refresh(self) -> None:
        self.s7.rebuild_behavior_index(self.s3.interactions)
        self.s8.rebuild(self.s4, self.now)
        self.s9.refresh()

    def run(self, max_batches: int | None = None) -> "EcoStreamPipeline":
        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()
        B = self.batch_size
        while True:
            # llegada variable (picos de hasta 1.4B por tick) con capacidad de servicio B
            arrive = int(B * rng.uniform(0.55, 1.4))
            if not self.s0.exhausted:
                self.s0.pump(arrive)
            batch = self.s0.next_batch(arrive)
            if batch:
                self.s2.push_batch(self.s1.process(batch))
            for _ in range(self.s2.capacity):
                ev = self.s2.pop()
                if ev is None:
                    break
                self._handle(ev)
            self.s2.end_tick()
            self.batches += 1
            if self.batches % self.refresh_every == 0:
                self.refresh()
                if self.verbose:
                    print(f"  lote {self.batches:4d}: {self.processed:,} eventos procesados, "
                          f"heap={len(self.s2.heap):,}, alertas={len(self.s6.alerts)}, "
                          f"{time.perf_counter() - t0:.1f}s", flush=True)
            if self.s0.exhausted and not len(self.s0.queue) and not len(self.s2.heap):
                break
            if max_batches is not None and self.batches >= max_batches:
                break
        self.refresh()
        self.run_seconds = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        orders = list(self.s3.orders.items())
        st = {"aprobado": 0, "bloqueado": 0, "pendiente": 0}
        for _, o in orders:
            st[o["status"]] = st.get(o["status"], 0) + 1
        return {
            "eventos_generados": self.stream.n,
            "eventos_procesados": self.processed,
            "segundos": round(self.run_seconds, 2),
            "throughput_ev_s": round(self.processed / max(self.run_seconds, 1e-9)),
            "ordenes": st,
            "alertas_fraude": len(self.s6.alerts),
            "ventas_soles": round(self.s9.revenue, 2),
            "espera_promedio_ticks": self.s2.avg_wait(),
            "max_backlog_heap": self.s2.max_backlog,
            "lru_hit_rate_productos": round(self.s3.cache.hit_rate, 4),
            "lru_hit_rate_kdtree": round(self.s5.cache.hit_rate, 4),
            "bptree_io": self.s4.io(),
            "grafo": dict(self.s8.metrics.counters),
            "generador": self.stream.meta,
        }

    def stage_table(self) -> list[dict]:
        return [s.metrics.as_dict() for s in self.stages]
