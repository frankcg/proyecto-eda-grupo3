"""Etapa 7 [FIJA] - Similitud aproximada (S7).

* **Conjuntos -> MinHash + LSH (Jaccard)**: cada producto se representa por
  el conjunto de usuarios que interactuaron con él (columnas de la matriz
  dispersa de la etapa 3; las compras solo cuentan si la etapa 6 aprobó el
  pago).  La similitud de Jaccard producto-producto es baja (máx. ~0.3), por
  lo que se usan b = 32 bandas de r = 2 filas (umbral ≈ 0.18) en lugar de
  16x4 (umbral 0.5, recall < 10 %).  "Productos comprados por las mismas personas" = productos
  relacionados por comportamiento.  Se eligió MinHash+LSH frente a SimHash
  porque la señal es un conjunto binario (Jaccard) y no un vector ponderado
  de características (coseno).
* **Vectores -> HNSW**: embeddings de contenido de 32 dimensiones
  (categoría, marca, precio).  Se eligió HNSW frente a IVF porque el catálogo
  crece por inserciones incrementales (HNSW no requiere re-entrenar
  centroides) y ofrece mejor recall a igual latencia; el espacio es
  euclidiano, por lo que no se requiere M-Tree/VP-Tree.
Ambos índices se reconstruyen/actualizan cada ``refresh`` lotes.
Salida -> listas de productos similares para la etapa 9.
"""
from __future__ import annotations

import numpy as np

from ..structures.hnsw import HNSW
from ..structures.kd_tree import knn_brute_force
from ..structures.minhash_lsh import CSRSets, LSHIndex, MinHasher, brute_topk, lsh_topk
from .base import StageMetrics, timed


class SimilitudAproximada:
    def __init__(self, catalog, tracer=None, hnsw_M: int = 12, hnsw_efc: int = 80):
        self.catalog = catalog
        self.metrics = StageMetrics("E7 MinHash+LSH y HNSW")
        self.tracer = tracer
        with timed(self.metrics):
            self.hnsw = HNSW(catalog.embedding.shape[1], M=hnsw_M, ef_construction=hnsw_efc,
                             ef_search=40, max_elements=catalog.n)
            self.hnsw.add_items(catalog.embedding)
        self.hnsw_build_s = self.metrics.seconds
        self.minhash = MinHasher(64, seed=11)
        self.bands, self.rows = 32, 2        # umbral ≈ (1/32)^(1/2) ≈ 0.18 (ver experimento de ajuste)
        self.lsh = None
        self.sets = None
        self.product_ids = None
        self.sigs = None
        self.pid_to_row = {}

    def rebuild_behavior_index(self, interactions, min_weight: float = 1.0) -> None:
        """Re-construye MinHash+LSH con los usuarios que interactuaron con cada producto."""
        with timed(self.metrics):
            cols: dict[int, list[int]] = {}
            for uid, row in interactions._rows.items():
                for pid, w in row.items():
                    if w >= min_weight:
                        cols.setdefault(pid, []).append(uid)
            pids = sorted(p for p, us in cols.items() if len(us) >= 2)
            if not pids:
                return
            sizes = [len(cols[p]) for p in pids]
            indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            indices = np.concatenate([np.sort(np.asarray(cols[p], dtype=np.int64)) for p in pids])
            self.sets = CSRSets(indptr, indices, universe=interactions.shape[0])
            self.sigs = self.minhash.signatures_csr(indptr, indices)
            self.lsh = LSHIndex(bands=self.bands, rows=self.rows)
            self.lsh.build(self.sigs)
            self.product_ids = np.asarray(pids)
            self.pid_to_row = {p: i for i, p in enumerate(pids)}
        self.metrics.counters["lsh_reconstrucciones"] += 1
        self.metrics.counters["productos_en_lsh"] = len(pids)

    def similar_behavior(self, pid: int, k: int = 10, exact: bool = False):
        row = self.pid_to_row.get(pid)
        if row is None or self.lsh is None:
            return []
        q = self.sets.get(row)
        if exact:
            ids, sims = brute_topk(self.sets, q, k, exclude=row)
        else:
            ids, sims, _ = lsh_topk(self.lsh, self.sets, self.sigs[row], q, k, exclude=row)
        return [(int(self.product_ids[i]), float(s)) for i, s in zip(ids, sims) if s > 0]

    def similar_content(self, pid: int, k: int = 10, exact: bool = False):
        q = self.catalog.embedding[pid]
        if exact:
            d, idx = knn_brute_force(self.catalog.embedding, q, k + 1)
        else:
            d, idx = self.hnsw.knn_query(q, k + 1)
        return [(int(i), float(1 - dd * dd / 2)) for dd, i in zip(d, idx) if i != pid][:k]   # coseno

    def process(self, ev: dict) -> dict:
        """Para compras aprobadas adjunta los similares (observable en la traza)."""
        if ev["event_type"] != "purchase" or ev.get("_order", {}).get("status") != "aprobado":
            return ev
        if self.tracer and self.tracer.watching(ev):
            pid = ev["product_id"]
            b = self.similar_behavior(pid, 3)
            c = self.similar_content(pid, 3)
            self.tracer(ev, "E7", f"compra aprobada alimenta los conjuntos de compradores; similares por "
                        f"comportamiento (LSH) {[p for p, _ in b]} y por contenido (HNSW) {[p for p, _ in c]}")
        self.metrics.items_in += 1
        self.metrics.items_out += 1
        return ev
