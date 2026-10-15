"""Etapa 7 [FIJA] (S7) - MinHash (Broder, 1997) + LSH por bandas
(Indyk & Motwani, 1998; Leskovec et al., cap. 3).

MinHash: para k permutaciones aleatorias π_i, sig_i(A) = min_{x∈A} π_i(x).
Propiedad clave: P[sig_i(A) = sig_i(B)] = J(A, B) (similitud de Jaccard), por
lo que la fracción de coincidencias de las firmas estima J con error
estándar sqrt(J(1-J)/k).  Las permutaciones se aproximan con hashes
universales h_i(x) = (a_i·x + b_i) mod p, p = 2^31 - 1.

LSH por bandas: la firma (k = b·r) se divide en b bandas de r filas; dos
conjuntos son candidatos si coinciden en al menos una banda completa:
    P[candidato] = 1 - (1 - J^r)^b      umbral ≈ (1/b)^{1/r}
Con b=16, r=4 (k=64) el umbral es ≈ 0.50: pares con J>=0.7 se detectan con
P>0.99 y pares con J<=0.2 casi nunca se comparan.

Complejidad: firmas O(k·|A|) por conjunto; índice O(n·b) memoria;
consulta O(b log n + |candidatos|·k) frente a O(n·|A|) de la fuerza bruta.
Implementación vectorizada: índice por banda como arreglo ordenado de
claves de 64 bits + ``searchsorted`` (más compacto que un dict de Python).
"""
from __future__ import annotations

import numpy as np

from .hashing import splitmix64

P31 = np.uint64((1 << 31) - 1)


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.k = num_perm
        self.a = rng.integers(1, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 31) - 1, size=num_perm, dtype=np.uint64)

    def signature(self, elements) -> np.ndarray:
        x = np.asarray(elements, dtype=np.uint64) % P31
        if x.size == 0:
            return np.full(self.k, (1 << 31) - 1, dtype=np.uint32)
        h = (x[:, None] * self.a[None, :] + self.b[None, :]) % P31
        return h.min(axis=0).astype(np.uint32)

    def signatures_csr(self, indptr: np.ndarray, indices: np.ndarray,
                       max_nnz_chunk: int = 2_000_000) -> np.ndarray:
        """Firmas de n conjuntos en formato CSR, por bloques para acotar memoria."""
        n = len(indptr) - 1
        out = np.full((n, self.k), (1 << 31) - 1, dtype=np.uint32)
        s = 0
        while s < n:
            # avanzar e hasta acumular ~max_nnz_chunk elementos
            e = int(np.searchsorted(indptr, indptr[s] + max_nnz_chunk, side="right")) - 1
            e = max(e, s + 1)
            e = min(e, n)
            lo, hi = indptr[s], indptr[e]
            x = indices[lo:hi].astype(np.uint64) % P31
            h = ((x[:, None] * self.a[None, :] + self.b[None, :]) % P31).astype(np.uint32)
            starts = (indptr[s:e] - lo).astype(np.int64)
            nonempty = indptr[s + 1:e + 1] > indptr[s:e]
            if nonempty.any():
                red = np.minimum.reduceat(h, starts[nonempty], axis=0)
                out[s:e][nonempty] = red
            s = e
        return out


def estimate_jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def jaccard(a, b) -> float:
    a = np.unique(np.asarray(a)); b = np.unique(np.asarray(b))
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


class LSHIndex:
    def __init__(self, bands: int = 16, rows: int = 4):
        self.b, self.r = bands, rows
        self.keys: list[np.ndarray] = []      # por banda: claves ordenadas
        self.ids: list[np.ndarray] = []       # ids alineados con keys
        self.n = 0

    @property
    def threshold(self) -> float:
        return (1 / self.b) ** (1 / self.r)

    def _band_keys(self, sigs: np.ndarray) -> np.ndarray:
        """(n, k) -> (b, n) claves uint64 por banda (r pasadas vectorizadas)."""
        n = sigs.shape[0]
        S = sigs[:, : self.b * self.r].reshape(n, self.b, self.r).astype(np.uint64)
        h = np.broadcast_to(np.arange(self.b, dtype=np.uint64), (n, self.b)).copy()
        for j in range(self.r):
            h = splitmix64(h ^ S[:, :, j])
        return np.ascontiguousarray(h.T)

    def build(self, sigs: np.ndarray, ids: np.ndarray | None = None) -> None:
        assert sigs.shape[1] >= self.b * self.r
        n = sigs.shape[0]
        ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids)
        bk = self._band_keys(sigs)
        self.keys, self.ids = [], []
        for bi in range(self.b):
            order = np.argsort(bk[bi], kind="stable")
            self.keys.append(bk[bi][order])
            self.ids.append(ids[order].astype(np.int32 if ids.max(initial=0) < 2**31 else np.int64))
        self.n = n

    def build_from_csr(self, hasher: "MinHasher", indptr: np.ndarray, indices: np.ndarray,
                       chunk_sets: int = 250_000) -> None:
        """Construcción en streaming: calcula firmas por bloques y guarda solo las
        claves de banda -> no materializa la matriz n x k de firmas.  Para
        n >= 5M las claves se truncan a 32 bits (las colisiones extra solo
        añaden candidatos, que luego se verifican con Jaccard exacto), lo que
        permite indexar 10M conjuntos en ~2.6 GB."""
        n = len(indptr) - 1
        kdt = np.uint32 if n >= 5_000_000 else np.uint64
        self.key_dtype = kdt
        bk = [np.empty(n, dtype=kdt) for _ in range(self.b)]
        for s in range(0, n, chunk_sets):
            e = min(n, s + chunk_sets)
            ip = indptr[s:e + 1] - indptr[s]
            sig = hasher.signatures_csr(ip, indices[indptr[s]:indptr[e]], max_nnz_chunk=500_000)
            keys = self._band_keys(sig)
            for bi in range(self.b):
                bk[bi][s:e] = keys[bi].astype(kdt, copy=False)
            del sig, keys
        self.keys, self.ids = [], []
        for bi in range(self.b):
            order = np.argsort(bk[bi]).astype(np.int32)
            self.keys.append(bk[bi][order])
            self.ids.append(order)
            bk[bi] = None                      # libera la banda ya indexada
        self.n = n

    def candidates(self, sig: np.ndarray) -> np.ndarray:
        bk = self._band_keys(sig[None, :])[:, 0].astype(getattr(self, "key_dtype", np.uint64))
        found = []
        for bi in range(self.b):
            ks = self.keys[bi]
            lo = np.searchsorted(ks, bk[bi], side="left")
            hi = np.searchsorted(ks, bk[bi], side="right")
            if hi > lo:
                found.append(self.ids[bi][lo:hi])
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(found))

    def memory_bytes(self) -> int:
        return sum(k.nbytes for k in self.keys) + sum(i.nbytes for i in self.ids)


class CSRSets:
    """Colección de conjuntos en CSR con Jaccard exacto (fuerza bruta vectorizada)."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, universe: int | None = None):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        idx = np.asarray(indices)
        self.indices = idx.astype(np.int32) if (idx.size == 0 or idx.max() < 2**31) else idx.astype(np.int64)
        self.sizes = np.diff(self.indptr)
        self.universe = int(universe if universe is not None else (self.indices.max() + 1 if self.indices.size else 1))

    def __len__(self):
        return len(self.sizes)

    def get(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def jaccard_all(self, q: np.ndarray) -> np.ndarray:
        """J(q, S_i) para todos los i: O(nnz) - línea base de fuerza bruta."""
        mask = np.zeros(self.universe, dtype=np.int32)
        q = np.unique(q)
        mask[q[q < self.universe]] = 1
        hits = mask[self.indices]
        inter = np.add.reduceat(hits, self.indptr[:-1]) if hits.size else np.zeros(len(self))
        inter = np.where(self.sizes > 0, inter, 0)
        union = self.sizes + q.size - inter
        return np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    def jaccard_subset(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        q = np.unique(q)
        out = np.empty(len(ids))
        for t, i in enumerate(ids):
            s = self.get(int(i))
            inter = np.intersect1d(q, s, assume_unique=True).size
            u = q.size + s.size - inter
            out[t] = inter / u if u else 0.0
        return out

    def memory_bytes(self) -> int:
        return self.indptr.nbytes + self.indices.nbytes


def lsh_topk(lsh: LSHIndex, sets: CSRSets, sig: np.ndarray, q: np.ndarray, k: int = 10,
             exclude: int | None = None):
    cand = lsh.candidates(sig)
    if exclude is not None:
        cand = cand[cand != exclude]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0), 0
    sims = sets.jaccard_subset(q, cand)
    order = np.argsort(-sims, kind="stable")[:k]
    return cand[order], sims[order], int(cand.size)


def brute_topk(sets: CSRSets, q: np.ndarray, k: int = 10, exclude: int | None = None):
    sims = sets.jaccard_all(q)
    if exclude is not None:
        sims[exclude] = -1
    k2 = min(k, len(sims))
    idx = np.argpartition(-sims, k2 - 1)[:k2]
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return idx, sims[idx]
