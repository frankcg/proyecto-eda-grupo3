"""Etapa 7 [FIJA]: MinHash+LSH vs fuerza bruta (Jaccard) y HNSW vs KNN exacto.

    python experiments/bench_similarity.py [--max=1e7] [--only=lsh,hnsw]
"""
from __future__ import annotations

import time

import numpy as np

from harness import Recorder, memory, save_platform, sizes_from_argv, timeit, SEED
from memrss import memory_rss

from ecostream.structures.hnsw import HNSW
from ecostream.structures.kd_tree import KDTree, knn_brute_force
from ecostream.structures.minhash_lsh import CSRSets, LSHIndex, MinHasher, brute_topk

REC = Recorder("similitud")


def make_sets(n, avg=10, seed=SEED):
    """n conjuntos: 40 % en grupos de casi-duplicados (J alto), resto aleatorios."""
    rng = np.random.default_rng(seed + n)
    U = max(10_000, n)
    sizes = np.clip(rng.poisson(avg, n), 3, 40)
    indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    indices = rng.integers(0, U, indptr[-1]).astype(np.int32)
    # grupos de 4: miembros 1..3 copian al miembro 0 con ~10 % de mutación
    n_groups = int(n * 0.4) // 4
    for gi in range(n_groups):
        base = gi * 4
        sb = indices[indptr[base]:indptr[base + 1]]
        for j in range(1, 4):
            i = base + j
            seg = indices[indptr[i]:indptr[i + 1]]
            L = min(len(seg), len(sb))
            seg[:L] = sb[:L]
            mut = rng.random(L) < 0.1
            seg[:L][mut] = rng.integers(0, U, mut.sum())
    # orden y unicidad por conjunto (CSR canónico)
    out_ptr = [0]; out = []
    for i in range(n):
        u = np.unique(indices[indptr[i]:indptr[i + 1]])
        out.append(u); out_ptr.append(out_ptr[-1] + len(u))
    return np.asarray(out_ptr, dtype=np.int64), np.concatenate(out).astype(np.int32), U


def make_sets_fast(n, avg=10, seed=SEED):
    """Versión vectorizada para n grande (mismo modelo generativo)."""
    if n <= 1_000_000:
        return make_sets(n, avg, seed)
    rng = np.random.default_rng(seed + n)
    U = n
    sizes = np.full(n, avg, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    M = rng.integers(0, U, (n, avg)).astype(np.int32)
    g = (np.arange(n) % 4 != 0) & (np.arange(n) < int(n * 0.4) // 4 * 4)
    src = (np.arange(n) // 4) * 4
    rows = np.where(g)[0]
    for s in range(0, len(rows), 1_000_000):
        r = rows[s:s + 1_000_000]
        block = M[src[r]].copy()
        mut = rng.random(block.shape) < 0.1
        block[mut] = rng.integers(0, U, mut.sum()).astype(np.int32)
        M[r] = block
    M.sort(axis=1)
    return indptr, M.ravel(), U


def bench_lsh(n, b=32, r=2, k=64):
    indptr, indices, U = make_sets_fast(n)
    sets = CSRSets(indptr, indices, U)
    mh = MinHasher(k, seed=1)
    def build(_):
        lsh = LSHIndex(b, r)
        lsh.build_from_csr(mh, indptr, indices)
        return lsh
    if n >= 10_000_000:       # memoria primero (sin mantener dos índices vivos a la vez)
        cur, peak = memory_rss(lambda: None, build)
    mean, std, reps, lsh = timeit(lambda: None, build, n, max_reps=20)
    if n < 10_000_000:
        cur, peak = memory(lambda: None, build)
    nq = 100 if n < 10_000_000 else 20
    qs = np.random.default_rng(7).choice(n, nq, replace=False)
    qsets = [sets.get(int(i)) for i in qs]
    qsigs = [mh.signature(q) for q in qsets]
    def q_lsh(_):
        out = []
        for i, q, sg in zip(qs, qsets, qsigs):
            cand = lsh.candidates(sg)
            cand = cand[cand != i]
            sims = sets.jaccard_subset(q, cand) if cand.size else np.empty(0)
            o = np.argsort(-sims)[:10]
            out.append((cand[o], sims[o], cand.size))
        return out
    def q_bf(_):
        return [brute_topk(sets, q, 10, exclude=int(i)) for i, q in zip(qs, qsets)]
    ml, sl, rl, res_l = timeit(lambda: None, q_lsh, n)
    mb, sb_, rb, res_b = timeit(lambda: None, q_bf, n, max_reps=20)
    rec_rel, rec10, cand = [], [], []
    for (il, jl, nc), (ib, jb) in zip(res_l, res_b):
        cand.append(nc)
        rel = set(ib[jb >= 0.5].tolist())
        if rel:
            rec_rel.append(len(rel & set(il.tolist())) / len(rel))
        pos = set(ib[jb > 0].tolist())
        if pos:
            rec10.append(len(pos & set(il.tolist())) / len(pos))
    REC.add(estructura=f"MinHash+LSH (k={k}, b={b}, r={r})", operacion="construir indice", n=n, reps=reps,
            media_s=round(mean, 5), std_s=round(std, 5), us_por_op=round(1e6 * mean / n, 3),
            mem_actual_mb=round(cur, 2), mem_pico_mb=round(peak, 2), umbral=round((1 / b) ** (1 / r), 3))
    REC.add(estructura=f"MinHash+LSH (k={k}, b={b}, r={r})", operacion="consulta top-10", n=n, reps=rl,
            media_s=round(ml, 6), std_s=round(sl, 6), us_por_op=round(1e6 * ml / nq, 2),
            recall_J05=round(float(np.mean(rec_rel)), 4) if rec_rel else "", recall_top10=round(float(np.mean(rec10)), 4),
            candidatos_prom=round(float(np.mean(cand)), 1), frac_comparada=round(float(np.mean(cand)) / n, 6))
    REC.add(estructura="Fuerza bruta Jaccard (NumPy, O(nnz))", operacion="consulta top-10", n=n, reps=rb,
            media_s=round(mb, 6), std_s=round(sb_, 6), us_por_op=round(1e6 * mb / nq, 2),
            mem_actual_mb=round(sets.memory_bytes() / 2**20, 2), recall_J05=1.0, recall_top10=1.0, candidatos_prom=n)


def lsh_curve(n=100_000):
    """Curva S: recall vs (b, r) para el informe."""
    indptr, indices, U = make_sets(n)
    sets = CSRSets(indptr, indices, U)
    qs = np.random.default_rng(7).choice(n, 100, replace=False)
    bf = [brute_topk(sets, sets.get(int(i)), 10, exclude=int(i)) for i in qs]
    for b, r in [(32, 2), (16, 4), (8, 8), (64, 2), (32, 4)]:
        mh = MinHasher(b * r, seed=1)
        lsh = LSHIndex(b, r); t = time.perf_counter(); lsh.build_from_csr(mh, indptr, indices); tb = time.perf_counter() - t
        rr, cc, tq = [], [], 0.0
        for i, (ib, jb) in zip(qs, bf):
            q = sets.get(int(i)); t = time.perf_counter()
            cand = lsh.candidates(mh.signature(q)); cand = cand[cand != i]
            sims = sets.jaccard_subset(q, cand) if cand.size else np.empty(0)
            top = cand[np.argsort(-sims)[:10]]; tq += time.perf_counter() - t
            rel = set(ib[jb >= 0.5].tolist())
            if rel:
                rr.append(len(rel & set(top.tolist())) / len(rel))
            cc.append(cand.size)
        REC.add(estructura=f"LSH curva b={b} r={r}", operacion="ajuste", n=n, reps=1, umbral=round((1 / b) ** (1 / r), 3),
                media_s=round(tq / 100, 6), recall_J05=round(float(np.mean(rr)), 4), candidatos_prom=round(float(np.mean(cc)), 1),
                construir_s=round(tb, 3))


def make_vectors(n, d=32, seed=SEED):
    rng = np.random.default_rng(seed + n)
    C = max(10, int(np.sqrt(n)))
    cent = rng.normal(0, 1, (C, d))
    X = cent[rng.integers(0, C, n)] + rng.normal(0, 0.45, (n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def bench_hnsw(n, M=16, efc=100):
    X = make_vectors(n)
    t0 = time.perf_counter()
    h = HNSW(32, M=M, ef_construction=efc, max_elements=n, seed=1)
    h.add_items(X)
    tb = time.perf_counter() - t0
    if n <= 100_000:
        def b2(_):
            hh = HNSW(32, M=M, ef_construction=efc, max_elements=n, seed=1); hh.add_items(X); return hh
        cur, peak = memory(lambda: None, b2)
    else:
        cur = h.memory_bytes() / 2**20 + 0.0
        peak = float("nan")
    rng = np.random.default_rng(3)
    Q = X[rng.choice(n, 100, replace=False)] + rng.normal(0, 0.02, (100, 32)).astype(np.float32)
    exact = [knn_brute_force(X, q, 10)[1] for q in Q]
    REC.add(estructura=f"HNSW (M={M}, efC={efc})", operacion="construir", n=n, reps=1, media_s=round(tb, 4),
            std_s=0, us_por_op=round(1e6 * tb / n, 2), mem_actual_mb=round(cur, 2), mem_pico_mb=round(peak, 2),
            niveles=h.max_level + 1)
    for ef in ([10, 20, 50, 100, 200] if n in (100_000,) else [50]):
        mq, sq, rq, res = timeit(lambda: None, lambda _: [h.knn_query(q, 10, ef=ef)[1] for q in Q], n)
        rec = np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(res, exact)])
        REC.add(estructura=f"HNSW (M={M}, efC={efc})", operacion=f"consulta k=10 ef={ef}", n=n, reps=rq,
                media_s=round(mq, 6), std_s=round(sq, 6), us_por_op=round(1e6 * mq / 100, 2), recall_10=round(float(rec), 4))
    mb, sb, rb, _ = timeit(lambda: None, lambda _: [knn_brute_force(X, q, 10) for q in Q], n)
    REC.add(estructura="KNN exacto fuerza bruta (NumPy)", operacion="consulta k=10", n=n, reps=rb, media_s=round(mb, 6),
            std_s=round(sb, 6), us_por_op=round(1e6 * mb / 100, 2), recall_10=1.0, mem_actual_mb=round(X.nbytes / 2**20, 2))
    kd = KDTree(X, 32)
    kd.nodes_visited = 0
    Qk = Q[:20]
    mk, sk, rk, resk = timeit(lambda: None, lambda _: [kd.query(q, 10)[1] for q in Qk], n, max_reps=10)
    REC.add(estructura="KD-Tree 32D (exacto)", operacion="consulta k=10", n=n, reps=rk, media_s=round(mk, 6),
            std_s=round(sk, 6), us_por_op=round(1e6 * mk / 20, 2), recall_10=1.0,
            nodos_visitados=round(kd.nodes_visited / ((rk + 1) * 20), 1), nodos_totales=len(kd.dim))


if __name__ == "__main__":
    save_platform()
    sizes, only = sizes_from_argv()
    if not only or "lsh" in only:
        for n in sizes:
            print(f"[lsh] n={n:,}", flush=True); bench_lsh(n)
    if not only or "curve" in only:
        print("[lsh curva]", flush=True); lsh_curve()
    if not only or "hnsw" in only:
        for n in sizes:
            if n > int(float(__import__("os").environ.get("HNSW_MAX", 100_000))):
                continue
            print(f"[hnsw] n={n:,}", flush=True); bench_hnsw(n)
