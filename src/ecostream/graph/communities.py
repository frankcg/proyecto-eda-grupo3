"""Etapa 8 [FIJA] - Detección de comunidades.

* **Louvain** (Blondel et al., 2008) - algoritmo elegido.
  Fase 1 (movimientos locales): cada nodo se mueve a la comunidad vecina que
  maximiza la ganancia de modularidad
      ΔQ(i→c) ∝ k_{i,c} − γ · Σ_tot(c) · k_i / (2m)
  hasta que ningún movimiento mejora.  Fase 2 (agregación): cada comunidad
  se colapsa en un super-nodo (pesos internos -> lazo) y se repite.
  Complejidad empírica ~O(m log n); memoria O(n + m).
* **Label Propagation** (Raghavan, Albert & Kumara, 2007) - alternativa más
  simple: cada nodo adopta la etiqueta de mayor peso entre sus vecinos
  (asíncrono, orden aleatorio) hasta que no hay cambios.  O(m) por iteración,
  no optimiza ninguna función objetivo y es sensible al orden (inestable).

Modularidad (Newman, 2006):
    Q = Σ_c [ L_c / m − γ (S_c / 2m)² ]
con L_c = peso de aristas internas y S_c = suma de fuerzas de la comunidad.
"""
from __future__ import annotations

import random

import numpy as np

from .graph import WeightedGraph


def modularity(g: WeightedGraph, labels: np.ndarray, resolution: float = 1.0) -> float:
    labels = np.asarray(labels)
    m2 = 2.0 * g.w.sum()
    if m2 == 0:
        return 0.0
    same = labels[g.u] == labels[g.v]
    k = labels.max() + 1
    L = np.bincount(labels[g.u][same], weights=g.w[same], minlength=k)
    S = np.bincount(labels, weights=g.strength, minlength=k)
    return float((2 * L / m2).sum() - resolution * ((S / m2) ** 2).sum())


def _relabel(labels) -> np.ndarray:
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64)


def louvain(g: WeightedGraph, resolution: float = 1.0, seed: int = 0,
            max_levels: int = 20, tol: float = 1e-9):
    """Devuelve (etiquetas por nodo, niveles)."""
    rnd = random.Random(seed)
    # Representación de trabajo: adj[i] = {j: w}, self_w[i] = peso de lazo
    adj = [dict() for _ in range(g.n)]
    for a, b, w in zip(g.u.tolist(), g.v.tolist(), g.w.tolist()):
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w
    self_w = [0.0] * g.n
    membership = list(range(g.n))           # nodo original -> super-nodo
    m2 = 2.0 * g.w.sum()
    if m2 == 0:
        return np.arange(g.n), 0
    levels = 0
    for _ in range(max_levels):
        n = len(adj)
        k = [sum(adj[i].values()) + 2 * self_w[i] for i in range(n)]
        comm = list(range(n))
        tot = k[:]
        improved_any = False
        order = list(range(n))
        moved = True
        while moved:
            moved = False
            rnd.shuffle(order)
            for i in order:
                ci = comm[i]
                ki = k[i]
                # pesos hacia comunidades vecinas
                wc: dict[int, float] = {}
                for j, w in adj[i].items():
                    cj = comm[j]
                    wc[cj] = wc.get(cj, 0.0) + w
                tot[ci] -= ki
                best_c = ci
                best_gain = wc.get(ci, 0.0) - resolution * tot[ci] * ki / m2
                for c, w in wc.items():
                    gain = w - resolution * tot[c] * ki / m2
                    if gain > best_gain + tol:
                        best_gain, best_c = gain, c
                tot[best_c] += ki
                if best_c != ci:
                    comm[i] = best_c
                    moved = True
                    improved_any = True
        if not improved_any:
            break
        levels += 1
        # --- fase 2: agregación
        new_id = {}
        for c in comm:
            if c not in new_id:
                new_id[c] = len(new_id)
        nc = len(new_id)
        new_adj = [dict() for _ in range(nc)]
        new_self = [0.0] * nc
        for i in range(n):
            ci = new_id[comm[i]]
            new_self[ci] += self_w[i]
            for j, w in adj[i].items():
                if j < i:
                    continue
                cj = new_id[comm[j]]
                if ci == cj:
                    new_self[ci] += w
                else:
                    new_adj[ci][cj] = new_adj[ci].get(cj, 0.0) + w
                    new_adj[cj][ci] = new_adj[cj].get(ci, 0.0) + w
        membership = [new_id[comm[s]] for s in membership]
        adj, self_w = new_adj, new_self
        if nc == n:
            break
    return np.asarray(membership, dtype=np.int64), levels


def label_propagation(g: WeightedGraph, seed: int = 0, max_iter: int = 100):
    rnd = random.Random(seed)
    adj = g.adjacency_lists()
    labels = list(range(g.n))
    order = list(range(g.n))
    it = 0
    for it in range(1, max_iter + 1):
        rnd.shuffle(order)
        changed = 0
        for i in order:
            if not adj[i]:
                continue
            score: dict[int, float] = {}
            for j, w in adj[i]:
                lj = labels[j]
                score[lj] = score.get(lj, 0.0) + w
            best = max(score.values())
            cands = [l for l, s in score.items() if s == best]
            if labels[i] in cands:
                continue
            labels[i] = cands[0] if len(cands) == 1 else rnd.choice(cands)
            changed += 1
        if changed == 0:
            break
    return _relabel(labels), it


def community_stats(labels: np.ndarray) -> dict:
    sizes = np.bincount(_relabel(labels))
    sizes = np.sort(sizes)[::-1]
    return {"n_communities": int(len(sizes)), "largest": int(sizes[0]),
            "median_size": float(np.median(sizes)),
            "singletons": int((sizes == 1).sum()), "sizes": sizes}


def nmi(a, b) -> float:
    """Información mutua normalizada (media aritmética), implementación propia."""
    a = _relabel(a); b = _relabel(b)
    n = len(a)
    ka, kb = a.max() + 1, b.max() + 1
    pairs, cnt = np.unique(a * kb + b, return_counts=True)   # tabla de contingencia dispersa
    ia, ib = pairs // kb, pairs % kb
    pa = np.bincount(a, minlength=ka) / n
    pb = np.bincount(b, minlength=kb) / n
    pij = cnt / n
    mi = (pij * np.log(pij / (pa[ia] * pb[ib]))).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    if ha + hb == 0:
        return 1.0
    return float(2 * mi / (ha + hb))
