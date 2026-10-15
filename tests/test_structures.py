"""Pruebas de corrección de las estructuras propias contra referencias exactas."""
import heapq
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecostream.structures.linked_queue import LinkedQueue
from ecostream.structures.sorting import merge_sort, bisect_left, binary_search, dedup_sorted
from ecostream.structures.heap import BinaryHeap, top_k
from ecostream.structures.hash_table import HashTable
from ecostream.structures.lru_cache import LRUCache
from ecostream.structures.sparse_matrix import SparseMatrix
from ecostream.structures.bplus_tree import BPlusTree
from ecostream.structures.kd_tree import KDTree, knn_brute_force
from ecostream.structures.bloom_filter import BloomFilter
from ecostream.structures.count_min_sketch import CountMinSketch, WindowedCountMin
from ecostream.structures.minhash_lsh import MinHasher, LSHIndex, CSRSets, jaccard, lsh_topk, brute_topk
from ecostream.structures.hnsw import HNSW
from ecostream.structures.trie import Trie, linear_autocomplete
from ecostream.graph.graph import WeightedGraph, planted_partition_graph
from ecostream.graph.pagerank import pagerank, personalized_pagerank
from ecostream.graph.communities import louvain, label_propagation, modularity, nmi


def test_linked_queue():
    q = LinkedQueue()
    for i in range(100):
        q.enqueue(i)
    assert q.dequeue() == 0
    assert q.dequeue_batch(10) == list(range(1, 11))
    assert len(q) == 89
    assert list(q)[:2] == [11, 12]
    q2 = LinkedQueue(capacity=2)
    assert q2.enqueue(1) and q2.enqueue(2) and not q2.enqueue(3)


def test_sorting_and_binary_search():
    rnd = random.Random(1)
    for n in [0, 1, 5, 17, 100, 1000]:
        xs = [(rnd.randint(0, 50), i) for i in range(n)]
        assert merge_sort(xs, key=lambda t: t[0]) == sorted(xs, key=lambda t: t[0])  # estable
    a = sorted(rnd.sample(range(10000), 500))
    for x in range(0, 10000, 37):
        import bisect
        assert bisect_left(a, x) == bisect.bisect_left(a, x)
        assert binary_search(a, x) == (x in set(a))
    assert dedup_sorted([1, 1, 2, 3, 3, 3], key=lambda x: x) == [1, 2, 3]


def test_heap_matches_heapq():
    rnd = random.Random(2)
    h, ref = BinaryHeap(), []
    for i in range(2000):
        p = rnd.randint(0, 100)
        h.push(p, i); heapq.heappush(ref, (p, i))
        if rnd.random() < 0.3:
            assert h.pop() == tuple(heapq.heappop(ref))
    while ref:
        assert h.pop() == tuple(heapq.heappop(ref))
    vals = [(rnd.random(), i) for i in range(1000)]
    assert [x[1] for x in top_k(vals, 5)] == [x[1] for x in sorted(vals, reverse=True)[:5]]


def test_hash_table():
    rnd = random.Random(3)
    t, ref = HashTable(), {}
    for _ in range(20000):
        k = rnd.randint(0, 5000)
        op = rnd.random()
        if op < 0.6:
            t.put(k, k * 2); ref[k] = k * 2
        elif op < 0.8:
            assert t.delete(k) == (k in ref); ref.pop(k, None)
        else:
            assert t.get(k) == ref.get(k)
    assert len(t) == len(ref)
    assert dict(t.items()) == ref
    t2 = HashTable(); t2["a"] = 1
    assert "a" in t2 and t2["a"] == 1


def test_lru():
    evicted = []
    c = LRUCache(2, on_evict=lambda k, v: evicted.append(k))
    c.put(1, "a"); c.put(2, "b"); c.get(1); c.put(3, "c")
    assert evicted == [2] and 1 in c and 3 in c


def test_sparse():
    s = SparseMatrix(3, 4)
    s.add(0, 1); s.add(0, 1); s.add(2, 3, 5)
    indptr, idx, data = s.to_csr()
    assert s.get(0, 1) == 2 and s.nnz == 2
    assert list(indptr) == [0, 1, 1, 2] and list(idx) == [1, 3]
    assert s.column_sets() == {1: [0], 3: [2]}


def test_bplus_tree_vs_dict():
    rnd = random.Random(4)
    t = BPlusTree(pool_pages=8, max_keys=8)     # orden pequeño + pool pequeño fuerza E/S
    ref = {}
    for _ in range(5000):
        k = rnd.randint(0, 100000)
        t.insert(k, k + 1); ref[k] = k + 1
    for k in rnd.sample(range(100000), 2000):
        assert t.search(k) == ref.get(k)
    got = list(t.range(1000, 20000))
    assert got == sorted((k, v) for k, v in ref.items() if 1000 <= k <= 20000)
    assert t.io["disk_reads"] > 0 and t.height >= 3
    t.close()
    t2 = BPlusTree(max_keys=16)
    items = [(i * 2, i) for i in range(3000)]
    t2.bulk_load(items)
    assert t2.search(200) == 100 and t2.search(201) is None
    assert list(t2.range(10, 20)) == [(10, 5), (12, 6), (14, 7), (16, 8), (18, 9), (20, 10)]
    t2.insert(201, -1)
    assert t2.search(201) == -1
    t2.close()


def test_kdtree_exact():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5000, 4)).astype(np.float32)
    kd = KDTree(X, leaf_size=16)
    for q in rng.normal(size=(50, 4)):
        d1, i1 = kd.query(q, 10)
        d2, i2 = knn_brute_force(X, q, 10)
        assert np.allclose(d1, d2, atol=1e-4)


def test_bloom_no_false_negatives_and_fpr():
    bf = BloomFilter(10000, 0.01)
    keys = np.arange(10000, dtype=np.uint64) * 7919
    bf.add_many(keys)
    assert bf.contains_many(keys).all()
    others = np.arange(10**6, 10**6 + 50000, dtype=np.uint64)
    fpr = bf.contains_many(others).mean()
    assert fpr < 0.02
    bf2 = BloomFilter(100, 0.01)
    bf2.add("tarjeta-123")
    assert "tarjeta-123" in bf2
    # escalar y vectorizado son consistentes
    bf3 = BloomFilter(1000, 0.01)
    for k in range(500):
        bf3.add(k)
    assert bf3.contains_many(np.arange(500, dtype=np.uint64)).all()


def test_count_min_never_underestimates():
    rng = np.random.default_rng(6)
    keys = rng.zipf(1.3, 100000) % 5000
    cms = CountMinSketch(epsilon=0.001, delta=0.01)
    cms.update_many(keys.astype(np.uint64))
    true = np.bincount(keys, minlength=5000)
    est = cms.estimate_many(np.arange(5000, dtype=np.uint64))
    assert (est >= true).all()
    assert ((est - true) <= cms.error_bound()).mean() >= 0.99
    c2 = CountMinSketch(0.01, 0.01)
    for _ in range(5):
        c2.update("x")
    assert c2.estimate("x") >= 5
    assert c2.estimate_many(np.array([__import__("ecostream.structures.hashing", fromlist=["x"]).key_to_u64("x")], dtype=np.uint64))[0] == c2.estimate("x")
    w = WindowedCountMin(window_seconds=60, n_buckets=6)
    for t in range(0, 60):
        w.update("tok", t)
    assert w.estimate("tok") >= 60
    w.update("otro", 200)          # pasan >60 s: la ventana expira
    assert w.estimate("tok") == 0


def test_minhash_lsh():
    mh = MinHasher(128, seed=1)
    a = list(range(0, 100)); b = list(range(20, 120))
    est = float(np.mean(mh.signature(a) == mh.signature(b)))
    assert abs(est - jaccard(a, b)) < 0.12
    rng = np.random.default_rng(7)
    sets = []
    for c in range(200):
        base = rng.choice(10000, 30, replace=False)
        for _ in range(5):
            s = base.copy(); s[rng.integers(0, 30, 3)] = rng.integers(0, 10000, 3)
            sets.append(np.unique(s))
    indptr = np.cumsum([0] + [len(s) for s in sets]); idx = np.concatenate(sets)
    cs = CSRSets(indptr, idx, 10000)
    mh2 = MinHasher(64, seed=3)
    sigs = mh2.signatures_csr(indptr, idx)
    assert (sigs[3] == mh2.signature(sets[3])).all()
    lsh = LSHIndex(16, 4); lsh.build(sigs)
    rec = []
    for qi in range(0, 1000, 50):
        ids_b, _ = brute_topk(cs, sets[qi], 4, exclude=qi)
        ids_l, _, _ = lsh_topk(lsh, cs, sigs[qi], sets[qi], 4, exclude=qi)
        rec.append(len(set(ids_b) & set(ids_l)) / 4)
    assert np.mean(rec) > 0.9


def test_hnsw_recall():
    rng = np.random.default_rng(8)
    centers = rng.normal(size=(20, 16))
    X = (centers[rng.integers(0, 20, 3000)] + 0.3 * rng.normal(size=(3000, 16))).astype(np.float32)
    h = HNSW(16, M=12, ef_construction=80, max_elements=3000)
    h.add_items(X)
    rec = []
    for q in X[rng.integers(0, 3000, 50)] + 0.05:
        _, ia = h.knn_query(q, 10, ef=50)
        _, ib = knn_brute_force(X, q, 10)
        rec.append(len(set(ia) & set(ib)) / 10)
    assert np.mean(rec) > 0.9


def test_trie():
    t = Trie(k=3)
    for name, w in [("Smartphone Samsung", 10), ("Smart TV LG", 30), ("Smartwatch Xiaomi", 20), ("Laptop HP", 5)]:
        t.insert(name, name, w)
    assert [x[1] for x in t.autocomplete("smart")] == ["Smart TV LG", "Smartwatch Xiaomi", "Smartphone Samsung"]
    t.update_weight("Smartphone Samsung", 100)
    assert t.autocomplete("sm")[0][1] == "Smartphone Samsung"
    terms = [("Smartphone Samsung", 100), ("Smart TV LG", 30), ("Smartwatch Xiaomi", 20), ("Laptop HP", 5)]
    assert [x[1] for x in linear_autocomplete(terms, "smart", 3)] == [x[1] for x in t.autocomplete("smart")]


def test_graph_algorithms_vs_networkx():
    import networkx as nx
    g, truth = planted_partition_graph(600, 6000, n_comms=8, seed=9)
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_weighted_edges_from(zip(g.u.tolist(), g.v.tolist(), g.w.tolist()))
    pr, _ = pagerank(g)
    pr_nx = nx.pagerank(G, alpha=0.85, tol=1e-12, max_iter=500)
    assert np.allclose(pr, [pr_nx[i] for i in range(g.n)], atol=1e-6)
    ppr, _ = personalized_pagerank(g, [0, 1])
    assert abs(ppr.sum() - 1) < 1e-6
    lab, _ = louvain(g, seed=1)
    comms = {}
    for i, c in enumerate(lab):
        comms.setdefault(c, set()).add(i)
    q_nx = nx.algorithms.community.modularity(G, comms.values())
    assert abs(modularity(g, lab) - q_nx) < 1e-9
    ref = nx.algorithms.community.louvain_communities(G, seed=1)
    q_ref = nx.algorithms.community.modularity(G, ref)
    assert modularity(g, lab) > q_ref - 0.02
    assert nmi(lab, truth) > 0.6
    lp, _ = label_propagation(g, seed=1)
    assert modularity(g, lp) > 0.0
    assert abs(nmi(lab, lab) - 1) < 1e-9
