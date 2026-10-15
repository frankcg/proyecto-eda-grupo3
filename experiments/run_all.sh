#!/usr/bin/env bash
# Reproduce todos los experimentos (≈3-4 h en 2 vCPU). Resultados en experiments/results, figuras en experiments/figures.
set -e
cd "$(dirname "$0")"
mkdir -p logs
# Cola A y B en paralelo (uno por núcleo) hasta 1M
( python3 bench_base.py --max=1e6 > logs/base.log 2>&1
  python3 bench_prob.py --max=1e6 > logs/prob.log 2>&1
  python3 bench_graph.py --max=1e6 > logs/graph.log 2>&1 ) &
( python3 bench_similarity.py --max=1e6 --only=lsh,curve > logs/lsh.log 2>&1
  HNSW_MAX=100000 python3 bench_similarity.py --max=1e5 --only=hnsw > logs/hnsw.log 2>&1
  python3 bench_pipeline.py --max=1e6 > logs/pipeline.log 2>&1 ) &
wait
# 10M secuencial (memoria)
python3 bench_base.py --min=1e7 --only=fifo,sort,heap,hash,lru,bptree,kdtree > logs/base10m.log 2>&1
python3 bench_prob.py --min=1e7 > logs/prob10m.log 2>&1
python3 bench_similarity.py --min=1e7 --only=lsh > logs/lsh10m.log 2>&1
python3 bench_graph.py --min=1e7 > logs/graph10m.log 2>&1
python3 bench_pipeline.py --min=1e7 > logs/pipeline10m.log 2>&1
HNSW_MAX=1000000 python3 bench_similarity.py --min=1e6 --max=1e6 --only=hnsw > logs/hnsw1m.log 2>&1
python3 plots.py
