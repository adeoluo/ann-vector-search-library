# Evaluation Report - ANN Vector Search Library

> Reproduce with `python -m ann_vector_search_library benchmark` and `benchmarks/benchmark.py`.
> Datasets and queries are generated deterministically.

## Setup

- **Dataset:** 2000 vectors in 16 dimensions drawn from eight Gaussian clusters.
- **Queries:** 100 held-out vectors from the same kind of distribution.
- **Index:** HNSW with M 8 and ef_construction 64.
- **Metric:** squared Euclidean distance (orders neighbors identically to Euclidean).
- **Primary metrics:** recall@10 against exact search, and distance computations per query for the
  index versus brute force.
- **Hardware / cost:** CPU only; building the index is a few seconds. No accelerators.

## Recall and cost by search width

| Search width (ef) | Recall@10 | HNSW distance ops/query | Brute ops/query | Reduction |
|---|---:|---:|---:|---:|
| 16 | 0.755 | 146 | 2000 | 13.7x |
| 32 | 0.871 | 208 | 2000 | 9.6x |
| 64 | 0.945 | 280 | 2000 | 7.1x |
| 128 | 0.969 | 377 | 2000 | 5.3x |

The search width ``ef`` moves smoothly along a recall/cost curve. At ``ef`` 64 the index reaches
0.945 recall while doing 7.1x fewer distance computations than an exhaustive scan.

## Scaling behavior

Because graph search touches a sublinear number of points while brute force touches all of them,
the reduction grows with dataset size: on 1000 vectors the same configuration shows roughly 4.5x
at 0.974 recall, and on 2000 vectors it shows 7.1x at 0.945 recall. The absolute comparison counts
for HNSW grow slowly with ``n``.

## Correctness

- The brute-force index is exact and serves as recall ground truth; a dedicated test confirms it
  returns the true nearest neighbors on a hand-checked example.
- HNSW results are always sorted by distance, deterministic under a fixed seed, and recall is
  monotonic in ``ef`` on the benchmark datasets.

## Service performance

Under a concurrent stress benchmark (3000 requests, 8 workers) the `/search` handler sustains
roughly 600 queries per second with p50 near 1.4 ms. Pure-Python distance math dominates absolute
latency; the algorithmic win is in the number of comparisons.

## Reproducibility

- Deterministic given a seed: datasets, index construction, and queries all reproduce.
- No network access, no external data.
- Regenerate every number with the CLI and `benchmarks/benchmark.py`.
