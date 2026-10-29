"""A deterministic micro-benchmark for the HNSW index.

Builds an index over a synthetic clustered dataset and reports, at several search
widths, recall against exact search and the number of distance computations per
query. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.evaluation import recall_curve
from ann_vector_search_library.hnsw import HNSW


def main() -> None:
    count, dim, k = 2000, 16, 10
    vectors = make_vectors(count, dim, seed=0)
    queries = make_queries(100, dim, seed=1)

    index = HNSW(dim=dim, m=8, ef_construction=64, seed=0)
    brute = BruteForce(dim=dim)
    start = time.perf_counter()
    for node_id, vector in enumerate(vectors):
        index.add(node_id, vector)
        brute.add(node_id, vector)
    build_s = time.perf_counter() - start

    print(f"Built HNSW over {count} vectors (dim {dim}) in {build_s:.1f} s")
    print(f"Recall@{k} and distance-computation cost against brute force:")
    for report in recall_curve(index, brute, queries, k, [16, 32, 64, 128]):
        print(f"  {report.format()}")


if __name__ == "__main__":
    main()
