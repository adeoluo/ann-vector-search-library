"""A large-workload scaling benchmark for the HNSW index.

Builds the index over synthetic datasets of increasing size and reports build
time, per-query latency against an exact scan, recall, and peak memory. This is
synthetic data and a pure-Python implementation, so the numbers measure how the
algorithm scales, not competitive wall-clock latency against a compiled library
such as FAISS or hnswlib. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import tracemalloc

from ann_vector_search_library.evaluation import timed_comparison


def main() -> None:
    print(
        "HNSW at scale (synthetic clustered vectors, dim 16, ef 64; "
        "pure-Python, not competitive on wall-clock with compiled libraries):"
    )
    for count in (1000, 2000, 4000, 8000):
        tracemalloc.start()
        report = timed_comparison(count=count, dim=16, k=10, ef=64, query_count=50, seeds=1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        throughput = 1000.0 / report.hnsw_query_ms if report.hnsw_query_ms else 0.0
        print(
            f"  n={count:>6}: build {report.build_seconds:6.2f}s, "
            f"index query {report.hnsw_query_ms:7.3f} ms "
            f"({throughput:7.0f} queries/sec), exact scan {report.exact_query_ms:8.3f} ms, "
            f"peak {peak / (1024 * 1024):6.1f} MiB, recall@10 {report.recall:.3f}"
        )


if __name__ == "__main__":
    main()
