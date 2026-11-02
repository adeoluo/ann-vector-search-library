"""Build and query speed of the Python and Rust HNSW on the same data and settings.

Both indexes receive the same vectors in the same order with the same seed, so they
build the same graph; the script checks that their answers agree. Requires
`make rust-develop`.
"""

from __future__ import annotations

import os
import time

from ann_vector_search_library.backend import RustHNSW, rust_available
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.hnsw import HNSW

DIM, K, EF = 32, 10, 64


def main() -> None:
    if not rust_available():
        raise SystemExit("build the extension first: make rust-develop")
    print(f"cpus={os.cpu_count()} dim={DIM} m=8 ef_construction=64 k={K} ef={EF}")
    print(
        "| vectors | build py s | build rust s | speedup | query py ms | query rust ms "
        "| speedup | batch rust ms/query | identical |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for n in (2_000, 10_000):
        vectors = make_vectors(n, DIM, seed=0)
        queries = make_queries(200, DIM, seed=1)
        timings = {}
        indexes = {}
        for name, cls in (("py", HNSW), ("rs", RustHNSW)):
            index = cls(dim=DIM)
            start = time.perf_counter()
            for i, v in enumerate(vectors):
                index.add(i, v)
            timings[f"build_{name}"] = time.perf_counter() - start
            start = time.perf_counter()
            results = [index.search(q, K, ef=EF) for q in queries]
            timings[f"query_{name}"] = (time.perf_counter() - start) / len(queries) * 1000
            indexes[name] = (index, results)
        rs_index, rs_results = indexes["rs"]
        start = time.perf_counter()
        batch = rs_index.search_batch(queries, K, ef=EF)
        batch_ms = (time.perf_counter() - start) / len(queries) * 1000
        same = rs_results == indexes["py"][1] and batch == rs_results
        print(
            f"| {n:,} | {timings['build_py']:.1f} | {timings['build_rs']:.2f} | "
            f"{timings['build_py'] / timings['build_rs']:.0f}x | {timings['query_py']:.2f} | "
            f"{timings['query_rs']:.3f} | {timings['query_py'] / timings['query_rs']:.0f}x | "
            f"{batch_ms:.4f} | {same} |"
        )


if __name__ == "__main__":
    main()
