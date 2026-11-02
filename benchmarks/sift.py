"""The Rust core against hnswlib and FAISS on real image descriptors (a SIFT1M subset).

Uses the first ``--n`` base vectors and ``--queries`` queries of SIFT1M (128 dims,
http://corpus-texmex.irisa.fr/). Ground truth is recomputed exactly on the subset.
Both HNSW indexes use the same M and ef_construction; build and single-query timings
run on one thread so the comparison is like for like, and hnswlib's multi-threaded
build plus the Rust core's parallel batch search are reported separately.

Requires ``make rust-develop`` and ``uv sync --group validation``.

    uv run python benchmarks/sift.py --data /path/to/sift --n 100000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import hnswlib
import numpy as np

from ann_vector_search_library.backend import RustHNSW, rust_available


def read_vecs(path: Path, dtype: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(dtype)


def recall(pred: list[list[int]] | np.ndarray, truth: np.ndarray, k: int) -> float:
    pairs = zip(pred, truth, strict=True)
    hits = sum(len(set(map(int, p[:k])) & set(map(int, t[:k]))) for p, t in pairs)
    return hits / (k * len(truth))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not rust_available():
        raise SystemExit("build the extension first: make rust-develop")

    k, efs = 10, (16, 64, 256)
    base = read_vecs(args.data / "sift_base.fvecs", "float32")[: args.n]
    queries = read_vecs(args.data / "sift_query.fvecs", "float32")[: args.queries]
    flat = faiss.IndexFlatL2(base.shape[1])
    start = time.perf_counter()
    flat.add(base)
    faiss_build = time.perf_counter() - start
    faiss.omp_set_num_threads(1)
    start = time.perf_counter()
    for q in queries:
        flat.search(q[None, :], k)
    faiss_ms = (time.perf_counter() - start) / len(queries) * 1000
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    _, truth = flat.search(queries, k)

    rows: list[dict] = [
        {
            "library": "FAISS flat (exact)",
            "build_s": faiss_build,
            "results": [{"ef": None, "recall": 1.0, "ms": faiss_ms}],
        }
    ]
    print(
        f"n={len(base):,} dim={base.shape[1]} queries={len(queries)} m={args.m} "
        f"efc={args.ef_construction} cpus={os.cpu_count()}"
    )
    print(f"faiss flat: query {faiss_ms:.3f} ms")

    for threads in (1, os.cpu_count() or 1):
        index = hnswlib.Index(space="l2", dim=base.shape[1])
        index.init_index(
            max_elements=len(base), M=args.m, ef_construction=args.ef_construction, random_seed=0
        )
        index.set_num_threads(threads)
        start = time.perf_counter()
        index.add_items(base, np.arange(len(base)))
        build = time.perf_counter() - start
        print(f"hnswlib build ({threads} threads): {build:.1f} s")
        if threads != 1:
            rows.append(
                {"library": f"hnswlib, {threads}-thread build", "build_s": build, "results": []}
            )
            continue
        index.set_num_threads(1)
        results = []
        for ef in efs:
            index.set_ef(ef)
            start = time.perf_counter()
            pred = [index.knn_query(q, k=k)[0][0] for q in queries]
            ms = (time.perf_counter() - start) / len(queries) * 1000
            results.append({"ef": ef, "recall": recall(pred, truth, k), "ms": ms})
            print(f"  hnswlib ef={ef}: recall {results[-1]['recall']:.3f}, {ms:.3f} ms")
        rows.append({"library": "hnswlib", "build_s": build, "results": results})

    vectors = base.astype(np.float64).tolist()
    qlist = queries.astype(np.float64).tolist()
    for precision in ("f64", "f32"):
        rs = RustHNSW(
            dim=base.shape[1], m=args.m, ef_construction=args.ef_construction, precision=precision
        )
        start = time.perf_counter()
        for i, v in enumerate(vectors):
            rs.add(i, v)
        build = time.perf_counter() - start
        print(f"rust core {precision} build (1 thread): {build:.1f} s")
        results = []
        for ef in efs:
            start = time.perf_counter()
            pred = [[i for i, _ in rs.search(q, k, ef=ef)] for q in qlist]
            ms = (time.perf_counter() - start) / len(qlist) * 1000
            start = time.perf_counter()
            batch = rs.search_batch(qlist, k, ef=ef)
            batch_ms = (time.perf_counter() - start) / len(qlist) * 1000
            assert [[i for i, _ in r] for r in batch] == pred
            results.append(
                {"ef": ef, "recall": recall(pred, truth, k), "ms": ms, "batch_ms": batch_ms}
            )
            print(
                f"  rust {precision} ef={ef}: recall {results[-1]['recall']:.3f}, {ms:.3f} ms, "
                f"batch {batch_ms:.4f} ms/query"
            )
        rows.append(
            {
                "library": f"ann-vector-search, Rust core {precision}",
                "build_s": build,
                "results": results,
            }
        )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "n": len(base),
                    "queries": len(queries),
                    "m": args.m,
                    "ef_construction": args.ef_construction,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
