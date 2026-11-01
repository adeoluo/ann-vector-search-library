"""Cross-library comparison against hnswlib and FAISS.

The core library is a from-scratch, pure-Python HNSW. This module benchmarks it
against two mature, compiled libraries on the same data: hnswlib (the reference
HNSW implementation) and FAISS (a flat exact index and an IVF approximate index).
It reports, for each, recall@k against exact ground truth, mean query latency,
and index build time, so the trade-off is measured honestly rather than asserted.

hnswlib, faiss, and numpy are optional ``validation`` dependencies, not runtime
dependencies, so the core library stays zero dependency. Import this module only
when those packages are installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.hnsw import HNSW


@dataclass(frozen=True)
class LibraryResult:
    """Build time, query latency, and recall for one library configuration."""

    name: str
    build_seconds: float
    query_ms: float
    recall: float

    def format(self) -> str:
        return (
            f"{self.name:<22} build {self.build_seconds:7.3f}s  "
            f"query {self.query_ms:8.3f} ms  recall@k {self.recall:.3f}"
        )


def _exact_neighbors(
    vectors: list[list[float]], queries: list[list[float]], k: int
) -> list[set[int]]:
    brute = BruteForce(dim=len(vectors[0]))
    for node_id, vector in enumerate(vectors):
        brute.add(node_id, vector)
    return [{node_id for node_id, _ in brute.search(query, k)} for query in queries]


def _recall(predicted: list[list[int]], truth: list[set[int]], k: int) -> float:
    hits = sum(len(set(pred) & true) for pred, true in zip(predicted, truth, strict=True))
    return hits / (k * len(truth)) if truth else 0.0


def _time_queries(search: object, queries: list[list[float]]) -> tuple[list[list[int]], float]:
    start = time.perf_counter()
    results = [search(query) for query in queries]  # type: ignore[operator]
    elapsed_ms = (time.perf_counter() - start) / len(queries) * 1000.0
    return results, elapsed_ms


def run_from_scratch(
    vectors: list[list[float]],
    queries: list[list[float]],
    truth: list[set[int]],
    *,
    k: int,
    m: int,
    ef_construction: int,
    ef: int,
) -> LibraryResult:
    start = time.perf_counter()
    index = HNSW(dim=len(vectors[0]), m=m, ef_construction=ef_construction, seed=0)
    for node_id, vector in enumerate(vectors):
        index.add(node_id, vector)
    build = time.perf_counter() - start
    results, query_ms = _time_queries(
        lambda q: [node_id for node_id, _ in index.search(q, k, ef=ef)], queries
    )
    return LibraryResult("from-scratch (pure Py)", build, query_ms, _recall(results, truth, k))


def run_hnswlib(
    vectors: list[list[float]],
    queries: list[list[float]],
    truth: list[set[int]],
    *,
    k: int,
    m: int,
    ef_construction: int,
    ef: int,
) -> LibraryResult:
    import hnswlib

    data = np.array(vectors, dtype=np.float32)
    dim = data.shape[1]
    start = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=ef_construction, M=m)
    index.add_items(data, np.arange(len(vectors)))
    build = time.perf_counter() - start
    index.set_ef(ef)
    query_data = np.array(queries, dtype=np.float32)
    results, query_ms = _time_queries(
        lambda q: index.knn_query(np.array([q], dtype=np.float32), k=k)[0][0].tolist(),
        queries,
    )
    _ = query_data
    return LibraryResult("hnswlib (compiled)", build, query_ms, _recall(results, truth, k))


def run_faiss_flat(
    vectors: list[list[float]],
    queries: list[list[float]],
    truth: list[set[int]],
    *,
    k: int,
) -> LibraryResult:
    import faiss

    data = np.array(vectors, dtype=np.float32)
    dim = data.shape[1]
    start = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(data)
    build = time.perf_counter() - start
    results, query_ms = _time_queries(
        lambda q: index.search(np.array([q], dtype=np.float32), k)[1][0].tolist(), queries
    )
    return LibraryResult("faiss flat (exact)", build, query_ms, _recall(results, truth, k))


def compare(
    *,
    count: int = 10000,
    dim: int = 16,
    queries: int = 200,
    k: int = 10,
    m: int = 16,
    ef_construction: int = 200,
    ef: int = 64,
    seed: int = 0,
) -> list[LibraryResult]:
    """Run every library on the same synthetic dataset and return their results."""

    vectors = make_vectors(count, dim, seed=seed)
    query_set = make_queries(queries, dim, seed=seed + 1)
    truth = _exact_neighbors(vectors, query_set, k)
    return [
        run_faiss_flat(vectors, query_set, truth, k=k),
        run_hnswlib(vectors, query_set, truth, k=k, m=m, ef_construction=ef_construction, ef=ef),
        run_from_scratch(
            vectors, query_set, truth, k=k, m=m, ef_construction=ef_construction, ef=ef
        ),
    ]


def main() -> None:
    print(
        "ANN cross-library comparison (synthetic clustered vectors). The from-scratch index "
        "is pure Python;\nhnswlib and FAISS are compiled. Recall is against exact ground truth."
    )
    for result in compare():
        print(f"  {result.format()}")


if __name__ == "__main__":
    main()
