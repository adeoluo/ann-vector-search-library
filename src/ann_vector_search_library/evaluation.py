"""Evaluation: recall against exact search and the recall/cost trade-off."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.distance import Vector
from ann_vector_search_library.hnsw import HNSW


@dataclass(frozen=True)
class RecallReport:
    """Recall and distance-computation cost for one search configuration."""

    ef: int
    recall: float
    hnsw_distances_per_query: float
    brute_distances_per_query: float

    @property
    def speedup(self) -> float:
        if self.hnsw_distances_per_query == 0:
            return 0.0
        return self.brute_distances_per_query / self.hnsw_distances_per_query

    def format(self) -> str:
        return (
            f"ef={self.ef}: recall={self.recall:.4f}, "
            f"{self.hnsw_distances_per_query:.0f} vs {self.brute_distances_per_query:.0f} "
            f"distance ops/query ({self.speedup:.1f}x fewer)"
        )


def recall_at_k(
    index: HNSW, brute: BruteForce, queries: list[Vector], k: int, *, ef: int
) -> RecallReport:
    """Return recall@k of the index against exact search, with cost counts."""

    index.counter.reset()
    brute.counter.reset()
    hits = 0
    for query in queries:
        truth = {node_id for node_id, _ in brute.search(query, k)}
        got = {node_id for node_id, _ in index.search(query, k, ef=ef)}
        hits += len(truth & got)
    total = k * len(queries) if queries else 1
    n = len(queries) if queries else 1
    return RecallReport(
        ef=ef,
        recall=hits / total,
        hnsw_distances_per_query=index.counter.count / n,
        brute_distances_per_query=brute.counter.count / n,
    )


def recall_curve(
    index: HNSW, brute: BruteForce, queries: list[Vector], k: int, ef_values: list[int]
) -> list[RecallReport]:
    """Return a recall/cost report for each ``ef`` in ``ef_values``."""

    return [recall_at_k(index, brute, queries, k, ef=ef) for ef in ef_values]


@dataclass(frozen=True)
class TimedComparison:
    """Wall-clock comparison of the index against an exact scan at a dataset size.

    Distance-computation counts show the algorithmic saving, but wall-clock time
    is what a caller experiences. Pure-Python interpreter overhead means fewer
    distance computations do not automatically translate into lower latency, so
    this report measures real time, averaged over several seeds for stability.
    """

    count: int
    dim: int
    seeds: int
    build_seconds: float
    hnsw_query_ms: float
    exact_query_ms: float
    recall: float

    @property
    def query_speed_ratio(self) -> float:
        """Exact query time divided by index query time (>1 means the index is faster)."""

        if self.hnsw_query_ms == 0:
            return 0.0
        return self.exact_query_ms / self.hnsw_query_ms

    def format(self) -> str:
        return (
            f"n={self.count:>6}: build {self.build_seconds:6.2f}s, "
            f"index query {self.hnsw_query_ms:7.3f} ms, exact scan {self.exact_query_ms:7.3f} ms "
            f"({self.query_speed_ratio:4.1f}x), recall@k {self.recall:.3f}"
        )


def timed_comparison(
    *,
    count: int,
    dim: int,
    k: int,
    ef: int,
    m: int = 8,
    ef_construction: int = 64,
    query_count: int = 50,
    seeds: int = 3,
) -> TimedComparison:
    """Average build time, index-query time, exact-scan time, and recall over seeds.

    The exact scan is this library's own pure-Python :class:`BruteForce`, which
    keeps the comparison dependency-free and apples-to-apples (same interpreter,
    same metric). It is an exact baseline, not an optimized compiled library.
    """

    from ann_vector_search_library.data import make_queries, make_vectors

    build_total = 0.0
    hnsw_total = 0.0
    exact_total = 0.0
    recall_total = 0.0
    for seed in range(seeds):
        vectors = make_vectors(count, dim, seed=seed)
        queries = make_queries(query_count, dim, seed=1000 + seed)
        index = HNSW(dim=dim, m=m, ef_construction=ef_construction, seed=seed)
        brute = BruteForce(dim=dim)
        start = time.perf_counter()
        for node_id, vector in enumerate(vectors):
            index.add(node_id, vector)
        build_total += time.perf_counter() - start
        for node_id, vector in enumerate(vectors):
            brute.add(node_id, vector)

        start = time.perf_counter()
        index_results = [index.search(query, k, ef=ef) for query in queries]
        hnsw_total += (time.perf_counter() - start) / len(queries) * 1000.0

        start = time.perf_counter()
        exact_results = [brute.search(query, k) for query in queries]
        exact_total += (time.perf_counter() - start) / len(queries) * 1000.0

        hits = 0
        for got, truth in zip(index_results, exact_results, strict=True):
            got_ids = {node_id for node_id, _ in got}
            truth_ids = {node_id for node_id, _ in truth}
            hits += len(got_ids & truth_ids)
        recall_total += hits / (k * len(queries))

    return TimedComparison(
        count=count,
        dim=dim,
        seeds=seeds,
        build_seconds=build_total / seeds,
        hnsw_query_ms=hnsw_total / seeds,
        exact_query_ms=exact_total / seeds,
        recall=recall_total / seeds,
    )
