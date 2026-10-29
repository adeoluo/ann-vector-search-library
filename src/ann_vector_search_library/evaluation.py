"""Evaluation: recall against exact search and the recall/cost trade-off."""

from __future__ import annotations

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
