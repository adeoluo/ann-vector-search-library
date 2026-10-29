"""Tests for distance functions, the HNSW index, and recall against brute force."""

from __future__ import annotations

import pytest

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.distance import cosine_distance, get_metric, squared_euclidean
from ann_vector_search_library.evaluation import recall_at_k, recall_curve
from ann_vector_search_library.hnsw import HNSW


def test_squared_euclidean_and_cosine() -> None:
    assert squared_euclidean([0.0, 0.0], [3.0, 4.0]) == 25.0
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_get_metric_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        get_metric("manhattan")


def test_add_validates_dimension_and_duplicates() -> None:
    index = HNSW(dim=3, seed=0)
    index.add(0, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="dimension"):
        index.add(1, [0.1, 0.2])
    with pytest.raises(ValueError, match="duplicate"):
        index.add(0, [0.4, 0.5, 0.6])


def test_search_on_empty_index_returns_empty() -> None:
    assert HNSW(dim=4, seed=0).search([0.0, 0.0, 0.0, 0.0], 5) == []


def test_single_element_index() -> None:
    index = HNSW(dim=2, seed=0)
    index.add(42, [1.0, 1.0])
    results = index.search([1.0, 1.1], 3)
    assert results[0][0] == 42


def test_results_sorted_by_distance() -> None:
    index = HNSW(dim=8, seed=0)
    for i, v in enumerate(make_vectors(200, 8, seed=0)):
        index.add(i, v)
    results = index.search(make_queries(1, 8, seed=1)[0], 10)
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_recall_improves_with_ef() -> None:
    vectors = make_vectors(400, 12, seed=0)
    queries = make_queries(30, 12, seed=1)
    index = HNSW(dim=12, m=8, ef_construction=64, seed=0)
    brute = BruteForce(dim=12)
    for i, v in enumerate(vectors):
        index.add(i, v)
        brute.add(i, v)
    low = recall_at_k(index, brute, queries, 10, ef=16)
    high = recall_at_k(index, brute, queries, 10, ef=128)
    assert high.recall >= low.recall
    assert high.recall > 0.9


def test_index_does_fewer_comparisons_than_brute() -> None:
    vectors = make_vectors(600, 12, seed=0)
    queries = make_queries(20, 12, seed=1)
    index = HNSW(dim=12, seed=0)
    brute = BruteForce(dim=12)
    for i, v in enumerate(vectors):
        index.add(i, v)
        brute.add(i, v)
    report = recall_at_k(index, brute, queries, 10, ef=64)
    assert report.hnsw_distances_per_query < report.brute_distances_per_query
    assert report.speedup > 1.5


def test_brute_force_is_exact() -> None:
    brute = BruteForce(dim=4)
    for i, v in enumerate([[0.0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 0, 0]]):
        brute.add(i, [float(x) for x in v])
    results = brute.search([0.1, 0.0, 0.0, 0.0], 2)
    assert results[0][0] == 0 and results[1][0] == 1


def test_determinism() -> None:
    vectors = make_vectors(100, 8, seed=0)

    def build() -> HNSW:
        idx = HNSW(dim=8, seed=0)
        for i, v in enumerate(vectors):
            idx.add(i, v)
        return idx

    q = make_queries(1, 8, seed=1)[0]
    assert build().search(q, 5) == build().search(q, 5)


def test_recall_curve_shape() -> None:
    vectors = make_vectors(200, 8, seed=0)
    queries = make_queries(10, 8, seed=1)
    index = HNSW(dim=8, seed=0)
    brute = BruteForce(dim=8)
    for i, v in enumerate(vectors):
        index.add(i, v)
        brute.add(i, v)
    reports = recall_curve(index, brute, queries, 10, [16, 64])
    assert len(reports) == 2
    assert "recall=" in reports[0].format()
