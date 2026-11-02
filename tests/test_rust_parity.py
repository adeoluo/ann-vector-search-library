"""The Rust HNSW must build the same graph and answer the same as the Python HNSW.

Skipped when ``ann_vector_search_rs`` is not built, unless ``ANN_REQUIRE_RUST=1`` (CI).
"""

from __future__ import annotations

import os
import random

import pytest

from ann_vector_search_library import backend
from ann_vector_search_library.backend import RustHNSW, rust_available
from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.hnsw import HNSW

if os.environ.get("ANN_REQUIRE_RUST") == "1":
    assert rust_available(), "ANN_REQUIRE_RUST=1 but ann_vector_search_rs is not importable"
pytestmark = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")


def build(cls: type, vectors: list[list[float]], **kwargs: object) -> object:
    index = cls(dim=len(vectors[0]), **kwargs)
    for i, v in enumerate(vectors):
        index.add(i, v)
    return index


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize(("m", "ef_construction", "seed"), [(4, 16, 0), (8, 64, 1), (16, 32, 7)])
def test_same_neighbors_and_distance_counts(
    metric: str, m: int, ef_construction: int, seed: int
) -> None:
    vectors = make_vectors(600, 12, seed=seed)
    queries = make_queries(40, 12, seed=seed + 100)
    kwargs = {"m": m, "ef_construction": ef_construction, "metric": metric, "seed": seed}
    py = build(HNSW, vectors, **kwargs)
    rs = build(RustHNSW, vectors, **kwargs)
    assert len(rs) == len(py) == 600  # type: ignore[arg-type]
    assert rs.counter.count == py.counter.count  # type: ignore[attr-defined]
    for ef in (None, 10, 100):
        py.counter.reset()  # type: ignore[attr-defined]
        rs.counter.reset()  # type: ignore[attr-defined]
        for q in queries:
            assert rs.search(q, 10, ef=ef) == py.search(q, 10, ef=ef)  # type: ignore[attr-defined]
        assert rs.counter.count == py.counter.count  # type: ignore[attr-defined]


def test_batch_equals_single_and_recall() -> None:
    vectors = make_vectors(2000, 16, seed=3)
    queries = make_queries(100, 16, seed=4)
    rs = build(RustHNSW, vectors, m=12, ef_construction=100)
    brute = build(BruteForce, vectors)
    batch = rs.search_batch(queries, 10, ef=64)  # type: ignore[attr-defined]
    assert batch == [rs.search(q, 10, ef=64) for q in queries]  # type: ignore[attr-defined]
    hits = sum(
        len({i for i, _ in got} & {i for i, _ in brute.search(q, 10)})  # type: ignore[attr-defined]
        for q, got in zip(queries, batch, strict=True)
    )
    assert hits / 1000 > 0.95


def test_errors_match_python() -> None:
    rs = RustHNSW(dim=3)
    assert rs.search([0.0, 0.0, 0.0], 3) == []
    rs.add(1, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="duplicate node id"):
        rs.add(1, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected dimension 3"):
        rs.add(2, [1.0])
    with pytest.raises(ValueError, match="expected dimension 3"):
        rs.search([1.0], 1)
    with pytest.raises(ValueError, match="unknown metric"):
        RustHNSW(dim=3, metric="dot")


def test_arbitrary_ids_and_random_order() -> None:
    rng = random.Random(9)
    vectors = make_vectors(300, 8, seed=9)
    ids = rng.sample(range(-(10**9), 10**9), len(vectors))
    py, rs = HNSW(dim=8, seed=2), RustHNSW(dim=8, seed=2)
    for i, v in zip(ids, vectors, strict=True):
        py.add(i, v)
        rs.add(i, v)
    for q in make_queries(20, 8, seed=10):
        assert rs.search(q, 5) == py.search(q, 5)


def test_missing_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_EXTENSION", None)
    with pytest.raises(RuntimeError, match="not installed"):
        RustHNSW(dim=2)
