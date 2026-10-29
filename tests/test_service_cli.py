"""Tests for the HTTP service routing and the CLI."""

from __future__ import annotations

from ann_vector_search_library.cli import main
from ann_vector_search_library.data import make_queries
from ann_vector_search_library.service import SearchService, handle

_SERVICE = SearchService.built(count=300, dim=8, seed=0)


def test_health_and_ready() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200 and payload["size"] == 300 and payload["dim"] == 8
    status, payload = handle("GET", "/ready", {}, _SERVICE)
    assert status == 200 and payload["status"] == "ready"


def test_search_route() -> None:
    query = make_queries(1, 8, seed=1)[0]
    status, payload = handle("POST", "/search", {"vector": query, "k": 5}, _SERVICE)
    assert status == 200
    assert len(payload["results"]) == 5
    distances = [r["distance"] for r in payload["results"]]
    assert distances == sorted(distances)


def test_search_validation() -> None:
    for body in (
        {"vector": [0.0, 0.0]},
        {"vector": ["x"] * 8},
        {"vector": [0.0] * 8, "k": 0},
        {"vector": [0.0] * 8, "k": 999},
        {"vector": [0.0] * 8, "ef": 1},
    ):
        status, payload = handle("POST", "/search", body, _SERVICE)
        assert status == 400 and "error" in payload


def test_recall_route() -> None:
    status, payload = handle("GET", "/recall", {}, _SERVICE)
    assert status == 200
    assert 0.0 <= payload["recall"] <= 1.0
    assert payload["speedup"] > 1.0


def test_unknown_route_and_metrics() -> None:
    status, _ = handle("GET", "/nope", {}, _SERVICE)
    assert status == 404
    handle("GET", "/recall", {}, _SERVICE)
    assert "service_requests_total" in _SERVICE.metrics.render_prometheus()


def test_cli_benchmark_and_search(capsys) -> None:  # type: ignore[no-untyped-def]
    common = ["--count", "300", "--dim", "8", "--queries", "10"]
    main(["benchmark", *common])
    main(["search", *common])
    out = capsys.readouterr().out
    assert "Recall and distance-computation cost" in out
    assert "HNSW:" in out
