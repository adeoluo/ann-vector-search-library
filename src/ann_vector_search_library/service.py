"""A dependency-free HTTP service exposing the HNSW index.

The service builds an index over a synthetic dataset at startup, then answers
nearest-neighbor queries and reports recall against exact search. The routing
logic is a pure function tested without a socket.

Endpoints
---------
``GET  /health``   liveness and index size.
``GET  /ready``    readiness once the index is built.
``GET  /metrics``  per-route request, error, and latency metrics (Prometheus text).
``POST /search``   body ``{"vector": [float], "k": int?, "ef": int?}``; nearest neighbors.
``GET  /recall``   recall@k of the index against exact search on held-out queries.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.evaluation import recall_at_k
from ann_vector_search_library.hnsw import HNSW
from ann_vector_search_library.observability import ServiceMetrics

JsonObject = dict[str, Any]
_MAX_K = 100


@dataclass(frozen=True)
class SearchService:
    """Holds the built index and answers search and recall requests."""

    index: HNSW
    brute: BruteForce
    queries: list[list[float]]
    dim: int
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def built(cls, *, count: int = 1000, dim: int = 16, seed: int = 0) -> SearchService:
        vectors = make_vectors(count, dim, seed=seed)
        index = HNSW(dim=dim, seed=seed)
        brute = BruteForce(dim=dim)
        for node_id, vector in enumerate(vectors):
            index.add(node_id, vector)
            brute.add(node_id, vector)
        return cls(index=index, brute=brute, queries=make_queries(50, dim, seed=seed + 1), dim=dim)

    def search(self, vector: list[float], k: int, ef: int | None) -> JsonObject:
        results = self.index.search(vector, k, ef=ef)
        return {"results": [{"id": nid, "distance": round(d, 6)} for nid, d in results]}

    def recall(self, k: int, ef: int) -> JsonObject:
        report = recall_at_k(self.index, self.brute, self.queries, k, ef=ef)
        return {
            "k": k,
            "ef": ef,
            "recall": round(report.recall, 6),
            "speedup": round(report.speedup, 3),
            "hnsw_distances_per_query": round(report.hnsw_distances_per_query, 2),
            "brute_distances_per_query": round(report.brute_distances_per_query, 2),
        }


def _valid_vector(value: Any, dim: int) -> list[float] | str:
    if not isinstance(value, list) or len(value) != dim:
        return f"must be a list of {dim} numbers"
    out: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "components must be numbers"
        out.append(float(v))
    return out


def _valid_int(value: Any, low: int, high: int) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < low or value > high:
        return None
    return value


def _route(
    method: str, path: str, body: Mapping[str, Any], service: SearchService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "size": len(service.index), "dim": service.dim}
    if method == "GET" and path == "/recall":
        return 200, service.recall(10, 64)
    if method == "POST" and path == "/search":
        vector = _valid_vector(body.get("vector"), service.dim)
        if isinstance(vector, str):
            return 400, {"error": f"'vector' {vector}"}
        k = _valid_int(body.get("k", 10), 1, _MAX_K)
        if k is None:
            return 400, {"error": f"'k' must be an integer in 1..{_MAX_K}"}
        ef_raw = body.get("ef")
        ef = None
        if ef_raw is not None:
            ef = _valid_int(ef_raw, k, 500)
            if ef is None:
                return 400, {"error": "'ef' must be an integer in k..500"}
        return 200, service.search(vector, k, ef)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: SearchService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: SearchService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": SearchService.built()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"ann-vector-search-library service listening on http://{host}:{port}")
    server.serve_forever()
