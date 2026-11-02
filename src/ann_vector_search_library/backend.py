"""Optional Rust backend: ``RustHNSW``, a drop-in for ``HNSW`` backed by ``ann_vector_search_rs``.

The Rust index reproduces the Python one: ``RustHNSW`` draws each node's level from the
same seeded ``random.Random`` with the same formula and passes it in, and the Rust code
makes the same distance calls in the same order. Given the same inserts, both build the
same graph and return the same neighbors and distance counts (when no two distances are
exactly equal). ``search_batch`` runs queries in parallel with the GIL released.
"""

from __future__ import annotations

import importlib
import math
import random
from typing import Any

from ann_vector_search_library.distance import Vector, get_metric


def _load_extension() -> Any:
    try:
        return importlib.import_module("ann_vector_search_rs")
    except ImportError:
        return None


_EXTENSION: Any = _load_extension()


def rust_available() -> bool:
    """Return whether the Rust extension is importable."""

    return _EXTENSION is not None


class _RustCounter:
    """Reads the Rust index's distance counter through the ``DistanceCounter`` interface."""

    def __init__(self, core: Any) -> None:
        self._core = core

    @property
    def count(self) -> int:
        return int(self._core.distance_count)

    def reset(self) -> None:
        self._core.reset_count()


class RustHNSW:
    """``HNSW`` with the graph and searches in Rust. Same constructor and methods."""

    def __init__(
        self,
        dim: int,
        m: int = 8,
        ef_construction: int = 64,
        metric: str = "euclidean",
        seed: int = 0,
    ) -> None:
        if _EXTENSION is None:
            raise RuntimeError("the Rust backend is not installed; run `make rust-develop`")
        get_metric(metric)  # same validation and error as the Python index
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.metric = metric
        self.seed = seed
        self._core = _EXTENSION.HNSW(dim, m, ef_construction, metric)
        self._rng = random.Random(seed)
        self._level_mult = 1.0 / math.log(m) if m > 1 else 1.0
        self.counter = _RustCounter(self._core)

    def __len__(self) -> int:
        return len(self._core)

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random() or 1e-12) * self._level_mult)

    def add(self, node_id: int, vector: Vector) -> None:
        """Insert ``vector`` under ``node_id``."""

        if len(vector) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {len(vector)}")
        level = self._random_level()
        self._core.add(node_id, vector, level)

    def search(self, query: Vector, k: int, *, ef: int | None = None) -> list[tuple[int, float]]:
        """Return the ``k`` approximate nearest ``(node_id, distance)`` pairs."""

        return list(self._core.search(query, k, ef))

    def search_batch(
        self, queries: list[Vector], k: int, *, ef: int | None = None
    ) -> list[list[tuple[int, float]]]:
        """Search many queries in parallel."""

        return [list(r) for r in self._core.search_batch(queries, k, ef)]
