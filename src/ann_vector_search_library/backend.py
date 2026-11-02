"""The Rust backend: ``RustHNSW``, a drop-in for ``HNSW`` backed by the compiled
``ann_vector_search_rs`` extension that every install builds into this package.

``create_index`` is the entry point: it returns the fastest index available, the Rust core
in ``f32`` mode, and falls back to the Python ``HNSW`` only if the extension is missing (for
example a source checkout where it has not been compiled).

The Rust index reproduces the Python one: ``RustHNSW`` draws each node's level from the
same seeded ``random.Random`` with the same formula and passes it in, and the Rust code
makes the same distance calls in the same order. Given the same inserts, both build the
same graph and return the same neighbors and distance counts (when no two distances are
exactly equal). ``search_batch`` runs queries in parallel with the GIL released.

``precision="f32"`` stores 32-bit floats and sums distances in SIMD lanes instead. It runs
the same algorithm faster in half the memory, but the reordered sums round differently,
so it does not reproduce the Python index and is checked on recall instead.
"""

from __future__ import annotations

import importlib
import math
import random
from typing import Any, Protocol

from ann_vector_search_library.distance import Vector, get_metric
from ann_vector_search_library.hnsw import HNSW


def _load_extension() -> Any:
    # Inside the package for normal installs; top level for an older `maturin develop`.
    for name in ("ann_vector_search_library.ann_vector_search_rs", "ann_vector_search_rs"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
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
    """``HNSW`` with the graph and searches in Rust. Same constructor and methods, plus
    ``precision`` ("f64" reproduces the Python index, "f32" trades that for speed)."""

    def __init__(
        self,
        dim: int,
        m: int = 8,
        ef_construction: int = 64,
        metric: str = "euclidean",
        seed: int = 0,
        precision: str = "f64",
    ) -> None:
        if _EXTENSION is None:
            raise RuntimeError(
                "the Rust backend is not built; run `uv sync` or `make rust-develop`"
            )
        get_metric(metric)  # same validation and error as the Python index
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.metric = metric
        self.seed = seed
        self.precision = precision
        self._core = _EXTENSION.HNSW(dim, m, ef_construction, metric, precision)
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


class CostCounter(Protocol):
    """Counts distance computations; ``DistanceCounter`` and the Rust counter both fit."""

    @property
    def count(self) -> int: ...

    def reset(self) -> None: ...


class VectorIndex(Protocol):
    """What callers need from an index: the Python ``HNSW`` and ``RustHNSW`` both fit."""

    @property
    def counter(self) -> CostCounter: ...

    def __len__(self) -> int: ...

    def add(self, node_id: int, vector: Vector) -> None: ...

    def search(
        self, query: Vector, k: int, *, ef: int | None = None
    ) -> list[tuple[int, float]]: ...


BACKENDS = ("auto", "rust", "python")


def create_index(
    dim: int,
    *,
    m: int = 8,
    ef_construction: int = 64,
    metric: str = "euclidean",
    seed: int = 0,
    backend: str = "auto",
    precision: str = "f32",
) -> VectorIndex:
    """Return an HNSW index on the fastest available backend.

    ``backend="auto"`` (the default) uses the Rust core when the extension is present,
    else the Python index; ``"rust"`` requires the extension and ``"python"`` forces the
    reference implementation. ``precision`` applies to the Rust core: ``"f32"`` (the
    default, fastest, same recall) or ``"f64"`` (bit-identical to the Python index).
    """

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "python" or (backend == "auto" and not rust_available()):
        return HNSW(dim=dim, m=m, ef_construction=ef_construction, metric=metric, seed=seed)
    return RustHNSW(
        dim=dim,
        m=m,
        ef_construction=ef_construction,
        metric=metric,
        seed=seed,
        precision=precision,
    )
