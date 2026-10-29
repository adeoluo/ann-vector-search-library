"""Deterministic synthetic vector datasets with cluster structure.

Vectors are drawn from a mixture of Gaussian clusters, which gives the data real
neighborhood structure for a nearest-neighbor index to exploit. Generation is
seeded, so datasets and queries are reproducible.
"""

from __future__ import annotations

import random

Vector = list[float]


def make_vectors(
    count: int, dim: int, *, clusters: int = 8, seed: int = 0, spread: float = 0.15
) -> list[Vector]:
    """Return ``count`` vectors drawn from ``clusters`` Gaussian blobs."""

    rng = random.Random(seed)
    centers = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(clusters)]
    vectors: list[Vector] = []
    for _ in range(count):
        center = centers[rng.randrange(clusters)]
        vectors.append([c + rng.gauss(0.0, spread) for c in center])
    return vectors


def make_queries(count: int, dim: int, *, clusters: int = 8, seed: int = 1) -> list[Vector]:
    """Return query vectors drawn from the same kind of distribution."""

    return make_vectors(count, dim, clusters=clusters, seed=seed)
