"""Distance functions over plain Python float vectors.

Squared Euclidean distance is used for ranking: it orders neighbors identically to
Euclidean distance while avoiding a square root per comparison. Cosine distance is
offered for normalized-embedding use. A module-level counter records how many
distance computations a search performs, which is the honest way to show that the
graph index touches far fewer points than a brute-force scan.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

Vector = list[float]
Metric = Callable[[Vector, Vector], float]


@dataclass
class DistanceCounter:
    """Counts distance computations so search cost can be reported, not guessed."""

    count: int = 0

    def reset(self) -> None:
        self.count = 0


def squared_euclidean(a: Vector, b: Vector) -> float:
    """Return the squared Euclidean distance between two equal-length vectors."""

    total = 0.0
    for x, y in zip(a, b, strict=True):
        diff = x - y
        total += diff * diff
    return total


def cosine_distance(a: Vector, b: Vector) -> float:
    """Return ``1 - cosine_similarity`` between two vectors."""

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


_METRICS: dict[str, Metric] = {"euclidean": squared_euclidean, "cosine": cosine_distance}


def get_metric(name: str) -> Metric:
    """Return the distance function for ``name`` ('euclidean' or 'cosine')."""

    try:
        return _METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric: {name!r}") from None
