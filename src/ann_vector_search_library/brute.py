"""Exact nearest-neighbor search by brute force, for recall ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

from ann_vector_search_library.distance import DistanceCounter, Vector, get_metric


@dataclass
class BruteForce:
    """An exact index that scans every stored vector on each query."""

    dim: int
    metric: str = "euclidean"
    counter: DistanceCounter = field(default_factory=DistanceCounter)

    def __post_init__(self) -> None:
        self._distance = get_metric(self.metric)
        self._items: dict[int, Vector] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, node_id: int, vector: Vector) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {len(vector)}")
        self._items[node_id] = vector

    def search(self, query: Vector, k: int) -> list[tuple[int, float]]:
        """Return the exact ``k`` nearest ``(node_id, distance)`` pairs."""

        scored = []
        for node_id, vector in self._items.items():
            self.counter.count += 1
            scored.append((node_id, self._distance(query, vector)))
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]
