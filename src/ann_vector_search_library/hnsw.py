"""Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.

HNSW builds a layered proximity graph. Upper layers are sparse and let a search
travel quickly across the space; the dense bottom layer refines to the true
neighborhood. Insertion assigns each point a random maximum level from a geometric
distribution, greedily descends to that level, then runs a bounded best-first
search at each remaining layer and connects to a selected set of neighbors. Search
mirrors insertion: greedy descent through the upper layers, then a beam search of
width ``ef`` at the bottom.

This is a from-scratch implementation following Malkov and Yashunin (2016), in
pure Python with no numerical libraries. A distance counter records how many
comparisons each search makes so the saving over a brute-force scan is measurable.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

from ann_vector_search_library.distance import DistanceCounter, Vector, get_metric


@dataclass
class _Node:
    vector: Vector
    neighbors: list[list[int]]  # neighbors[level] = list of node ids


@dataclass
class HNSW:
    """An HNSW index over float vectors identified by integer ids."""

    dim: int
    m: int = 8
    ef_construction: int = 64
    metric: str = "euclidean"
    seed: int = 0
    counter: DistanceCounter = field(default_factory=DistanceCounter)

    def __post_init__(self) -> None:
        self._distance = get_metric(self.metric)
        self._rng = random.Random(self.seed)
        self._nodes: dict[int, _Node] = {}
        self._entry: int | None = None
        self._max_level = 0
        self._m_max0 = 2 * self.m
        self._level_mult = 1.0 / math.log(self.m) if self.m > 1 else 1.0

    def __len__(self) -> int:
        return len(self._nodes)

    def _dist(self, a: Vector, b: Vector) -> float:
        self.counter.count += 1
        return self._distance(a, b)

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random() or 1e-12) * self._level_mult)

    def add(self, node_id: int, vector: Vector) -> None:
        """Insert ``vector`` under ``node_id`` into the index."""

        if len(vector) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {len(vector)}")
        if node_id in self._nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        level = self._random_level()
        node = _Node(vector=vector, neighbors=[[] for _ in range(level + 1)])
        self._nodes[node_id] = node

        if self._entry is None:
            self._entry = node_id
            self._max_level = level
            return

        entry = self._entry
        # Greedy descent through layers above the new node's top level.
        for lc in range(self._max_level, level, -1):
            entry = self._greedy(vector, entry, lc)

        # Connect at each layer from min(level, max_level) down to 0.
        for lc in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(vector, [entry], lc, self.ef_construction)
            neighbors = self._select_neighbors(vector, candidates, self.m)
            node.neighbors[lc] = neighbors
            for neighbor_id in neighbors:
                self._connect(neighbor_id, node_id, lc)
            if candidates:
                entry = min(candidates, key=lambda nid: self._dist(vector, self._nodes[nid].vector))

        if level > self._max_level:
            self._max_level = level
            self._entry = node_id

    def _connect(self, source: int, target: int, level: int) -> None:
        node = self._nodes[source]
        node.neighbors[level].append(target)
        limit = self._m_max0 if level == 0 else self.m
        if len(node.neighbors[level]) > limit:
            pruned = self._select_neighbors(node.vector, node.neighbors[level], limit)
            node.neighbors[level] = pruned

    def _greedy(self, query: Vector, entry: int, level: int) -> int:
        current = entry
        current_dist = self._dist(query, self._nodes[current].vector)
        improved = True
        while improved:
            improved = False
            for neighbor in self._nodes[current].neighbors[level]:
                d = self._dist(query, self._nodes[neighbor].vector)
                if d < current_dist:
                    current_dist = d
                    current = neighbor
                    improved = True
        return current

    def _search_layer(self, query: Vector, entries: list[int], level: int, ef: int) -> list[int]:
        visited = set(entries)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        for e in entries:
            d = self._dist(query, self._nodes[e].vector)
            heapq.heappush(candidates, (d, e))
            heapq.heappush(results, (-d, e))
        while candidates:
            dist, current = heapq.heappop(candidates)
            worst = -results[0][0]
            if dist > worst and len(results) >= ef:
                break
            for neighbor in self._nodes[current].neighbors[level]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._dist(query, self._nodes[neighbor].vector)
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [nid for _, nid in results]

    def _select_neighbors(self, query: Vector, candidates: list[int], m: int) -> list[int]:
        """Select up to ``m`` neighbors using the distance heuristic.

        A candidate is kept only if it is closer to the query than to any already
        selected neighbor, which spreads connections around the point and avoids
        clustering all edges in one direction.
        """

        ordered = sorted(
            set(candidates), key=lambda nid: self._dist(query, self._nodes[nid].vector)
        )
        selected: list[int] = []
        for candidate in ordered:
            if len(selected) >= m:
                break
            cand_vec = self._nodes[candidate].vector
            cand_dist = self._dist(query, cand_vec)
            keep = True
            for chosen in selected:
                if self._dist(cand_vec, self._nodes[chosen].vector) < cand_dist:
                    keep = False
                    break
            if keep:
                selected.append(candidate)
        return selected

    def search(self, query: Vector, k: int, *, ef: int | None = None) -> list[tuple[int, float]]:
        """Return the ``k`` approximate nearest ``(node_id, distance)`` pairs."""

        if self._entry is None:
            return []
        if len(query) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {len(query)}")
        ef = ef if ef is not None else max(self.ef_construction, k)
        entry = self._entry
        for lc in range(self._max_level, 0, -1):
            entry = self._greedy(query, entry, lc)
        found = self._search_layer(query, [entry], 0, ef)
        scored = sorted(
            ((nid, self._distance(query, self._nodes[nid].vector)) for nid in found),
            key=lambda pair: pair[1],
        )
        return scored[:k]
