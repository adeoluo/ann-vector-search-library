"""ANN Vector Search Library - HNSW approximate nearest neighbor search from scratch."""

from __future__ import annotations

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.distance import DistanceCounter, cosine_distance, squared_euclidean
from ann_vector_search_library.evaluation import RecallReport, recall_at_k, recall_curve
from ann_vector_search_library.hnsw import HNSW

__version__ = "0.1.0"

__all__ = [
    "HNSW",
    "BruteForce",
    "DistanceCounter",
    "RecallReport",
    "__version__",
    "cosine_distance",
    "make_queries",
    "make_vectors",
    "recall_at_k",
    "recall_curve",
    "squared_euclidean",
]
