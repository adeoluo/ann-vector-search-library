"""ANN Vector Search Library - HNSW approximate nearest neighbor search from scratch."""

from __future__ import annotations

from ann_vector_search_library.backend import RustHNSW, create_index, rust_available
from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.distance import DistanceCounter, cosine_distance, squared_euclidean
from ann_vector_search_library.evaluation import RecallReport, recall_at_k, recall_curve
from ann_vector_search_library.hnsw import HNSW

__version__ = "0.3.0"

__all__ = [
    "HNSW",
    "BruteForce",
    "DistanceCounter",
    "RecallReport",
    "RustHNSW",
    "__version__",
    "cosine_distance",
    "create_index",
    "make_queries",
    "make_vectors",
    "recall_at_k",
    "recall_curve",
    "rust_available",
    "squared_euclidean",
]
