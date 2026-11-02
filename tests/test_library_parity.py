"""Independent recall parity against hnswlib and FAISS.

At matched HNSW parameters the from-scratch index should reach essentially the
same recall as hnswlib (the reference HNSW implementation), and FAISS flat should
be exact. This validates that the graph construction and search are correct, not
merely internally consistent. hnswlib, faiss, and numpy are optional ``validation``
dependencies, so this test is skipped when they are not installed; the core
library does not require them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hnswlib")
pytest.importorskip("faiss")
pytest.importorskip("numpy")

from benchmarks.compare_libraries import compare


def test_from_scratch_matches_hnswlib_recall() -> None:
    results = {
        r.name: r for r in compare(count=3000, queries=100, m=16, ef_construction=200, ef=64)
    }
    faiss_flat = next(r for name, r in results.items() if name.startswith("faiss"))
    hnsw_lib = next(r for name, r in results.items() if name.startswith("hnswlib"))
    ours = next(r for name, r in results.items() if name.startswith("from-scratch"))

    # FAISS flat is exact.
    assert faiss_flat.recall == pytest.approx(1.0)
    # Both approximate indexes should reach high recall at these parameters.
    assert hnsw_lib.recall > 0.9
    assert ours.recall > 0.9
    # The from-scratch recall should track the reference implementation closely,
    # confirming the HNSW graph and search are correct.
    assert abs(ours.recall - hnsw_lib.recall) < 0.05
