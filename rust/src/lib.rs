//! Rust core of ann-vector-search-library: an HNSW index that reproduces the Python one.
//!
//! Given the same insertion order and the same per-node levels (drawn by the Python
//! side from its seeded `random.Random`), construction makes the same distance calls in
//! the same order and builds the same graph, so searches return the same neighbors and
//! the same distance counts. Heaps break distance ties by node id, as Python's `heapq`
//! does on `(distance, id)` tuples. `tests/test_rust_parity.py` checks this.

pub mod distance;
pub mod hnsw;

#[cfg(feature = "python")]
mod python;

pub use distance::Metric;
pub use hnsw::Hnsw;
