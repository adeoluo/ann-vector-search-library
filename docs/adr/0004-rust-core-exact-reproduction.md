# ADR 0004: A Rust core that reproduces the Python index exactly

## Status

Accepted

## Context

The Python index does far fewer distance computations than an exact scan, but each one is
slow, so build and query times are far behind compiled libraries. A compiled backend should
not change what the index computes, otherwise the recall and cost numbers measured on the
Python index would no longer describe it.

## Decision

Port HNSW construction and search to Rust (`rust/`), exposed as `RustHNSW` with the same
constructor and methods as `HNSW`.

* Node levels come from the Python side: `RustHNSW` draws them from the same seeded
  `random.Random` with the same formula and passes each to the Rust `add`.
* The Rust code makes the same distance calls in the same order: greedy descent iterates the
  neighbor list it started with, beam search uses heaps ordered like Python `heapq` on
  `(distance, id)` tuples, and neighbor selection sorts by distance with the same counting.
* A distance counter in Rust follows the same accounting as `DistanceCounter`.
* `search_batch` runs queries in parallel with rayon, with the GIL released.

## Consequences

* Parity tests compare search results and distance counts call for call. Any divergence in
  the graph shows up as a different count, which is a much stronger check than recall alone.
* Exact ties in distance can be ordered differently (Python sorts tied ids by set iteration
  order, Rust by id). Random float data does not produce ties.
* The Rust core is 21 to 40 times faster than Python on the benchmark. Its distances stay
  serial `f64` sums, which rules out SIMD in the default mode. An opt-in `precision="f32"` mode
  runs the same algorithm with lane-parallel `f32` sums; it gives up bit-exact parity and is
  tested on recall instead (see the README's SIFT results).
