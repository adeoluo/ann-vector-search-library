# ADR-0003: Measure distance computations, do not assert speed

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The value of an approximate index is that it touches fewer points than a brute-force scan. In pure
Python, wall-clock time is noisy and dominated by interpreter overhead, so a timing number would
be a weak and unstable claim.

## Decision
Count distance computations directly. Both the HNSW index and the brute-force baseline share a
distance counter, and every recall report includes comparisons per query for each. The headline
metric is recall against exact search paired with the reduction in distance computations.

## Consequences
- The speed-recall trade-off is concrete and reproducible, independent of machine speed.
- The reduction (for example 7x fewer comparisons at 0.945 recall) is a property of the algorithm,
  not of the benchmark hardware.
- Absolute latency is still reported from the service stress benchmark for context, but the
  primary claim rests on measured comparison counts.
