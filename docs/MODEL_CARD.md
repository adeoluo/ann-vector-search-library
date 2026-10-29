# Model / System Card - ANN Vector Search Library

## Overview
- **What it is:** a from-scratch HNSW approximate nearest neighbor index with a brute-force exact
  baseline, in pure Python with no numerical libraries.
- **Base model:** none. The index is a data structure built over user-provided vectors.
- **Version and date:** 0.1.0, 2026.

## Intended use
- **Primary use case:** approximate nearest neighbor retrieval for embeddings, and a readable
  reference implementation of HNSW.
- **Primary users:** engineers building retrieval or recommendation systems.

## Out-of-scope / prohibited use
- Not an optimized production vector database; pure-Python distance math is slow per comparison.
- Not persistent; the index is in-memory with no deletion support.

## Performance
- Recall@10 up to 0.969 against exact search, at 5 to 14x fewer distance computations than a scan.
- Full method and tables: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Limitations and failure modes
- Recall depends on M and ef; defaults suit the bundled datasets and may need tuning elsewhere.
- High-dimensional, unclustered data reduces the advantage of any graph index.

## Ethical considerations
- No personal data, no network access. Vectors are synthetic and generated in-process.
