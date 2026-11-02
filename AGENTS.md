# AGENTS.md

Working notes for coding agents and contributors in **ANN Vector Search Library**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

An approximate nearest neighbor vector search library built from scratch in Python with a Rust core, implementing HNSW (Hierarchical Navigable Small World) graphs, with recall and distance-computation cost measured against exact search.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/distance.rs`: Distance functions, summed in the same order as the Python implementation.
- `rust/src/hnsw.rs`: HNSW (Malkov and Yashunin, 2016), mirroring `ann_vector_search_library.hnsw.HNSW`.
- `rust/src/lib.rs`: Rust core of ann-vector-search-library: an HNSW index that reproduces the Python one.
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `ann_vector_search_rs` extension module.
- `src/ann_vector_search_library/__init__.py`: ANN Vector Search Library - HNSW approximate nearest neighbor search from scratch.
- `src/ann_vector_search_library/backend.py`: Optional Rust backend: ``RustHNSW``, a drop-in for ``HNSW`` backed by ``ann_vector_search_rs``.
- `src/ann_vector_search_library/brute.py`: Exact nearest-neighbor search by brute force, for recall ground truth.
- `src/ann_vector_search_library/cli.py`: Command line interface for the HNSW vector search library.
- `src/ann_vector_search_library/data.py`: Deterministic synthetic vector datasets with cluster structure.
- `src/ann_vector_search_library/distance.py`: Distance functions over plain Python float vectors.
- `src/ann_vector_search_library/evaluation.py`: Evaluation: recall against exact search and the recall/cost trade-off.
- `src/ann_vector_search_library/hnsw.py`: Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index.
- `src/ann_vector_search_library/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/ann_vector_search_library/service.py`: A standard-library HTTP service exposing the HNSW index.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Build the Rust extension into .venv (opt-in rust group) |
| `make parity` | Rust vs Python parity tests (requires rust-develop) |
| `make bench-compare` | Python vs Rust build and query speed (requires rust-develop) |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the HTTP service |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut by the maintainer with `make cut-release`, never from CI.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
