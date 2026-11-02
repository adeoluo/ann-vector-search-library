# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.0] 2026-11-02

### Features

- **rust:** Rust HNSW core that reproduces the Python index exactly
- **backend:** RustHNSW drop-in with parity tests in CI

### Documentation

- Standard README structure and language framing (#6)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Rust core benchmark, usage, ADR 0004 and diagram
- List the Rust modules in AGENTS.md and llms.txt

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Add release automation, AGENTS.md, llms.txt and contributor docs

## [0.1.0] 2026-11-01

### Features

- Implement HNSW approximate nearest neighbor index (#1)
- Cross-library recall comparison against hnswlib and FAISS (#5)

### Documentation

- Add explicit comparison scope and honest baseline framing (#2)

### Maintenance

- Initialize repository with license and overview
- Remove Dependabot configuration

### Bench

- Add wall-clock comparison against an exact scan (#3)
- Add large-workload scale benchmark (#4)
