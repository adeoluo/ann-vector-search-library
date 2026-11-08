# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.3.1] 2026-11-08

### Bug fixes

- Report the released version and keep version strings in step on release (#8)

### Documentation

- Describe the supply chain accurately (#5)
- Complete citation metadata and bump it on release (#7)

### Maintenance

- Latest toolchain and CI versions (#6)

## [0.3.0] 2026-11-02

### Features

- Build the Rust core into the package and use it by default (#4)

### Performance

- Faster exact Rust core, opt-in f32 SIMD mode, SIFT benchmark (#3)

### Documentation

- Remove authorship statements from docs and scripts
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)

### Maintenance

- Classify squash merges in the changelog by the commits they contain

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
