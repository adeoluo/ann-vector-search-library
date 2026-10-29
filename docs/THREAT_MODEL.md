# Threat Model - ANN Vector Search Library

## Assets
- The HTTP service's availability and the integrity of its responses.

## Trust boundaries
- The only external input is the JSON body and path of a request. The index and datasets are
  in-process and trusted.

## Entry points and mitigations

### `POST /search`
- **Malformed JSON** is rejected with 400 before routing.
- **Validation:** the query vector must be a list of exactly the index dimension with numeric
  components; ``k`` and ``ef`` are range-checked; booleans are rejected explicitly. Invalid input
  returns 400 with a message, never a stack trace.
- **Bounded work:** ``k`` and ``ef`` are capped, so a single request cannot force an unbounded
  search.

### `GET` routes
- `/health`, `/ready`, `/recall`, and `/metrics` take no user input and return fixed-shape
  payloads. `/metrics` exposes only aggregate counters.

## Non-threats (by construction)
- **No secrets:** the service holds no credentials and makes no outbound calls.
- **No persistence:** state is the in-memory index; nothing is written on the request path.
- **No code execution:** inputs are numbers; no user-supplied string is evaluated.

## Supply chain
- Zero runtime dependencies. Development tooling is pinned via `uv.lock`. CodeQL and dependency
  scanning run in CI.

## Residual risk
- A flood of searches could exhaust CPU; a production deployment would add a rate limiter and run
  behind a reverse proxy.
