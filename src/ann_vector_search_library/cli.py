"""Command line interface for the HNSW vector search library."""

from __future__ import annotations

import argparse

from ann_vector_search_library.brute import BruteForce
from ann_vector_search_library.data import make_queries, make_vectors
from ann_vector_search_library.evaluation import recall_curve
from ann_vector_search_library.hnsw import HNSW


def _build(args: argparse.Namespace) -> tuple[HNSW, BruteForce, list[list[float]]]:
    vectors = make_vectors(args.count, args.dim, seed=args.seed)
    index = HNSW(dim=args.dim, m=args.m, ef_construction=args.ef_construction, seed=args.seed)
    brute = BruteForce(dim=args.dim)
    for node_id, vector in enumerate(vectors):
        index.add(node_id, vector)
        brute.add(node_id, vector)
    queries = make_queries(args.queries, args.dim, seed=args.seed + 1)
    return index, brute, queries


def _cmd_benchmark(args: argparse.Namespace) -> None:
    index, brute, queries = _build(args)
    print(f"Built HNSW over {len(index)} vectors (dim {args.dim}, M {args.m}).")
    print("Recall and distance-computation cost against exact search:")
    for report in recall_curve(index, brute, queries, args.k, [16, 32, 64, 128]):
        print(f"  {report.format()}")


def _cmd_search(args: argparse.Namespace) -> None:
    index, brute, queries = _build(args)
    query = queries[0]
    approx = index.search(query, args.k, ef=args.ef_construction)
    exact = brute.search(query, args.k)
    print(f"Query 0, top {args.k} (id, distance):")
    print(f"  HNSW:  {[(nid, round(d, 3)) for nid, d in approx]}")
    print(f"  exact: {[(nid, round(d, 3)) for nid, d in exact]}")


def _cmd_serve(args: argparse.Namespace) -> None:
    from ann_vector_search_library.service import serve

    serve(host=args.host, port=args.port)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ann-vector-search-library", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="report recall and cost against exact search")
    _add_common(bench)

    search = sub.add_parser("search", help="show approximate vs exact neighbors for one query")
    _add_common(search)

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {"benchmark": _cmd_benchmark, "search": _cmd_search, "serve": _cmd_serve}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
