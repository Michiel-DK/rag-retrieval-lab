"""``ragexp`` command line. Subcommands are added one queue item at a time."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _cmd_diff(args: argparse.Namespace) -> int:
    from .compare import diff, format_diff

    d = diff(args.a, args.b, metric=args.metric, n_resamples=args.resamples, seed=args.seed)
    print(format_diff(d))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from . import check, methods, runs

    if bool(args.names) == bool(args.all):
        raise ValueError("give method names or --all, not both and not neither")
    if args.all:
        names = [n for n in sorted(methods.REGISTRY) if (runs._SCORES / f"{n}.json").is_file()]
    else:
        names = list(args.names)
    for n in names:  # fail on usage errors before the (slow) corpus load
        if n not in methods.REGISTRY:
            raise ValueError(f"unknown method {n!r}; registered: {', '.join(sorted(methods.REGISTRY))}")
        if not (runs._SCORES / f"{n}.json").is_file():
            raise FileNotFoundError(f"no committed scores for {n!r} at {runs._SCORES / (n + '.json')}")
    corpus = check.load_nfcorpus()
    results = [check.check_method(n, corpus) for n in names]
    for r in results:
        print(check.format_result(r))
    return 0 if all(r.ok for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragexp", description="Retrieval experiments on NFCorpus, predictions first.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="paired bootstrap between two committed score files (b - a)")
    d.add_argument("a", help="baseline run: name under results/scores/ or a path to a .json")
    d.add_argument("b", help="candidate run")
    d.add_argument("--metric", default="ndcg@10")
    d.add_argument("--resamples", type=int, default=10_000)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=_cmd_diff)

    c = sub.add_parser("check", help="recompute registered methods and compare with committed scores (read-only)")
    c.add_argument("names", nargs="*", help="registered method names")
    c.add_argument("--all", action="store_true", help="every registered method with a committed scores file")
    c.set_defaults(func=_cmd_check)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"ragexp {args.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
