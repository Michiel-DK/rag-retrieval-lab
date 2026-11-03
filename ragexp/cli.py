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
