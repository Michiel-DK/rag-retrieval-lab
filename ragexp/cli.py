"""``ragexp`` command line. Subcommands are added one queue item at a time."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _cmd_diff(args: argparse.Namespace) -> int:
    from .compare import diff, format_diff

    d = diff(args.a, args.b, metric=args.metric, metric_b=args.metric_b, n_resamples=args.resamples, seed=args.seed)
    print(format_diff(d))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from . import check, manifest, methods, runs

    if bool(args.names) == bool(args.all):
        raise ValueError("give method names or --all, not both and not neither")
    mdir = manifest.manifest_dir()
    manifested = {p.stem for p in mdir.glob("*.json")} if mdir.is_dir() else set()
    if args.all:
        # --all = every registered method + every `ragexp run` output with a manifest. A missing
        # committed file is an error, never a skip, so a green --all cannot be vacuous (#40 review).
        # RUN_ONLY methods have no default run under their registry key; their runs are
        # covered through the manifest half, so the registry half skips only the key.
        names = sorted((set(methods.REGISTRY) - methods.RUN_ONLY) | manifested)
        if not names:
            raise ValueError("--all: no registered methods")
    else:
        names = list(args.names)
    for n in names:  # fail on usage errors before the (slow) corpus load
        if n not in methods.REGISTRY and n not in manifested:
            raise ValueError(f"unknown method {n!r} and no manifest; registered: {', '.join(sorted(methods.REGISTRY))}")
        if not (runs._SCORES / f"{n}.json").is_file():
            raise FileNotFoundError(f"no committed scores for {n!r} at {runs._SCORES / (n + '.json')}")
    corpus = check.load_nfcorpus()
    results = [check.check_method(n, corpus) for n in names]
    for r in results:
        print(check.format_result(r))
    return 0 if all(r.ok for r in results) else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from . import runner

    ks = runner.normalize_recall_ks(args.recall_k)  # ValueError (exit 2) before any compute
    method, prediction = runner.build(args.method, runner.parse_params(args.param), args.name)  # refuses before compute
    print(runner.format_run(runner.run(method, prediction, args.method, recall_ks=ks)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragexp", description="Retrieval experiments on NFCorpus, predictions first.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="paired bootstrap between two committed score files (b - a)")
    d.add_argument("a", help="baseline run: name under results/scores/ or a path to a .json")
    d.add_argument("b", help="candidate run")
    d.add_argument("--metric", default="ndcg@10")
    d.add_argument("--metric-b", default=None, help="metric read from run b (default: --metric); pairs metric of a with metric-b of b")
    d.add_argument("--resamples", type=int, default=10_000)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=_cmd_diff)

    c = sub.add_parser("check", help="recompute registered methods and compare with committed scores (read-only)")
    c.add_argument("names", nargs="*", help="registered method names")
    c.add_argument("--all", action="store_true", help="every registered method; exit 2 if any lacks a committed scores file")
    c.set_defaults(func=_cmd_check)

    r = sub.add_parser("run", help="rank + score a registered method into a NEW results/scores/<name>.json")
    r.add_argument("method", help="registered method name")
    r.add_argument("--name", help="output name (default: the method's name); must not exist yet")
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="method parameter, JSON-typed (repeatable)")
    r.add_argument("--recall-k", action="append", type=int, default=None, metavar="K", help="recall depth to score (repeatable; default 10 and 50)")
    r.set_defaults(func=_cmd_run)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"ragexp {args.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
