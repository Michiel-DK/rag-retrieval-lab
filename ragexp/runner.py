"""``ragexp run``: rank with a registered method, score, write a NEW results/scores file.

Committed score files are evidence and are never regenerated (CLAUDE.md). So
``run`` refuses before any compute when ``results/scores/<name>.json`` already
exists, tracked or not. To try a variant, give it its own ``--name``. To
recompute a committed run and compare, use ``ragexp check``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import runs
from .data import Corpus, load_nfcorpus
from .methods import Method, get_method
from .runs import evaluate_run, save_scores, summarize

RECALL_KS = (10, 50)
NDCG_KS = (10,)


def parse_params(items: Sequence[str]) -> Dict[str, object]:
    """``["depth=100", "model=foo/bar", "depth=null"]`` -> typed dict. Values are JSON when they parse, else strings."""
    out: Dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--param expects key=value, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def target_path(name: str) -> Path:
    return runs._SCORES / f"{name}.json"


@dataclass
class RunResult:
    name: str
    method: Method
    path: Path
    n_queries: int
    means: Dict[str, Optional[float]]


def build(method_name: str, params: Dict[str, object], name: Optional[str]) -> Method:
    """Resolve the method and the output name, and refuse an existing file. No compute here."""
    try:
        method = get_method(method_name, **params)
    except KeyError as e:
        raise ValueError(e.args[0]) from None
    except TypeError as e:  # unknown parameter for this method
        raise ValueError(f"bad --param for {method_name!r}: {e}") from None
    if name:
        method.name = name
    path = target_path(method.name)
    if path.exists():
        raise FileExistsError(
            f"{path} exists; committed results are never regenerated. "
            f"Use --name for a new run, or `ragexp check {method.name}` to verify this one."
        )
    return method


def run(method: Method, corpus: Optional[Corpus] = None) -> RunResult:
    if target_path(method.name).exists():  # re-check: build() may have run long before
        raise FileExistsError(f"{target_path(method.name)} appeared during the run; not overwriting")
    if corpus is None:
        corpus = load_nfcorpus()
    scores = evaluate_run(corpus, method.rank(corpus), recall_ks=RECALL_KS, ndcg_ks=NDCG_KS)
    path = save_scores(method.name, scores)
    return RunResult(method.name, method, path, len(scores), summarize(scores))


def format_run(r: RunResult) -> str:
    lines: List[str] = [f"wrote {r.path}  ({r.n_queries} queries)"]
    for m, v in r.means.items():
        lines.append(f"  {m:>10}: {'n/a' if v is None else f'{v:.4f}'}")
    return "\n".join(lines)
