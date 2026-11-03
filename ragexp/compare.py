"""Compare two committed score files with the paired bootstrap, no corpus needed.

The bootstrap draws resample *positions*, so query order moves the CI in the
third decimal (see ``metrics.paired_bootstrap``). The notebooks use corpus order.
Score files are written with sorted keys, so that order is lost from them;
``results/query_order.json`` keeps it. Runs whose queries are not all in that
file (nb 06's MSLR runs) fall back to sorted ids, and the result says so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .metrics import paired_bootstrap
from .paths import RESULTS_DIR
from .runs import Scores, metric_vector, summarize

QUERY_ORDER = RESULTS_DIR / "query_order.json"


@dataclass
class Diff:
    a: str
    b: str
    metric: str
    n: int  # queries where both are defined
    mean_a: Optional[float]
    mean_b: Optional[float]
    delta: float  # b - a
    ci: Tuple[float, float]
    p: float
    order: str  # "corpus" or "sorted"


def load_run(name_or_path: str) -> Tuple[str, Scores]:
    """A run by name (``results/scores/<name>.json``) or by path to a json file."""
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = RESULTS_DIR / "scores" / f"{name_or_path}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no scores file at {p}")
    return p.stem, json.loads(p.read_text())


def _order(qids: set, order_file: Path) -> Tuple[List[str], str]:
    if order_file.is_file():
        canonical = json.loads(order_file.read_text())
        if qids <= set(canonical):
            return [q for q in canonical if q in qids], "corpus"
    return sorted(qids), "sorted"


def diff(
    a: str,
    b: str,
    metric: str = "ndcg@10",
    n_resamples: int = 10_000,
    seed: int = 42,
    order_file: Path = QUERY_ORDER,
) -> Diff:
    name_a, sa = load_run(a)
    name_b, sb = load_run(b)
    shared = set(sa) & set(sb)
    if not shared:
        raise ValueError(f"{name_a} and {name_b} share no queries")
    qids, order = _order(shared, order_file)
    va, vb = metric_vector(sa, metric, qids), metric_vector(sb, metric, qids)
    if all(v is None for v in va) or all(v is None for v in vb):
        raise ValueError(f"metric {metric!r} missing from {name_a} or {name_b}")
    delta, ci, p = paired_bootstrap(va, vb, n_resamples=n_resamples, seed=seed)
    n = sum(1 for x, y in zip(va, vb) if x is not None and y is not None)
    mean_a = summarize({q: sa[q] for q in qids}).get(metric)
    mean_b = summarize({q: sb[q] for q in qids}).get(metric)
    return Diff(name_a, name_b, metric, n, mean_a, mean_b, delta, ci, p, order)


def format_diff(d: Diff) -> str:
    lines = [
        f"{d.metric}  {d.a} -> {d.b}   n={d.n} paired queries",
        f"  {d.a:<24} {d.mean_a:.4f}",
        f"  {d.b:<24} {d.mean_b:.4f}",
        f"  delta (b - a)            {d.delta:+.4f}   95% CI [{d.ci[0]:+.4f}, {d.ci[1]:+.4f}]   p={d.p:.4f}",
        "  verdict                  " + ("CI excludes 0" if d.ci[0] > 0 or d.ci[1] < 0 else "CI straddles 0: no difference shown"),
    ]
    if d.order != "corpus":
        lines.append("  note: queries not all in results/query_order.json; sorted-id order used, CI may differ from the notebooks in the third decimal")
    return "\n".join(lines)
