"""``ragexp check``: recompute a registered method and compare with its committed scores.

Never writes committed scores. A cold embedding cache does get filled
(``results/emb/``, gitignored), as with any dense run. The recompute uses the same
``evaluate_run`` call the notebooks used; recall@k must match exactly, ndcg@k
within a float tolerance (summation-order noise only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import runs
from .data import Corpus, load_nfcorpus
from .methods import get_method
from .runs import Scores, evaluate_run

NDCG_TOL = 1e-12
MISSING_QUERY = "<missing query>"


@dataclass(frozen=True)
class Mismatch:
    query_id: str
    metric: str
    expected: Optional[float]
    got: Optional[float]


@dataclass
class CheckResult:
    name: str
    n_queries: int
    mismatches: List[Mismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def compare_scores(expected: Scores, got: Scores, ndcg_tol: float = NDCG_TOL) -> List[Mismatch]:
    out: List[Mismatch] = []
    for qid in sorted(set(expected) | set(got)):
        if qid not in expected or qid not in got:
            out.append(Mismatch(qid, MISSING_QUERY, None, None))
            continue
        e_row, g_row = expected[qid], got[qid]
        for metric in sorted(set(e_row) | set(g_row)):
            if metric not in e_row or metric not in g_row:
                out.append(Mismatch(qid, metric, e_row.get(metric), g_row.get(metric)))
                continue
            e, g = e_row[metric], g_row[metric]
            if e is None or g is None:
                same = e is None and g is None
            elif metric.startswith("ndcg@"):
                same = abs(e - g) <= ndcg_tol
            else:
                same = e == g
            if not same:
                out.append(Mismatch(qid, metric, e, g))
    return out


def check_method(name: str, corpus: Optional[Corpus] = None) -> CheckResult:
    try:
        method = get_method(name)
    except KeyError as e:
        raise ValueError(e.args[0]) from None
    expected = runs.load_scores(name)  # FileNotFoundError -> usage error in cli.main
    if corpus is None:
        corpus = load_nfcorpus()
    got = evaluate_run(corpus, method.rank(corpus), recall_ks=(10, 50), ndcg_ks=(10,))
    return CheckResult(name, len(expected), compare_scores(expected, got))


def format_result(r: CheckResult, max_lines: int = 10) -> str:
    if r.ok:
        return f"OK  {r.name}  {r.n_queries} queries identical"
    lines = [f"FAIL {r.name}  {len(r.mismatches)} mismatches"]
    for m in r.mismatches[:max_lines]:
        lines.append(f"  {m.query_id}  {m.metric}  expected={m.expected!r}  got={m.got!r}")
    return "\n".join(lines)
