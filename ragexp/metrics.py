"""Retrieval metrics — the ruler.

Stdlib + numpy only. No dataset, no model, no API. Everything here is pure
math you can unit-test, which is the point: the ruler must be simpler than
the thing it measures.

Ported from a production system's retrieval-eval harness. What came across is
the math and one discipline that most tutorial code gets wrong:

    **Undefined is None, never 0.0.**

If a query has no relevant documents, its recall is *undefined* — not zero.
Averaging a 0.0 in there silently understates your score, and the bug is
invisible because the number still looks like a number. Every function here
returns ``None`` when the metric is undefined, and ``mean_ignoring_none``
drops them from the aggregate. See notebook 01.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ── Recall@k ──────────────────────────────────────────────────────────────


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> Optional[float]:
    """Fraction of relevant ids present in the top-k of ``ranked_ids``.

    Returns ``None`` when there are zero relevant ids to find — callers must
    exclude these from any aggregate rather than treating them as 0.0.

    Note what recall@k *cannot* see: it's a set metric. Reordering within the
    top-k does not change it. That property is load-bearing in notebook 04.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return None
    hit = len(set(ranked_ids[:k]) & relevant)
    return hit / len(relevant)


# ── nDCG@k ────────────────────────────────────────────────────────────────


def _dcg(grades_in_rank_order: Sequence[int]) -> float:
    """Standard (2^grade - 1) / log2(rank + 1) DCG, 1-indexed rank."""
    return sum(
        (2**grade - 1) / math.log2(i + 1)
        for i, grade in enumerate(grades_in_rank_order, start=1)
    )


def ndcg_at_k(
    ranked_ids: Sequence[str],
    grades: Dict[str, int],
    k: int,
) -> Optional[float]:
    """Normalized DCG@k. ``grades`` maps id -> relevance grade (0 if absent).

    Returns ``None`` when no id has a positive grade (ideal DCG is 0, so the
    normalization is undefined) — same "exclude, don't zero" rule as
    ``recall_at_k``.

    Unlike recall@k, nDCG *is* order-sensitive. That's why the two disagree,
    and why reporting both is not redundant.
    """
    actual = [grades.get(i, 0) for i in ranked_ids[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]

    idcg = _dcg(ideal)
    if idcg == 0:
        return None
    return _dcg(actual) / idcg


# ── Aggregation ───────────────────────────────────────────────────────────


def mean_ignoring_none(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values; None if empty/all-None."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ── Paired bootstrap ──────────────────────────────────────────────────────
#
# NOT ported — the source system compared systems with independent CIs and
# checked whether they overlapped. That's a strictly weaker (and needlessly
# harsher) test than pairing, because both systems face the *same* queries,
# so per-query difficulty is a shared nuisance term you can cancel. Notebook
# 01 shows the two side by side on the same data.


def paired_bootstrap(
    scores_a: Sequence[Optional[float]],
    scores_b: Sequence[Optional[float]],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float], float]:
    """Bootstrap the per-query mean difference (b - a).

    Resamples *queries*, not scores — that's what makes it paired. Queries
    where either system is undefined (None) are dropped, since a difference
    is undefined if either side is.

    Returns ``(mean_delta, (ci_low, ci_high), p_two_sided)`` where the CI is
    95% and ``p_two_sided`` is the fraction of resamples whose sign flips,
    doubled. A CI excluding 0 means the difference survived resampling.
    """
    pairs = [
        (a, b)
        for a, b in zip(scores_a, scores_b)
        if a is not None and b is not None
    ]
    if not pairs:
        raise ValueError("no queries where both systems are defined")

    deltas = np.array([b - a for a, b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
    boot_means = deltas[idx].mean(axis=1)

    observed = float(deltas.mean())
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    p = 2 * min(
        float((boot_means <= 0).mean()),
        float((boot_means >= 0).mean()),
    )
    return observed, ci, min(p, 1.0)


def independent_ci(
    scores: Sequence[Optional[float]],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap CI for a single system's mean, ignoring pairing.

    Here only so notebook 01 can show *why* you shouldn't use two of these
    and check for overlap. Non-overlapping independent CIs is a sufficient
    but not necessary condition for a real difference — you can have a
    genuine, paired-significant effect whose independent CIs overlap.
    """
    vals = np.array([v for v in scores if v is not None], dtype=float)
    if vals.size == 0:
        raise ValueError("no defined scores")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    boot_means = vals[idx].mean(axis=1)
    return float(vals.mean()), (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )
