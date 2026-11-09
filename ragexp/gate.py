"""Per-query rerank gate: skip the reranker where a pre-rerank feature says dense is enough.

Lane ``rerank-gate-oracle`` (``predictions/rerank-gate-oracle.md``). The gated system
returns the dense pool order where the gate skips and the judge's permutation of the
same pool where it calls, so recall@50 equals dense's on every query by construction.

A gate is one feature, one direction and one threshold. They are fitted inside k-fold
cross-validation over queries: each fold's queries get the threshold fitted on the
other folds, so every per-query score the method emits is held out.

Fit rule, fixed before any run: among (direction, threshold) pairs whose training skip
rate is at least ``min_skip``, take the one with the highest training mean of the gated
nDCG@10 (equivalently, the smallest rerank gain given up on skipped queries). Ties go
to the higher skip rate, then direction "high" before "low", then the lower threshold.

Features come from the dense cosine scores of the 50-doc pool. ``results/rerank/
dense_pool.json`` holds the doc ids only, so the scores are recomputed from the
embedding cache (the same vectors ``Dense`` ranks with; a cache hit is not a model call).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURES = ("top1", "gap", "above_median")
JUDGES = {"llm": "llm_grades", "zerank": "zerank_scores"}
DIRECTIONS = ("high", "low")  # "high": skip when feature > threshold; "low": skip when feature < threshold


def pool_features(pool_scores: Sequence[float]) -> Dict[str, float]:
    """Pre-rerank features of one query's dense pool scores (rank order, descending).

    ``above_median`` is the count of pool scores strictly above the pool median. With
    distinct scores it is floor(n/2) for every query, so as specified it cannot gate.
    It stays in so that negative is on the record.
    """
    s = np.asarray(pool_scores, dtype=float)
    if len(s) < 2:
        raise ValueError("a pool needs at least two scores for the gap feature")
    return {
        "top1": float(s[0]),
        "gap": float(s[0] - s[1]),
        "above_median": float(np.sum(s > np.median(s))),
    }


def assign_folds(query_ids: Sequence[str], folds: int = 5, seed: int = 42) -> Dict[str, int]:
    """query_id -> fold, from a seeded permutation of the given (canonical) order."""
    perm = np.random.default_rng(seed).permutation(len(query_ids))
    return {query_ids[int(pos)]: rank % folds for rank, pos in enumerate(perm)}


def skips(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "high":
        return values > threshold
    if direction == "low":
        return values < threshold
    raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


@dataclass(frozen=True)
class Fit:
    direction: str
    threshold: float
    train_skip_rate: float
    train_gain_kept: float  # mean over training queries of the rerank gain on called queries


def fit_threshold(values: Sequence[float], gains: Sequence[float], min_skip: float = 0.3) -> Fit:
    """Fit one gate on training data. ``gains`` = rerank nDCG@10 - dense nDCG@10 per query.

    Candidate thresholds are -inf, +inf and the midpoints between consecutive distinct
    values, so every achievable split is tried once.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(gains, dtype=float)
    if len(v) != len(g) or len(v) == 0:
        raise ValueError("values and gains must be non-empty and the same length")
    u = np.unique(v)
    cands = [-math.inf, math.inf] + [float((a + b) / 2) for a, b in zip(u[:-1], u[1:])]
    best = None
    best_key = None
    for d_i, direction in enumerate(DIRECTIONS):
        for t in cands:
            sk = skips(v, direction, t)
            rate = float(sk.mean())
            if rate < min_skip:
                continue
            kept = float(g[~sk].sum() / len(g))
            key = (kept, rate, -d_i, -t)
            if best_key is None or key > best_key:
                best_key, best = key, Fit(direction, t, rate, kept)
    if best is None:  # unreachable: skip-all always meets min_skip <= 1
        raise ValueError(f"no threshold reaches skip rate {min_skip}")
    return best


def cross_validate(features: Dict[str, float], gains: Dict[str, float], query_ids: Sequence[str],
                   folds: int = 5, seed: int = 42, min_skip: float = 0.3) -> Tuple[Dict[str, bool], List[Fit]]:
    """Held-out gate decisions: query_id -> True when the reranker is called."""
    fold_of = assign_folds(query_ids, folds, seed)
    call: Dict[str, bool] = {}
    fits: List[Fit] = []
    for f in range(folds):
        train = [q for q in query_ids if fold_of[q] != f]
        test = [q for q in query_ids if fold_of[q] == f]
        fit = fit_threshold([features[q] for q in train], [gains[q] for q in train], min_skip)
        fits.append(fit)
        sk = skips(np.array([features[q] for q in test], dtype=float), fit.direction, fit.threshold)
        call.update({q: not bool(s) for q, s in zip(test, sk)})
    return call, fits


def auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positive, dtype=bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs both classes")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


__all__ = ["FEATURES", "JUDGES", "DIRECTIONS", "Fit", "pool_features", "assign_folds", "skips", "fit_threshold",
           "cross_validate", "auc"]
