"""Unit tests for ragexp.metrics — pure math, no dataset, no model."""
import math

import pytest

from ragexp.metrics import (
    independent_ci,
    mean_ignoring_none,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)


# ── recall@k ──────────────────────────────────────────────────────────────


def test_recall_basic():
    assert recall_at_k(["a", "b", "c", "d"], ["a", "c"], k=2) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], ["a", "c"], k=4) == 1.0


def test_recall_undefined_is_none_not_zero():
    assert recall_at_k(["a", "b"], [], k=2) is None


def test_recall_is_a_set_metric():
    # Reordering within the top-k cannot change recall — the property
    # notebook 04 leans on.
    rel = ["a", "b"]
    assert recall_at_k(["a", "b", "x"], rel, k=2) == recall_at_k(
        ["b", "a", "x"], rel, k=2
    )


# ── nDCG@k ────────────────────────────────────────────────────────────────


def test_ndcg_perfect_ranking_is_one():
    grades = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], grades, k=3) == pytest.approx(1.0)


def test_ndcg_order_sensitive():
    grades = {"a": 2, "b": 1}
    best = ndcg_at_k(["a", "b"], grades, k=2)
    worse = ndcg_at_k(["b", "a"], grades, k=2)
    assert worse < best


def test_ndcg_undefined_is_none():
    assert ndcg_at_k(["a", "b"], {}, k=2) is None
    assert ndcg_at_k(["a"], {"a": 0}, k=1) is None


def test_ndcg_known_value():
    # ranked: grade 1 at rank 1, grade 2 at rank 2; ideal: 2 then 1.
    grades = {"a": 1, "b": 2}
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(["a", "b"], grades, k=2) == pytest.approx(dcg / idcg)


# ── aggregation ───────────────────────────────────────────────────────────


def test_mean_ignoring_none():
    assert mean_ignoring_none([1.0, None, 0.0]) == 0.5
    assert mean_ignoring_none([None, None]) is None
    assert mean_ignoring_none([]) is None


# ── bootstrap ─────────────────────────────────────────────────────────────


def test_paired_bootstrap_detects_consistent_difference():
    a = [0.5] * 50
    b = [0.6] * 50
    delta, (lo, hi), p = paired_bootstrap(a, b, n_resamples=2000)
    assert delta == pytest.approx(0.1)
    assert lo <= delta <= hi
    assert p < 0.05


def test_paired_bootstrap_null_difference_not_significant():
    rng_scores = [0.4, 0.6] * 25
    a = rng_scores
    b = list(reversed(rng_scores))
    _, (lo, hi), p = paired_bootstrap(a, b, n_resamples=2000)
    assert lo <= 0.0 <= hi
    assert p > 0.05


def test_paired_bootstrap_drops_none_pairs():
    a = [0.5, None, 0.5]
    b = [0.6, 0.9, None]
    delta, _, _ = paired_bootstrap(a, b, n_resamples=500)
    assert delta == pytest.approx(0.1)


def test_paired_bootstrap_all_none_raises():
    with pytest.raises(ValueError):
        paired_bootstrap([None], [None])


def test_independent_ci_contains_mean():
    mean, (lo, hi) = independent_ci([0.2, 0.4, 0.6, 0.8], n_resamples=2000)
    assert mean == pytest.approx(0.5)
    assert lo <= mean <= hi


def test_bootstrap_deterministic_under_seed():
    a, b = [0.3, 0.5, 0.7], [0.4, 0.5, 0.9]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)
