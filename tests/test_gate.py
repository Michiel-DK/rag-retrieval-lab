"""Per-query rerank gate (lane rerank-gate-oracle) on synthetic data: no dataset, no model."""
import math

import numpy as np
import pytest

import ragexp.embed
import ragexp.rerank
from ragexp.data import Corpus, Doc, Query
from ragexp.gate import assign_folds, auc, cross_validate, fit_threshold, pool_features, skips
from ragexp.methods import GatedRerank, get_method
from ragexp.runs import evaluate_run


def test_pool_features_and_degenerate_above_median():
    f = pool_features([0.9, 0.7, 0.5, 0.4, 0.1, 0.0])
    assert f["top1"] == 0.9
    assert f["gap"] == pytest.approx(0.2)
    assert f["above_median"] == 3.0
    # distinct scores: always floor(n/2), whatever the scores are
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = np.sort(rng.random(50))[::-1]
        assert pool_features(s)["above_median"] == 25.0


def test_folds_are_seeded_balanced_and_cover_every_query():
    q = [f"q{i}" for i in range(23)]
    a, b = assign_folds(q, 5, 42), assign_folds(q, 5, 42)
    assert a == b and set(a) == set(q)
    sizes = sorted(list(a.values()).count(f) for f in range(5))
    assert sizes == [4, 4, 5, 5, 5]
    assert assign_folds(q, 5, 7) != a


def test_fit_picks_the_split_that_gives_up_least_gain():
    # high feature = rerank hurts there; skipping the top 40% loses nothing
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gains = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -0.1, -0.1, -0.1, -0.1]
    fit = fit_threshold(values, gains, min_skip=0.3)
    assert (fit.direction, fit.threshold, fit.train_skip_rate) == ("high", 6.5, 0.4)
    assert fit.train_gain_kept == pytest.approx(0.18)
    assert skips(np.array(values, float), fit.direction, fit.threshold).sum() == 4


def test_fit_respects_min_skip_even_when_skipping_costs():
    values = list(range(10))
    gains = [0.1] * 10  # reranking always helps: the cheapest legal skip is exactly 30%
    fit = fit_threshold(values, gains, min_skip=0.3)
    assert fit.train_skip_rate == pytest.approx(0.3)


def test_constant_feature_can_only_skip_everything():
    fit = fit_threshold([25.0] * 8, [0.2, -0.1] * 4, min_skip=0.3)
    assert fit.train_skip_rate == 1.0 and fit.threshold == -math.inf


def test_cross_validate_never_fits_on_the_held_out_fold():
    q = [f"q{i}" for i in range(20)]
    feats = {x: float(i) for i, x in enumerate(q)}
    gains = {x: (0.2 if i < 14 else -0.2) for i, x in enumerate(q)}
    call, fits = cross_validate(feats, gains, q, folds=5, seed=42, min_skip=0.3)
    assert set(call) == set(q) and len(fits) == 5
    fold = assign_folds(q, 5, 42)
    for f, fit in enumerate(fits):
        train = [x for x in q if fold[x] != f]
        assert fit == fit_threshold([feats[x] for x in train], [gains[x] for x in train], 0.3)
    assert all(call[x] for x in q[:12]) and not any(call[x] for x in q[15:])


def test_auc_with_ties():
    assert auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert auc([0, 1], [True, False]) == 0.0
    with pytest.raises(ValueError):
        auc([1, 2], [True, True])


class FakeEmbedder:
    def __init__(self, model_name=None):
        pass

    def encode(self, texts):
        m = np.zeros((len(texts), 26))
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    m[i, ord(ch) - 97] += 1
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def _toy(monkeypatch, n_q=10, n_d=6):
    rng = np.random.default_rng(1)
    words = ["statin", "fibre", "vitamin", "cancer", "heart", "sugar", "salt", "milk"]
    docs = [Doc(f"d{i}", words[i % 8], " ".join(rng.choice(words, 4))) for i in range(n_d)]
    queries = [Query(f"q{j}", " ".join(rng.choice(words, 2))) for j in range(n_q)]
    qrels = {q.query_id: {f"d{int(i)}": int(rng.integers(1, 3)) for i in rng.choice(n_d, 2, replace=False)}
             for q in queries}
    pool = {q.query_id: [f"d{int(i)}" for i in rng.permutation(n_d)] for q in queries}
    grades = {q.query_id: {d: float(rng.integers(0, 4)) for d in pool[q.query_id]} for q in queries}
    monkeypatch.setattr(ragexp.embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ragexp.rerank, "load_pool", lambda name="dense_pool": pool)
    monkeypatch.setattr(ragexp.rerank, "load_sweep", lambda name: grades)
    return Corpus(docs, queries, qrels), pool


def test_gated_rerank_is_a_per_query_choice_between_two_permutations_of_the_pool(monkeypatch):
    corpus, pool = _toy(monkeypatch)
    m = get_method("gated-rerank", judge="llm", feature="gap", folds=5, seed=42, min_skip=0.3)
    g = m.gate(corpus)
    ranked = m.rank(corpus)
    assert set(ranked) == set(pool)
    for q, r in ranked.items():
        assert sorted(r) == sorted(pool[q])  # a permutation of the pool: recall@pool-size frozen
        assert r == (g["judge_order"][q] if g["call"][q] else pool[q])
    dense = evaluate_run(corpus, pool, recall_ks=(6,), ndcg_ks=())
    gated = evaluate_run(corpus, ranked, recall_ks=(6,), ndcg_ks=())
    assert {q: v["recall@6"] for q, v in dense.items()} == {q: v["recall@6"] for q, v in gated.items()}
    assert m.params()["feature"] == "gap" and m.name == "gated-rerank"


def test_gated_rerank_rejects_unknown_judge_or_feature():
    with pytest.raises(ValueError):
        GatedRerank(judge="gpt")
    with pytest.raises(ValueError):
        GatedRerank(feature="query_length")
