"""Unit tests for ragexp.runs — synthetic corpus, no dataset, no model."""
import pytest

from ragexp.data import Corpus, Doc, Query
from ragexp.runs import evaluate_run, load_scores, metric_vector, save_scores, summarize


@pytest.fixture
def tiny_corpus():
    docs = [Doc(f"d{i}", f"title {i}", f"text {i}") for i in range(4)]
    queries = [Query("q1", "one"), Query("q2", "two")]
    qrels = {"q1": {"d0": 2, "d1": 1}, "q2": {"d3": 1}}
    return Corpus(docs=docs, queries=queries, qrels=qrels)


def test_evaluate_run_scores_per_query(tiny_corpus):
    rankings = {"q1": ["d0", "d1", "d2", "d3"], "q2": ["d0", "d1", "d2", "d3"]}
    scores = evaluate_run(tiny_corpus, rankings, recall_ks=(2,), ndcg_ks=(2,))
    assert scores["q1"]["recall@2"] == 1.0
    assert scores["q1"]["ndcg@2"] == pytest.approx(1.0)
    assert scores["q2"]["recall@2"] == 0.0  # d3 ranked last


def test_evaluate_run_skips_missing_queries(tiny_corpus):
    scores = evaluate_run(tiny_corpus, {"q1": ["d0"]}, recall_ks=(1,), ndcg_ks=(1,))
    assert set(scores) == {"q1"}


def test_summarize_means_and_drops_none(tiny_corpus):
    rankings = {"q1": ["d0", "d1"], "q2": ["d3", "d0"]}
    scores = evaluate_run(tiny_corpus, rankings, recall_ks=(1,), ndcg_ks=(1,))
    summary = summarize(scores)
    assert summary["recall@1"] == pytest.approx((0.5 + 1.0) / 2)


def test_metric_vector_aligns_on_query_order(tiny_corpus):
    rankings = {"q1": ["d0"], "q2": ["d3"]}
    scores = evaluate_run(tiny_corpus, rankings, recall_ks=(1,), ndcg_ks=(1,))
    vec = metric_vector(scores, "recall@1", ["q2", "q1", "missing"])
    assert vec == [1.0, 0.5, None]


def test_save_load_roundtrip(tiny_corpus, tmp_path, monkeypatch):
    import ragexp.runs as runs

    monkeypatch.setattr(runs, "_SCORES", tmp_path)
    scores = {"q1": {"recall@10": 0.5, "ndcg@10": None}}
    save_scores("test_run", scores)
    assert load_scores("test_run") == scores
