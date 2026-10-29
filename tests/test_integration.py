"""Integration smoke tests — need the HF dataset cache and results/emb/.

These are the smoke-test gate for embed.py / retrieve.py: load the
real corpus, hit the committed embedding cache, and check each retrieval
method finds relevant documents for a known query.

Run with: pytest -m integration
"""
import pytest

from ragexp.data import load_nfcorpus
from ragexp.embed import Embedder
from ragexp.metrics import recall_at_k
from ragexp.retrieve import bm25, build_bm25, dense, rrf

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def corpus():
    return load_nfcorpus()


@pytest.fixture(scope="module")
def doc_matrix(corpus):
    return Embedder().encode([d.full for d in corpus.docs])


def test_corpus_shape_matches_documented_numbers(corpus):
    assert len(corpus.docs) == 3633
    assert len(corpus.queries) == 323
    assert sum(len(v) for v in corpus.qrels.values()) == 12334


def test_doc_matrix_from_cache(corpus, doc_matrix):
    assert doc_matrix.shape == (3633, 384)


def test_dense_finds_relevant_docs(corpus, doc_matrix):
    q = corpus.queries[0]
    ranked = dense(q.text, corpus.doc_ids, doc_matrix, Embedder())
    r = recall_at_k(ranked, corpus.relevant_ids(q.query_id), k=50)
    assert r is not None and r > 0.2


def test_bm25_finds_relevant_docs(corpus):
    index, ids = build_bm25(corpus.docs)
    q = corpus.queries[0]
    ranked = bm25(q.text, index, ids)
    r = recall_at_k(ranked, corpus.relevant_ids(q.query_id), k=50)
    assert r is not None and r > 0.1


def test_rrf_fuses_dense_and_bm25(corpus, doc_matrix):
    q = corpus.queries[0]
    ranked_d = dense(q.text, corpus.doc_ids, doc_matrix, Embedder())
    index, ids = build_bm25(corpus.docs)
    ranked_b = bm25(q.text, index, ids)
    fused = rrf([ranked_d, ranked_b], depth=1000)
    r = recall_at_k(fused, corpus.relevant_ids(q.query_id), k=50)
    assert r is not None and r > 0.2


def test_committed_pool_reproduces_committed_rerank_scores(corpus):
    # Defect 2: committed files alone (pool order + grades) rebuild the
    # published per-query scores exactly, with no embedding step.
    import json
    import math
    from pathlib import Path

    from ragexp.rerank import load_pool, load_sweep, rerank_by_scores
    from ragexp.runs import evaluate_run

    scores_dir = Path(__file__).resolve().parent.parent / "results" / "scores"
    pool = load_pool("dense_pool")
    llm, zr = load_sweep("llm_grades"), load_sweep("zerank_scores")
    runs = {
        "dense": pool,  # the un-reranked pool is the dense baseline at k <= 50
        "rerank_llm": {q: rerank_by_scores(pool[q], llm[q]) for q in pool},
        "rerank_zerank": {q: rerank_by_scores(pool[q], zr[q]) for q in pool},
    }
    for name, ranking in runs.items():
        got = evaluate_run(corpus, ranking, recall_ks=(10, 50), ndcg_ks=(10,))
        want = json.loads((scores_dir / f"{name}.json").read_text())
        assert set(got) == set(want), name
        for qid, row in want.items():
            assert set(got[qid]) == set(row), (name, qid)
            for metric, w in row.items():
                g = got[qid][metric]
                if w is None or metric.startswith("recall@"):
                    # recall is a ratio of small integers: exact, None included
                    assert g == w, (name, qid, metric, g, w)
                else:
                    # ndcg sums log2 discounts, so another numpy can move the
                    # last bit (seen: 3.3e-16). A wrong order moves it ~1e-2.
                    assert g is not None and math.isclose(
                        g, w, rel_tol=1e-12, abs_tol=1e-12
                    ), (name, qid, metric, g, w)
