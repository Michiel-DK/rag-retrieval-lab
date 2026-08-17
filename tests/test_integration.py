"""Integration smoke tests — need the HF dataset cache and results/emb/.

These are the BUILD.md "smoke test embed.py / retrieve.py" gate: load the
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
