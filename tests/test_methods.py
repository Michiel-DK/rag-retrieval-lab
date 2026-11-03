"""Method interface on a toy corpus. Exact reproduction of committed runs is in test_integration."""
import numpy as np
import pytest

import ragexp.embed
from ragexp.data import Corpus, Doc, Query
from ragexp.methods import Bm25, Dense, Method, Rrf, describe, get_method
from ragexp.retrieve import bm25, build_bm25, rrf

DOCS = [
    Doc("d1", "statins", "cholesterol statin drugs and cancer risk"),
    Doc("d2", "weather", "the weather in brussels is grey"),
    Doc("d3", "diet", "vegetables fibre and cholesterol"),
]
QUERIES = [Query("q1", "statin cholesterol"), Query("q2", "grey weather")]
CORPUS = Corpus(DOCS, QUERIES, {"q1": {"d1": 2}, "q2": {"d2": 1}})


class FakeEmbedder:
    """Bag-of-letters vectors: deterministic, normalized, no model download."""

    def __init__(self, model_name=None):
        self.model_name = model_name

    def encode(self, texts):
        m = np.zeros((len(texts), 26))
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    m[i, ord(ch) - 97] += 1
        return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture
def fake_emb(monkeypatch):
    monkeypatch.setattr(ragexp.embed, "Embedder", FakeEmbedder)


def test_registered_methods_satisfy_protocol(fake_emb):
    for name in ("bm25", "dense", "rrf"):
        m = get_method(name)
        assert isinstance(m, Method) and m.name == name
        r = m.rank(CORPUS)
        assert set(r) == {"q1", "q2"}
        assert all(sorted(v) == ["d1", "d2", "d3"] for v in r.values())


def test_bm25_method_is_the_free_function():
    index, ids = build_bm25(DOCS)
    assert Bm25().rank(CORPUS) == {q.query_id: bm25(q.text, index, ids) for q in QUERIES}


def test_rrf_method_fuses_dense_then_bm25(fake_emb):
    d, b = Dense().rank(CORPUS), Bm25().rank(CORPUS)
    got = Rrf(depth=2).rank(CORPUS)
    assert got == {q: rrf([d[q], b[q]], k=60, depth=2) for q in d}


def test_defaults_match_committed_runs():
    # rrf.json is nb 03's depth=1000 config; dense.json is MiniLM-L6.
    assert describe(get_method("rrf"))["params"] == {
        "k": 60, "depth": 1000, "model": "sentence-transformers/all-MiniLM-L6-v2", "parts": ["dense", "bm25"],
    }
    assert get_method("dense").params() == {"model": "sentence-transformers/all-MiniLM-L6-v2"}


def test_unknown_method_lists_registry():
    with pytest.raises(KeyError, match="registered: bm25, dense, rrf"):
        get_method("colbert")
