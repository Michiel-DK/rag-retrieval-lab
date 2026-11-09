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
    with pytest.raises(KeyError, match="registered: bm25, dense, gated-rerank, rrf"):
        get_method("colbert")


# ── uibk-first-stage: interface + MaxSim on synthetic vectors (no pylate, no download) ──

from ragexp.manifest import init_params
from ragexp.methods import Uibk, maxsim_rank


def test_maxsim_rank_sums_per_query_vector_max_and_breaks_ties_stably():
    # doc a: rows 0-1, doc b: row 2, doc c: rows 3-4 (identical to a -> tie, a first by corpus order)
    d = np.array([[1, 0], [0, 1], [0.5, 0.5], [1, 0], [0, 1]], dtype=np.float32)
    offsets = np.array([0, 2, 3, 5])
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)  # a and c: 1+1=2; b: 0.5+0.5=1
    assert maxsim_rank(q, d, offsets, ["a", "b", "c"]) == ["a", "c", "b"]
    q2 = np.array([[0.5, 0.5]], dtype=np.float32)  # a, b, c all 0.5 -> corpus order
    assert maxsim_rank(q2, d, offsets, ["a", "b", "c"]) == ["a", "b", "c"]


class FakeColBERT:
    """Letter-presence token vectors, one per word; the doc/query prompts are recorded."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, is_query, prompt_name, batch_size, **kw):
        self.calls.append((is_query, prompt_name))
        out = []
        for t in texts:
            rows = []
            for w in t.lower().split():
                v = np.zeros(26, dtype=np.float32)
                for ch in w:
                    if "a" <= ch <= "z":
                        v[ord(ch) - 97] = 1
                rows.append(v / max(np.linalg.norm(v), 1e-9))
            out.append(np.stack(rows))
        return out


def test_uibk_method_interface(monkeypatch):
    fake = FakeColBERT()
    monkeypatch.setattr(Uibk, "_load", lambda self: fake)
    m = get_method("uibk-first-stage")
    assert isinstance(m, Method) and m.name == "uibk-first-stage"
    r = m.rank(CORPUS)
    assert set(r) == {"q1", "q2"} and all(sorted(v) == ["d1", "d2", "d3"] for v in r.values())
    assert r["q1"][0] == "d1" and r["q2"][0] == "d2"
    assert fake.calls == [(False, "document"), (True, "query")]
    # the manifest records the HF revision, and check rebuilds the same method from it
    p = init_params(m)
    assert p["revision"] == "b18df82e77306006a1dd0162199f6dfeacb5492e" and p["model"] == "DataScience-UIBK/uibk-embedding-v1"
    assert get_method("uibk-first-stage", **p).params() == m.params()
