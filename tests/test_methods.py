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
    with pytest.raises(KeyError, match="registered: bm25, dense, dense-hyde, dense-prf, dense-q2d, gated-rerank, rrf"):
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


# ── dense-prf (lane query-prf): vector PRF on the dense query ──

from ragexp.methods import DensePrf


def test_dense_prf_k0_reproduces_dense_exactly(fake_emb):
    assert DensePrf(k=0).rank(CORPUS) == Dense().rank(CORPUS)


def test_dense_prf_k1_ranks_by_normalised_query_plus_top_doc(fake_emb):
    e = FakeEmbedder()
    docs = e.encode([d.full for d in DOCS])
    expected = {}
    for q in QUERIES:
        qv = e.encode([q.text])[0]
        top = np.argsort(-(docs @ qv))[0]
        q2 = (qv + docs[top]) / 2
        q2 /= np.linalg.norm(q2)
        expected[q.query_id] = [DOCS[i].doc_id for i in np.argsort(-(docs @ q2))]
    assert DensePrf(k=1).rank(CORPUS) == expected


def test_dense_prf_changes_the_ranking_when_feedback_pulls(fake_emb):
    # q "s": first pass a (s,x) > b (s,w,w) > c (x,x,x,x,s). a's x pulls q' toward c, past b.
    docs = [Doc("a", "sx", ""), Doc("b", "sww", ""), Doc("c", "xxxxs", "")]
    corpus = Corpus(docs, [Query("q", "s")], {"q": {"c": 1}})
    assert Dense().rank(corpus)["q"] == ["a", "b", "c"]
    assert DensePrf(k=1).rank(corpus)["q"] == ["a", "c", "b"]


def test_dense_prf_defaults_and_validation():
    m = get_method("dense-prf")
    assert m.params() == {"k": 3, "weights": "equal", "model": "sentence-transformers/all-MiniLM-L6-v2"}
    assert init_params(m) == {"k": 3, "weights": "equal", "model": "sentence-transformers/all-MiniLM-L6-v2"}
    for bad in ({"k": -1}, {"k": True}, {"k": 1.5}, {"weights": "rocchio"}):
        with pytest.raises(ValueError):
            DensePrf(**bad)


# ── dense-hyde / dense-q2d (lane query-rewrite): a committed model passage per query ──

import json

import ragexp.rewrite
from ragexp.methods import DenseHyde, DenseQ2d
from ragexp.rewrite import sha256_file


@pytest.fixture
def gens_file(tmp_path, monkeypatch):
    def write(passages):
        path = tmp_path / "generations.json"
        path.write_text(json.dumps({"generations": {q: (None if p is None else {"passage": p}) for q, p in passages.items()}}))
        monkeypatch.setattr(ragexp.rewrite, "GENERATIONS_FILE", path)
        return sha256_file(path)
    return write


def test_hyde_with_the_query_as_passage_reproduces_dense(fake_emb, gens_file):
    sha = gens_file({q.query_id: q.text for q in QUERIES})
    assert DenseHyde(sha256=sha).rank(CORPUS) == Dense().rank(CORPUS)


def test_failed_generation_falls_back_to_dense_in_both_arms(fake_emb, gens_file):
    sha = gens_file({q.query_id: None for q in QUERIES})
    assert DenseHyde(sha256=sha).rank(CORPUS) == Dense().rank(CORPUS)
    assert DenseQ2d(sha256=sha).rank(CORPUS) == Dense().rank(CORPUS)


def test_hyde_averages_query_and_passage_vectors(fake_emb, gens_file):
    sha = gens_file({"q1": "grey weather brussels", "q2": "vegetables"})
    e = FakeEmbedder()
    docs = e.encode([d.full for d in DOCS])
    expected = {}
    for q, p in (("q1", "grey weather brussels"), ("q2", "vegetables")):
        v = e.encode([dict((x.query_id, x.text) for x in QUERIES)[q]])[0] + e.encode([p])[0]
        v /= np.linalg.norm(v)
        expected[q] = [DOCS[i].doc_id for i in np.argsort(-(docs @ v))]
    assert DenseHyde(sha256=sha).rank(CORPUS) == expected


def test_q2d_encodes_query_sep_passage(fake_emb, gens_file, monkeypatch):
    sha = gens_file({"q1": "pills", "q2": None})
    seen = []

    class Spy(FakeEmbedder):
        def encode(self, texts):
            seen.extend(texts)
            return super().encode(texts)

    monkeypatch.setattr(ragexp.embed, "Embedder", Spy)
    DenseQ2d(sha256=sha).rank(CORPUS)
    assert "statin cholesterol [SEP] pills" in seen and "grey weather" in seen


def test_rewrite_methods_refuse_a_changed_or_unpinned_file(fake_emb, gens_file):
    sha = gens_file({q.query_id: "x" for q in QUERIES})
    for cls in (DenseHyde, DenseQ2d):
        with pytest.raises(ValueError, match="changed"):
            cls(sha256="0" * 64).rank(CORPUS)
        with pytest.raises(ValueError, match="sha256"):
            cls().rank(CORPUS)
    gens_file({"q1": "x"})  # q2 missing
    with pytest.raises(ValueError, match="no generation entry"):
        DenseHyde(sha256=sha256_file(ragexp.rewrite.GENERATIONS_FILE)).rank(CORPUS)


def test_rewrite_methods_defaults():
    for key, cls in (("dense-hyde", DenseHyde), ("dense-q2d", DenseQ2d)):
        m = get_method(key, sha256="ab")
        assert isinstance(m, cls) and m.name == key
        assert init_params(m) == {"sha256": "ab", "model": "sentence-transformers/all-MiniLM-L6-v2"}
