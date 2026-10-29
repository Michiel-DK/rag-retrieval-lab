"""Unit tests for ragexp.retrieve — synthetic inputs, no dataset, no model."""
import zlib

import numpy as np
import pytest

from ragexp.data import Doc
from ragexp.retrieve import _chunks, bm25, build_bm25, dense, pool_doc_matrix, rrf


class FakeEmbedder:
    """Maps known strings to fixed unit vectors so dense() is testable
    without loading a model."""

    def __init__(self, table):
        self.table = table

    def encode(self, texts, normalize=True):
        return np.array([self.table[t] for t in texts], dtype=np.float32)


def test_dense_ranks_by_cosine():
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    mid = np.array([1.0, 1.0]) / np.sqrt(2)
    emb = FakeEmbedder({"q": e1})
    doc_matrix = np.stack([e2, mid, e1])
    ranked = dense("q", ["far", "mid", "near"], doc_matrix, emb)
    assert ranked == ["near", "mid", "far"]


def test_bm25_prefers_term_overlap():
    docs = [
        Doc("d1", "statins", "cholesterol statin drugs and cancer risk"),
        Doc("d2", "unrelated", "the weather in brussels is grey"),
    ]
    index, ids = build_bm25(docs)
    ranked = bm25("statin cholesterol", index, ids)
    assert ranked[0] == "d1"


# ── RRF ───────────────────────────────────────────────────────────────────


def test_rrf_agreement_wins():
    # "b" is ranked 2nd by both lists; beats docs that only one list likes.
    fused = rrf([["a", "b", "c"], ["d", "b", "c"]])
    assert fused[0] == "b"


def test_rrf_known_scores():
    fused_scores = {}
    for ranking in (["a", "b"], ["b", "a"]):
        for rank, d in enumerate(ranking, start=1):
            fused_scores[d] = fused_scores.get(d, 0) + 1 / (60 + rank)
    assert fused_scores["a"] == pytest.approx(fused_scores["b"])
    # symmetric case: both docs present in output
    assert set(rrf([["a", "b"], ["b", "a"]])) == {"a", "b"}


def test_rrf_depth_truncates_inputs():
    # With depth=1, "c" (rank 2+ everywhere) contributes from neither list.
    fused = rrf([["a", "c"], ["b", "c"]], depth=1)
    assert "c" not in fused
    assert set(fused) == {"a", "b"}


def test_rrf_depth_none_fuses_full_lists():
    fused = rrf([["a", "c"], ["b", "c"]], depth=None)
    assert set(fused) == {"a", "b", "c"}
    assert fused[0] == "c"  # only doc scored by both lists


def test_rrf_depth_changes_tail_not_head():
    # A doc deep in one list stops contributing under truncation, but the
    # head of the fused ranking (agreement docs) is stable.
    l1 = ["a", "b", "x", "y", "z"]
    l2 = ["a", "b", "z", "y", "x"]
    full = rrf([l1, l2])
    trunc = rrf([l1, l2], depth=2)
    assert full[:2] == trunc[:2] == ["a", "b"]


# ── Chunk pooling (nb 05) ─────────────────────────────────────────────────


class HashEmbedder:
    """Deterministic, deliberately non-unit vectors for any string, and a
    record of every encode call, so pooling is testable without a model."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def _vec(self, t):
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        return rng.normal(size=self.dim) * 3.0

    def encode(self, texts, normalize=True):
        texts = list(texts)
        self.calls.append(texts)
        return np.array([self._vec(t) for t in texts], dtype=np.float32)


def test_chunks_short_text_is_one_chunk():
    assert _chunks("short") == ["short"]
    assert _chunks("x" * 400) == ["x" * 400]


def test_chunks_windows_overlap_and_cover_the_text():
    text = "".join(chr(97 + i % 26) for i in range(1000))
    ch = _chunks(text)
    assert [len(c) for c in ch] == [400, 400, 360, 40]
    assert ch[1][:80] == ch[0][-80:]
    assert ch[-1] == text[960:]


def test_chunks_drops_whitespace_only_windows():
    text = "a" * 400 + " " * 400
    assert all(c.strip() for c in _chunks(text))


def test_pool_doc_matrix_title_and_full_modes():
    docs = [Doc("d1", "Title one", "body one"), Doc("d2", "", "body two " * 20)]
    emb = HashEmbedder()
    pool_doc_matrix(docs, emb, mode="title")
    assert emb.calls[-1] == ["Title one", ("body two " * 20)[:80]]
    pool_doc_matrix(docs, emb, mode="full")
    assert emb.calls[-1] == [d.full for d in docs]


def test_pool_doc_matrix_pool_matches_per_doc_blend():
    docs = [
        Doc("d1", "Short doc", "one chunk only"),
        Doc("d2", "Long doc", "lorem ipsum " * 100),
        Doc("d3", "Mid doc", "dolor sit amet " * 40),
    ]
    emb = HashEmbedder()
    got = pool_doc_matrix(docs, emb, mode="pool")
    assert got.shape == (3, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    for i, d in enumerate(docs):
        title = emb._vec(d.title)
        pooled = np.mean([emb._vec(c) for c in _chunks(d.text)], axis=0)
        blend = 0.25 * title + 0.75 * pooled
        np.testing.assert_allclose(got[i], blend / np.linalg.norm(blend), atol=1e-5)


def test_pool_doc_matrix_pool_encodes_in_two_calls():
    docs = [Doc(f"d{i}", f"t{i}", "word " * (50 * (i + 1))) for i in range(5)]
    emb = HashEmbedder()
    pool_doc_matrix(docs, emb, mode="pool")
    assert len(emb.calls) == 2


def test_pool_doc_matrix_unknown_mode_raises():
    with pytest.raises(ValueError):
        pool_doc_matrix([Doc("d1", "t", "x")], HashEmbedder(), mode="nope")
