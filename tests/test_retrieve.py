"""Unit tests for ragexp.retrieve — synthetic inputs, no dataset, no model."""
import numpy as np
import pytest

from ragexp.data import Doc
from ragexp.retrieve import bm25, build_bm25, dense, rrf


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
