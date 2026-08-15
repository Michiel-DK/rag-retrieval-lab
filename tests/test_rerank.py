"""Unit tests for ragexp.rerank — fake judges, no subprocess, no model."""
import json

import pytest

from ragexp.rerank import ClaudeCLIJudge, rerank_by_scores, sweep


# ── the contract ──────────────────────────────────────────────────────────


def test_rerank_is_a_permutation():
    pool = ["a", "b", "c"]
    out = rerank_by_scores(pool, {"a": 0.1, "b": 0.9, "c": 0.5})
    assert sorted(out) == sorted(pool)
    assert out == ["b", "c", "a"]


def test_rerank_ties_keep_original_order():
    pool = ["a", "b", "c"]
    assert rerank_by_scores(pool, {"a": 1.0, "b": 1.0, "c": 1.0}) == pool


def test_rerank_missing_scores_sink_to_bottom():
    assert rerank_by_scores(["a", "b"], {"b": 0.2}) == ["b", "a"]


# ── judge reply parsing ───────────────────────────────────────────────────


def test_parse_clean_json():
    grades = ClaudeCLIJudge._parse('{"1": 3, "2": 0}', n=2)
    assert grades == {1: 3, 2: 0}


def test_parse_json_embedded_in_prose():
    raw = 'Here are the grades:\n{"1": 2, "2": 1}\nDone.'
    assert ClaudeCLIJudge._parse(raw, n=2) == {1: 2, 2: 1}


def test_parse_rejects_missing_indices():
    with pytest.raises(ValueError, match="missing"):
        ClaudeCLIJudge._parse('{"1": 2}', n=3)


def test_parse_rejects_out_of_range_grades():
    with pytest.raises(ValueError, match="range"):
        ClaudeCLIJudge._parse('{"1": 7, "2": 0}', n=2)


def test_parse_rejects_no_json():
    with pytest.raises(ValueError, match="no JSON"):
        ClaudeCLIJudge._parse("I cannot grade these.", n=2)


# ── checkpointed sweep ────────────────────────────────────────────────────


def test_sweep_checkpoints_and_resumes(tmp_path, monkeypatch):
    import ragexp.rerank as rr

    monkeypatch.setattr(rr, "_RERANK_DIR", tmp_path)

    pool = {"q1": ["d1", "d2"], "q2": ["d2", "d3"]}
    qtexts = {"q1": "one", "q2": "two"}
    dtexts = {"d1": "x", "d2": "y", "d3": "z"}
    calls = []

    def fake_score(qtext, docs):
        calls.append(qtext)
        return {doc_id: 1.0 for doc_id, _ in docs}

    out = sweep("t", pool, qtexts, dtexts, fake_score, verbose=False)
    assert set(out) == {"q1", "q2"}
    assert len(calls) == 2
    # checkpoint file written
    assert json.loads((tmp_path / "t.json").read_text()) == out

    # resume: nothing re-runs
    out2 = sweep("t", pool, qtexts, dtexts, fake_score, verbose=False)
    assert out2 == out
    assert len(calls) == 2


def test_sweep_partial_checkpoint_only_runs_missing(tmp_path, monkeypatch):
    import ragexp.rerank as rr

    monkeypatch.setattr(rr, "_RERANK_DIR", tmp_path)
    (tmp_path / "t.json").write_text(json.dumps({"q1": {"d1": 2.0}}))

    pool = {"q1": ["d1"], "q2": ["d2"]}
    calls = []

    def fake_score(qtext, docs):
        calls.append(qtext)
        return {"d2": 1.0}

    out = sweep("t", pool, {"q1": "a", "q2": "b"}, {"d1": "x", "d2": "y"},
                fake_score, verbose=False)
    assert calls == ["b"]
    assert out["q1"] == {"d1": 2.0}
