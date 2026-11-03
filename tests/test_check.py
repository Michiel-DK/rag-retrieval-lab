"""ragexp check: comparison rules and CLI exit codes, on a toy corpus. No network, no model."""
import json

import pytest

import ragexp.check as check
import ragexp.methods as methods
import ragexp.runs as runs
from ragexp.check import compare_scores
from ragexp.cli import main
from ragexp.data import Corpus, Doc, Query
from ragexp.methods import Bm25
from ragexp.runs import evaluate_run

S = {"q1": {"recall@10": 0.5, "ndcg@10": 0.25}, "q2": {"recall@10": None, "ndcg@10": None}}


def copy(s):
    return {q: dict(r) for q, r in s.items()}


def test_identical_is_empty():
    assert compare_scores(S, copy(S)) == []


def test_recall_is_exact():
    g = copy(S)
    g["q1"]["recall@10"] += 1e-15
    assert g["q1"]["recall@10"] != 0.5
    assert [m.metric for m in compare_scores(S, g)] == ["recall@10"]


def test_ndcg_tolerance():
    g = copy(S)
    g["q1"]["ndcg@10"] += 1e-13
    assert compare_scores(S, g) == []
    g["q1"]["ndcg@10"] = 0.25 + 1e-9
    assert [m.metric for m in compare_scores(S, g)] == ["ndcg@10"]


def test_none_only_equals_none():
    g = copy(S)
    g["q2"]["ndcg@10"] = 0.0
    assert len(compare_scores(S, g)) == 1
    assert len(compare_scores(g, S)) == 1


def test_query_on_one_side_is_mismatch():
    g = copy(S)
    del g["q2"]
    assert [(m.query_id, m.metric) for m in compare_scores(S, g)] == [("q2", "<missing query>")]
    g = copy(S)
    g["q3"] = {"recall@10": 1.0}
    assert [(m.query_id, m.metric) for m in compare_scores(S, g)] == [("q3", "<missing query>")]


def test_metric_on_one_side_is_mismatch():
    g = copy(S)
    g["q1"]["recall@50"] = 1.0
    assert [(m.query_id, m.metric) for m in compare_scores(S, g)] == [("q1", "recall@50")]
    assert [(m.query_id, m.metric) for m in compare_scores(g, S)] == [("q1", "recall@50")]


# --- CLI on a toy corpus -------------------------------------------------------------------

CORPUS = Corpus(
    [Doc("d1", "statins", "cholesterol statin drugs"), Doc("d2", "weather", "grey weather in brussels")],
    [Query("q1", "statin cholesterol"), Query("q2", "grey weather")],
    {"q1": {"d1": 2}, "q2": {"d2": 1}},
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    monkeypatch.setattr(runs, "_SCORES", scores_dir)
    monkeypatch.setattr(check, "load_nfcorpus", lambda: CORPUS)
    monkeypatch.setitem(methods.REGISTRY, "toy", Bm25)
    monkeypatch.setitem(methods.REGISTRY, "nofile", Bm25)
    good = evaluate_run(CORPUS, Bm25().rank(CORPUS), recall_ks=(10, 50), ndcg_ks=(10,))
    (scores_dir / "toy.json").write_text(json.dumps(good, indent=1, sort_keys=True))
    return scores_dir


def snapshot(d):
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in d.iterdir()}


def test_exit_0_when_identical(env, capsys):
    assert main(["check", "toy"]) == 0
    assert capsys.readouterr().out.startswith("OK  toy  2 queries identical")


def test_exit_1_when_one_value_differs(env, capsys):
    p = env / "toy.json"
    d = json.loads(p.read_text())
    d["q1"]["recall@10"] = 0.123
    p.write_text(json.dumps(d))
    assert main(["check", "toy"]) == 1
    assert "FAIL toy  1 mismatches" in capsys.readouterr().out


def test_all_checks_every_registered_method(env, monkeypatch):
    monkeypatch.setattr(methods, "REGISTRY", {"toy": Bm25})
    assert main(["check", "--all"]) == 0


def test_all_exits_2_when_a_registered_method_has_no_committed_file(env, capsys):
    # #40 review: --all used to skip "nofile" and pass. A skip is not a check.
    assert main(["check", "--all"]) == 2
    assert "no committed scores" in capsys.readouterr().err


def test_all_exits_2_on_empty_registry(env, monkeypatch):
    monkeypatch.setattr(methods, "REGISTRY", {})
    assert main(["check", "--all"]) == 2


def test_exit_2_unknown_method(env, capsys):
    assert main(["check", "nope"]) == 2
    assert "registered:" in capsys.readouterr().err


def test_exit_2_method_without_committed_file(env):
    assert main(["check", "nofile"]) == 2


def test_exit_2_neither_names_nor_all(env):
    assert main(["check"]) == 2


def test_exit_2_both_names_and_all(env):
    assert main(["check", "toy", "--all"]) == 2


def test_check_never_writes_scores_dir(env):
    before = snapshot(env)
    main(["check", "toy"])
    main(["check", "--all"])
    main(["check", "nofile"])
    assert snapshot(env) == before
