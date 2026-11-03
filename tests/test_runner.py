"""ragexp run: writes new score files, never overwrites one. Toy corpus, no network, no model."""
import json

import pytest

import ragexp.methods as methods
import ragexp.runner as runner
import ragexp.runs as runs
from ragexp.check import compare_scores
from ragexp.cli import main
from ragexp.data import Corpus, Doc, Query
from ragexp.methods import Bm25, Rrf
from ragexp.runner import parse_params
from ragexp.runs import evaluate_run

CORPUS = Corpus(
    [Doc("d1", "statins", "cholesterol statin drugs"), Doc("d2", "weather", "grey weather in brussels")],
    [Query("q1", "statin cholesterol"), Query("q2", "grey weather")],
    {"q1": {"d1": 2}, "q2": {"d2": 1}},
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "scores"
    d.mkdir()
    monkeypatch.setattr(runs, "_SCORES", d)
    monkeypatch.setattr(runner, "load_nfcorpus", lambda: CORPUS)
    return d


def test_parse_params_types():
    assert parse_params(["depth=100", "k=60", "depth2=null", "model=a/b"]) == {
        "depth": 100, "k": 60, "depth2": None, "model": "a/b",
    }
    with pytest.raises(ValueError):
        parse_params(["nokey"])


def test_run_writes_new_file_equal_to_evaluate_run(env, capsys):
    assert main(["run", "bm25"]) == 0
    written = json.loads((env / "bm25.json").read_text())
    expected = evaluate_run(CORPUS, Bm25().rank(CORPUS), recall_ks=(10, 50), ndcg_ks=(10,))
    assert compare_scores(expected, written) == []
    assert "wrote" in capsys.readouterr().out


def test_run_refuses_existing_file_before_compute(env, monkeypatch, capsys):
    p = env / "bm25.json"
    p.write_text('{"sentinel": {}}')
    before = (p.stat().st_mtime_ns, p.read_bytes())

    def boom():
        raise AssertionError("corpus loaded: refusal came after compute")

    monkeypatch.setattr(runner, "load_nfcorpus", boom)
    assert main(["run", "bm25"]) == 2
    assert (p.stat().st_mtime_ns, p.read_bytes()) == before
    assert "never regenerated" in capsys.readouterr().err


def test_run_with_name_writes_variant_and_passes_params(env, monkeypatch):
    seen = {}

    class Spy(Rrf):
        def rank(self, corpus):
            seen.update(self.params())
            return Bm25().rank(corpus)

    monkeypatch.setitem(methods.REGISTRY, "rrf", Spy)
    assert main(["run", "rrf", "--name", "rrf_depth100", "--param", "depth=100"]) == 0
    assert (env / "rrf_depth100.json").is_file() and not (env / "rrf.json").exists()
    assert seen["depth"] == 100


def test_run_exit_2_unknown_method_and_bad_param(env):
    assert main(["run", "colbert"]) == 2
    assert main(["run", "bm25", "--param", "depth=3"]) == 2  # Bm25 takes no depth
    assert list(env.iterdir()) == []


def test_run_against_real_results_dir_refuses_every_committed_method():
    # The real committed files: run must refuse all of them without computing.
    for name in ("bm25", "dense", "rrf"):
        assert (runs._SCORES / f"{name}.json").is_file()
        assert main(["run", name]) == 2
