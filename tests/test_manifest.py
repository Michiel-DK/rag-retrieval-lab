"""Run manifest: written by `ragexp run`, used by `ragexp check` to rebuild variants."""
import json
from dataclasses import dataclass

import pytest

import ragexp.check as check
import ragexp.methods as methods
import ragexp.runner as runner
import ragexp.runs as runs
from ragexp.cli import main
from ragexp.data import Corpus, Doc, Query
from ragexp.manifest import load_manifest, sha256
from ragexp.methods import Bm25

CORPUS = Corpus(
    [Doc("d1", "statins", "cholesterol statin drugs"), Doc("d2", "weather", "grey weather in brussels")],
    [Query("q1", "statin cholesterol"), Query("q2", "grey weather")],
    {"q1": {"d1": 2}, "q2": {"d2": 1}},
)


@dataclass
class Rev:
    """bm25, optionally reversed: a cheap method whose scores depend on a param."""

    reverse: bool = False
    name: str = "rev"

    def params(self):
        return {"reverse": self.reverse}

    def rank(self, corpus):
        r = Bm25().rank(corpus)
        return {q: v[::-1] for q, v in r.items()} if self.reverse else r


class NotADataclass:
    name = "nodc"

    def params(self):
        return {}

    def rank(self, corpus):
        raise AssertionError("must be refused before compute")


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "results" / "scores"
    d.mkdir(parents=True)
    monkeypatch.setattr(runs, "_SCORES", d)  # exploratory location: no prediction gate here
    monkeypatch.setattr(runner, "load_nfcorpus", lambda: CORPUS)
    monkeypatch.setattr(check, "load_nfcorpus", lambda: CORPUS)
    monkeypatch.setitem(methods.REGISTRY, "rev", Rev)
    monkeypatch.setitem(methods.REGISTRY, "nodc", NotADataclass)
    return tmp_path / "results"


def test_run_writes_manifest_describing_the_scores_file(env):
    assert main(["run", "rev", "--name", "rev_true", "--param", "reverse=true"]) == 0
    m = load_manifest("rev_true")
    assert (env / "manifests" / "rev_true.json").is_file()
    assert m["method"] == "rev" and m["init_params"] == {"reverse": True}
    assert m["scores_sha256"] == sha256(env / "scores" / "rev_true.json")
    assert m["exploratory"] is True and m["prediction"] is None
    assert m["n_queries"] == 2 and set(m["means"]) == {"recall@10", "recall@50", "ndcg@10"}
    assert m["versions"]["ragexp"] and m["schema"] == 1
    assert set(m["repo"]) == {"commit", "dirty"}


def test_check_rebuilds_variant_from_manifest(env, monkeypatch):
    main(["run", "rev", "--name", "rev_true", "--param", "reverse=true"])
    monkeypatch.setattr(methods, "REGISTRY", {"rev": Rev})  # only rev + the manifested variant
    assert main(["check", "rev_true"]) == 0


def test_check_fails_when_manifest_params_do_not_produce_the_file(env, capsys):
    main(["run", "rev", "--name", "rev_true", "--param", "reverse=true"])
    p = env / "manifests" / "rev_true.json"
    m = json.loads(p.read_text())
    m["init_params"]["reverse"] = False
    p.write_text(json.dumps(m))
    assert main(["check", "rev_true"]) == 1
    assert "FAIL rev_true" in capsys.readouterr().out


def test_check_all_includes_manifested_runs(env, monkeypatch, capsys):
    main(["run", "rev", "--name", "rev_true", "--param", "reverse=true"])
    main(["run", "rev"])
    monkeypatch.setattr(methods, "REGISTRY", {"rev": Rev})
    assert main(["check", "--all"]) == 0
    out = capsys.readouterr().out
    assert "OK  rev " in out and "OK  rev_true " in out


def test_variant_without_manifest_is_unknown_to_check(env):
    main(["run", "rev", "--name", "rev_true", "--param", "reverse=true"])
    (env / "manifests" / "rev_true.json").unlink()
    assert main(["check", "rev_true"]) == 2


def test_run_refuses_orphan_manifest_and_non_dataclass_before_compute(env, monkeypatch):
    (env / "manifests").mkdir()
    (env / "manifests" / "rev.json").write_text("{}")
    monkeypatch.setattr(runner, "load_nfcorpus", lambda: pytest.fail("computed before refusing"))
    assert main(["run", "rev"]) == 2
    assert main(["run", "nodc"]) == 2
    assert list((env / "scores").iterdir()) == []


# --- --recall-k: the scoring call is recorded in the manifest and re-derived by check ----------

def _edit_manifest(env, name, fn):
    p = env / "manifests" / f"{name}.json"
    m = json.loads(p.read_text())
    fn(m)
    p.write_text(json.dumps(m))


def test_run_recall_k_scores_exactly_the_given_ks(env):
    assert main(["run", "rev", "--name", "deep", "--recall-k", "300", "--recall-k", "10", "--recall-k", "300"]) == 0
    rows = json.loads((env / "scores" / "deep.json").read_text())
    assert rows and all(set(r) == {"recall@10", "recall@300", "ndcg@10"} for r in rows.values())


def test_run_manifest_records_eval_ks_and_default_run_records_defaults(env):
    main(["run", "rev", "--name", "deep", "--recall-k", "10", "--recall-k", "300"])
    main(["run", "rev"])
    m = load_manifest("deep")
    assert m["eval"] == {"recall_ks": [10, 300], "ndcg_ks": [10]}
    assert "recall_ks" not in m["init_params"]
    assert load_manifest("rev")["eval"] == {"recall_ks": [10, 50], "ndcg_ks": [10]}
    assert set(json.loads((env / "scores" / "rev.json").read_text())["q1"]) == {"recall@10", "recall@50", "ndcg@10"}


def test_check_uses_manifest_ks_and_fails_when_they_are_edited(env, monkeypatch, capsys):
    main(["run", "rev", "--name", "deep", "--recall-k", "10", "--recall-k", "300"])
    monkeypatch.setattr(methods, "REGISTRY", {"rev": Rev})
    assert main(["check", "deep"]) == 0
    _edit_manifest(env, "deep", lambda m: m["eval"].update(recall_ks=[10, 50]))
    assert main(["check", "deep"]) == 1  # scores file has recall@300, recompute with (10, 50) does not
    assert "recall@300" in capsys.readouterr().out


def test_check_manifest_without_eval_field_uses_10_50(env, monkeypatch):
    main(["run", "rev", "--name", "old"])
    _edit_manifest(env, "old", lambda m: m.pop("eval"))
    monkeypatch.setattr(methods, "REGISTRY", {"rev": Rev})
    assert main(["check", "old"]) == 0


@pytest.mark.parametrize("bad", ["0", "-5"])
def test_run_nonpositive_recall_k_exits_2_before_corpus_load(env, monkeypatch, bad, capsys):
    def boom():
        raise AssertionError("corpus loaded")

    monkeypatch.setattr(runner, "load_nfcorpus", boom)
    assert main(["run", "rev", "--name", "deep", "--recall-k", "10", "--recall-k", bad]) == 2
    assert "--recall-k" in capsys.readouterr().err
    assert list((env / "scores").iterdir()) == []
