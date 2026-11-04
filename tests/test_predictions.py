"""Predict-first gate: ragexp run refuses to write evidence without a committed prediction."""
import functools
import subprocess
from pathlib import Path

import pytest

import ragexp.paths as paths
import ragexp.runner as runner
import ragexp.runs as runs
from ragexp.cli import main
from ragexp.data import Corpus, Doc, Query
from ragexp.predictions import PredictionMissing, require_prediction, sections

GOOD = "# rrf_depth100\n\n## Prediction\nndcg@10 within 0.01 of rrf.\n\n## Mechanism\nDepth 100 vs 1000 only moves the tail.\n"
CORPUS = Corpus(
    [Doc("d1", "statins", "cholesterol statin drugs"), Doc("d2", "weather", "grey weather in brussels")],
    [Query("q1", "statin cholesterol"), Query("q2", "grey weather")],
    {"q1": {"d1": 2}, "q2": {"d2": 1}},
)


def git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "predictions").mkdir(parents=True)
    git(r, "init", "-q")
    (r / "README").write_text("x")
    git(r, "add", "README")
    git(r, "commit", "-q", "-m", "init")
    return r


def commit_prediction(repo, name, text=GOOD):
    (repo / "predictions" / f"{name}.md").write_text(text)
    git(repo, "add", f"predictions/{name}.md")
    git(repo, "commit", "-q", "-m", f"predict {name}")


def test_sections_parses_level2_headings():
    assert sections(GOOD) == {"Prediction": "ndcg@10 within 0.01 of rrf.", "Mechanism": "Depth 100 vs 1000 only moves the tail."}


def test_committed_prediction_passes_and_returns_commit(repo):
    commit_prediction(repo, "p")
    ref = require_prediction("p", repo / "predictions")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert ref.commit == head


def test_missing_file_refused(repo):
    with pytest.raises(PredictionMissing, match="no prediction"):
        require_prediction("p", repo / "predictions")


@pytest.mark.parametrize("text", [GOOD.replace("## Mechanism", "## Notes"), "## Prediction\n\n## Mechanism\nwhy\n"])
def test_missing_or_empty_section_refused(repo, text):
    commit_prediction(repo, "p", text)
    with pytest.raises(PredictionMissing, match="section"):
        require_prediction("p", repo / "predictions")


def test_untracked_refused(repo):
    (repo / "predictions" / "p.md").write_text(GOOD)
    with pytest.raises(PredictionMissing, match="not committed"):
        require_prediction("p", repo / "predictions")


def test_staged_but_uncommitted_refused(repo):
    (repo / "predictions" / "p.md").write_text(GOOD)
    git(repo, "add", "predictions/p.md")
    with pytest.raises(PredictionMissing, match="uncommitted"):
        require_prediction("p", repo / "predictions")


def test_edited_after_commit_refused(repo):
    commit_prediction(repo, "p")
    (repo / "predictions" / "p.md").write_text(GOOD + "\nedited after seeing something\n")
    with pytest.raises(PredictionMissing, match="uncommitted"):
        require_prediction("p", repo / "predictions")


# --- the gate inside `ragexp run` ---------------------------------------------------------------


@pytest.fixture
def evidence_env(repo, monkeypatch):
    """Point the evidence location at the temp repo, so runs there are gated."""
    results = repo / "results"
    (results / "scores").mkdir(parents=True)
    monkeypatch.setattr(paths, "EVIDENCE_RESULTS_DIR", results)
    monkeypatch.setattr(runs, "_SCORES", results / "scores")
    monkeypatch.setattr(runner, "require_prediction", functools.partial(require_prediction, predictions_dir=repo / "predictions"))
    monkeypatch.setattr(runner, "load_nfcorpus", lambda: CORPUS)
    return repo


def test_run_without_prediction_refused_before_compute(evidence_env, monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_nfcorpus", lambda: pytest.fail("corpus loaded before the gate"))
    assert main(["run", "bm25", "--name", "bm25_new"]) == 2
    assert "no prediction" in capsys.readouterr().err
    assert list((evidence_env / "results" / "scores").iterdir()) == []


def test_run_with_committed_prediction_writes_and_reports_commit(evidence_env, capsys):
    commit_prediction(evidence_env, "bm25_new")
    assert main(["run", "bm25", "--name", "bm25_new"]) == 0
    out = capsys.readouterr().out
    assert (evidence_env / "results" / "scores" / "bm25_new.json").is_file()
    assert "prediction: bm25_new.md @" in out


def test_exploratory_run_outside_evidence_dir_is_not_gated(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runs, "_SCORES", tmp_path / "scratch" / "scores")
    monkeypatch.setattr(runner, "load_nfcorpus", lambda: CORPUS)
    assert main(["run", "bm25"]) == 0
    assert "exploratory" in capsys.readouterr().out


def test_real_checkout_is_the_gated_location():
    assert paths.EVIDENCE_RESULTS_DIR is not None
    assert runner.writes_evidence()  # default config in this repo: runs are evidence, so gated


FENCED_ONLY = "# fake\n\n```markdown\n## Prediction\nsomething\n\n## Mechanism\nbecause\n```\n"


def test_sections_ignores_headings_inside_fences():
    assert sections(FENCED_ONLY) == {}
    assert sections("## A\nx\n~~~\n## B\n~~~\ny\n") == {"A": "x\n~~~\n## B\n~~~\ny"}
    assert sections("## A\n  ```\n## B\n  ```\n## C\nz") == {"A": "```\n## B\n  ```", "C": "z"}
    # a longer closing fence closes; a different fence char does not
    assert sections("## A\n```\n## B\n````\n## C\nz") == {"A": "```\n## B\n````", "C": "z"}
    assert "B" not in sections("## A\n```\n~~~\n## B\n```\n")
    # unclosed fence runs to the end of the text
    assert sections("## A\nx\n```\n## B\ny") == {"A": "x\n```\n## B\ny"}


def test_fenced_only_headings_refused_under_non_readme_name(repo):
    commit_prediction(repo, "p", FENCED_ONLY)
    with pytest.raises(PredictionMissing):
        require_prediction("p", repo / "predictions")


def test_prediction_with_fenced_block_in_mechanism_still_parses():
    text = "## Prediction\nup.\n\n## Mechanism\nBecause:\n\n```\n## Something\nx = 1\n```\n\nDone.\n"
    got = sections(text)
    assert set(got) == {"Prediction", "Mechanism"}
    assert "## Something\nx = 1" in got["Mechanism"] and got["Mechanism"].endswith("Done.")


def test_current_predictions_readme_is_not_a_prediction(repo):
    text = (Path(__file__).resolve().parent.parent / "predictions" / "README.md").read_text()
    commit_prediction(repo, "README", text)
    with pytest.raises(PredictionMissing):
        require_prediction("README", repo / "predictions")


@pytest.mark.parametrize("name", ["README", "readme", "ReadMe"])
def test_readme_name_refused_even_if_well_formed(repo, name):
    commit_prediction(repo, "README", GOOD)
    with pytest.raises(PredictionMissing, match="format doc"):
        require_prediction(name, repo / "predictions")
