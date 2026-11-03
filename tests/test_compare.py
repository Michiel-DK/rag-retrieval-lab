"""`ragexp diff` against committed evidence. Needs only results/, no corpus download."""
import json

import pytest

from ragexp.cli import main
from ragexp.compare import diff


def test_diff_reproduces_showcase_llm_rerank():
    # SHOWCASE.md: LLM-judge rerank 0.4052 (+0.0885, CI +0.071 – +0.106), corpus order, seed 42.
    d = diff("dense", "rerank_llm")
    assert d.order == "corpus" and d.n == 323
    assert round(d.mean_b, 4) == 0.4052
    assert round(d.delta, 4) == 0.0885
    assert (round(d.ci[0], 3), round(d.ci[1], 3)) == (0.071, 0.106)


def test_sorted_order_would_not_reproduce_showcase(tmp_path):
    # Why query_order.json exists: sorted ids shift the CI low end to +0.072.
    d = diff("dense", "rerank_llm", order_file=tmp_path / "missing.json")
    assert d.order == "sorted"
    assert round(d.ci[0], 3) != 0.071


def test_diff_accepts_paths_and_drops_unshared_queries(tmp_path):
    a = {"q1": {"ndcg@10": 0.1}, "q2": {"ndcg@10": 0.2}, "q3": {"ndcg@10": None}}
    b = {"q1": {"ndcg@10": 0.3}, "q2": {"ndcg@10": 0.4}, "q3": {"ndcg@10": 0.9}, "q4": {"ndcg@10": 1.0}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    d = diff(str(pa), str(pb), order_file=tmp_path / "none.json")
    assert d.n == 2 and d.delta == pytest.approx(0.2)


def test_cli_diff_prints_and_exits_zero(capsys):
    assert main(["diff", "dense", "rerank_llm"]) == 0
    out = capsys.readouterr().out
    assert "+0.0885" in out and "CI excludes 0" in out


def test_cli_unknown_run_exits_2(capsys):
    assert main(["diff", "dense", "no_such_run"]) == 2
    assert "no scores file" in capsys.readouterr().err


def test_cli_missing_metric_exits_2(capsys):
    assert main(["diff", "dense", "rerank_llm", "--metric", "mrr@10"]) == 2
