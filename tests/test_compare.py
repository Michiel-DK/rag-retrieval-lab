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


# --- --metric-b: pair metric M of run a with metric M2 of run b ---------------------------------

def _depth_run(tmp_path):
    rows = {"q1": (0.2, 0.4), "q2": (0.5, 0.5), "q3": (0.1, 0.4), "q4": (0.3, 0.6)}
    p = tmp_path / "depth.json"
    p.write_text(json.dumps({q: {"recall@50": a, "recall@300": b} for q, (a, b) in rows.items()}))
    return str(p), rows


def test_diff_metric_b_within_one_run_is_mean_difference(tmp_path):
    p, rows = _depth_run(tmp_path)
    d = diff(p, p, metric="recall@50", metric_b="recall@300", order_file=tmp_path / "none.json")
    expected = sum(b - a for a, b in rows.values()) / len(rows)
    assert d.delta == pytest.approx(expected) and d.delta > 0
    assert d.mean_a == pytest.approx(sum(a for a, _ in rows.values()) / 4)
    assert d.mean_b == pytest.approx(sum(b for _, b in rows.values()) / 4)
    assert d.metric == "recall@50" and d.metric_b == "recall@300"


def test_diff_default_metric_b_is_metric(tmp_path):
    p, _ = _depth_run(tmp_path)
    same = diff(p, p, metric="recall@50", order_file=tmp_path / "none.json")
    explicit = diff(p, p, metric="recall@50", metric_b="recall@50", order_file=tmp_path / "none.json")
    assert same.delta == explicit.delta == 0 and same.mean_b == same.mean_a


def test_cli_diff_metric_b_shows_both_names(tmp_path, capsys):
    p, _ = _depth_run(tmp_path)
    assert main(["diff", p, p, "--metric", "recall@50", "--metric-b", "recall@300"]) == 0
    out = capsys.readouterr().out
    assert "recall@50" in out and "recall@300" in out and "CI excludes 0" in out


def test_cli_diff_without_metric_b_header_unchanged(tmp_path, capsys):
    p, _ = _depth_run(tmp_path)
    main(["diff", p, p, "--metric", "recall@50"])
    assert "recall@300" not in capsys.readouterr().out


def test_cli_missing_metric_b_exits_2(tmp_path, capsys):
    p, _ = _depth_run(tmp_path)
    assert main(["diff", p, p, "--metric", "recall@50", "--metric-b", "recall@999"]) == 2
    assert "recall@999" in capsys.readouterr().err
