"""Unit tests for ragexp.ltr — synthetic LETOR-shaped data, no MSLR download."""
import numpy as np
import pytest

from ragexp.ltr import (
    best_single_feature,
    grades_by_query,
    group_index,
    load_split,
    ndcg_by_query,
    query_groups,
    rank_by_scores,
)


def test_query_groups_are_contiguous_blocks():
    assert query_groups(["1", "1", "2", "2", "2", "3"]) == [("1", 0, 2), ("2", 2, 5), ("3", 5, 6)]


def test_query_groups_reject_a_qid_that_reappears():
    with pytest.raises(ValueError):
        query_groups(["1", "2", "1"])


def test_rank_by_scores_descending_ties_keep_file_order():
    groups = query_groups(["7", "7", "7", "7"])
    ranked = rank_by_scores(np.array([0.5, 0.9, 0.5, 0.1]), groups)
    assert ranked == {"7": ["7:1", "7:0", "7:2", "7:3"]}


def test_ndcg_by_query_perfect_and_undefined():
    y = np.array([2, 0, 0, 0])
    groups = query_groups(["a", "a", "b", "b"])
    grades = grades_by_query(y, groups)
    vals = ndcg_by_query(rank_by_scores(np.array([1.0, 0.0, 1.0, 0.0]), groups), grades)
    assert vals["a"] == pytest.approx(1.0)
    assert vals["b"] is None  # no relevant doc: undefined, not 0.0


def test_best_single_feature_finds_the_signal_and_its_direction():
    rng = np.random.default_rng(0)
    qids, rows, ys = [], [], []
    for q in range(30):
        y = rng.integers(0, 5, size=12)
        noise = rng.normal(size=(12, 3))
        signal = -y + 0.01 * rng.normal(size=12)  # informative, but lower is better
        rows.append(np.column_stack([noise[:, 0], signal, noise[:, 1:]]))
        ys.append(y)
        qids += [str(q)] * 12
    X, y = np.vstack(rows), np.concatenate(ys)
    col, sign, score = best_single_feature(X, y, query_groups(qids))
    assert (col, sign) == (1, -1)
    assert score > 0.95


def test_load_split_reads_letor_lines(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_text("2 qid:1 1:0.5 2:3\n0 qid:1 1:0.1 2:0\n1 qid:4 1:0.0 2:1\n")
    X, y, qid = load_split(p)
    assert X.shape == (3, 2) and X.dtype == np.float32
    assert y.tolist() == [2, 0, 1]
    assert qid.tolist() == ["1", "1", "4"]


def test_group_index_is_monotone_for_unsorted_qids():
    groups = query_groups(["9", "9", "2", "5", "5", "5"])
    assert group_index(groups).tolist() == [0, 0, 1, 2, 2, 2]
