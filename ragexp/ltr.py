"""Learning-to-rank helpers for MSLR-WEB10K (nb 06).

MSLR ships one LETOR line per (query, document) pair: a relevance grade 0..4,
a ``qid`` and 136 numeric features. There are no document ids and no text, so
a document is named here by its position inside its query block
(``"<qid>:<row>"``). That is enough for ``ragexp.metrics.ndcg_at_k``, which
only needs a ranked id list and an id -> grade map.

The data stays outside the repo (the licence does not allow redistribution).
``load_split`` reads one file of a fold with ``load_svmlight_file``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import ndcg_at_k

Groups = List[Tuple[str, int, int]]  # (qid, start row, end row), end exclusive


def load_split(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one LETOR file -> (X dense float32, grades int, qids as str)."""
    from sklearn.datasets import load_svmlight_file

    X, y, qid = load_svmlight_file(str(path), query_id=True)
    return X.toarray().astype(np.float32), y.astype(int), qid.astype(str)


def query_groups(qids: Sequence[str]) -> Groups:
    """Contiguous blocks of rows per query, in file order.

    Rankers and ``XGBRanker`` both assume a query's rows are adjacent; a qid
    that reappears after another query would silently merge two lists, so it
    raises instead.
    """
    groups: Groups = []
    seen = set()
    start = 0
    for i in range(1, len(qids) + 1):
        if i == len(qids) or qids[i] != qids[start]:
            if qids[start] in seen:
                raise ValueError(f"qid {qids[start]} is not contiguous")
            seen.add(qids[start])
            groups.append((qids[start], start, i))
            start = i
    return groups


def group_index(groups: Groups) -> np.ndarray:
    """Per-row block number 0, 1, 2, ... in file order.

    ``XGBRanker`` wants ``qid`` non-decreasing. MSLR's blocks are contiguous
    but not sorted by qid, so the block number stands in for it: same
    grouping, monotone by construction.
    """
    return np.repeat(np.arange(len(groups)), [e - s for _, s, e in groups])


def grades_by_query(y: np.ndarray, groups: Groups) -> Dict[str, Dict[str, int]]:
    return {q: {f"{q}:{r}": int(y[s + r]) for r in range(e - s)} for q, s, e in groups}


def rank_by_scores(scores: np.ndarray, groups: Groups) -> Dict[str, List[str]]:
    """Per query, doc ids by descending score; ties keep file order."""
    out = {}
    for q, s, e in groups:
        order = np.argsort(-scores[s:e], kind="stable")
        out[q] = [f"{q}:{r}" for r in order]
    return out


def ndcg_by_query(
    rankings: Dict[str, List[str]],
    grades: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, Optional[float]]:
    return {q: ndcg_at_k(rankings[q], grades[q], k) for q in rankings}


def best_single_feature(
    X: np.ndarray, y: np.ndarray, groups: Groups, k: int = 10
) -> Tuple[int, int, float]:
    """The one feature (and direction) that ranks best on this split.

    Tries every column with sign +1 (higher is better) and -1 (lower is
    better) and returns ``(column, sign, mean nDCG@k)``. Meant for the
    validation split: picking it on test would leak the answer.
    """
    grades = grades_by_query(y, groups)
    best = (-1, 0, -np.inf)
    for col in range(X.shape[1]):
        for sign in (1, -1):
            vals = ndcg_by_query(rank_by_scores(sign * X[:, col], groups), grades, k)
            defined = [v for v in vals.values() if v is not None]
            mean = float(np.mean(defined)) if defined else -np.inf
            if mean > best[2]:
                best = (col, sign, mean)
    return best
