"""Score a retrieval run and persist per-query results.

A "run" is a ranking per query. Scoring one produces per-query metrics —
kept per-query, never pre-averaged, because the paired bootstrap needs the
query-level scores (see notebook 01). Saved runs land in ``results/scores/``
so later notebooks can compare against earlier methods without recomputing
them, and so the comparisons are reproducible from the committed files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .data import Corpus
from .metrics import mean_ignoring_none, ndcg_at_k, recall_at_k

_SCORES = Path(__file__).resolve().parent.parent / "results" / "scores"

Scores = Dict[str, Dict[str, Optional[float]]]  # query_id -> metric -> value


def evaluate_run(
    corpus: Corpus,
    rankings: Dict[str, Sequence[str]],
    recall_ks: Sequence[int] = (10, 50),
    ndcg_ks: Sequence[int] = (10,),
) -> Scores:
    """Score ``rankings`` (query_id -> ranked doc_ids) against the qrels.

    Every judged query in ``rankings`` gets an entry; metrics follow the
    None-when-undefined rule from ``ragexp.metrics``.
    """
    scores: Scores = {}
    for q in corpus.queries:
        if q.query_id not in rankings:
            continue
        ranked = list(rankings[q.query_id])
        grades = corpus.qrels.get(q.query_id, {})
        relevant = corpus.relevant_ids(q.query_id)
        row: Dict[str, Optional[float]] = {}
        for k in recall_ks:
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        for k in ndcg_ks:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
        scores[q.query_id] = row
    return scores


def summarize(scores: Scores) -> Dict[str, Optional[float]]:
    """Mean of each metric across queries, dropping None per the ruler."""
    metrics: Dict[str, List[Optional[float]]] = {}
    for row in scores.values():
        for m, v in row.items():
            metrics.setdefault(m, []).append(v)
    return {m: mean_ignoring_none(vals) for m, vals in metrics.items()}


def metric_vector(scores: Scores, metric: str, query_ids: Sequence[str]) -> List[Optional[float]]:
    """Per-query values for ``metric`` in a fixed query order — the paired
    bootstrap needs both systems' scores aligned on the same queries."""
    return [scores.get(qid, {}).get(metric) for qid in query_ids]


def save_scores(name: str, scores: Scores) -> Path:
    _SCORES.mkdir(parents=True, exist_ok=True)
    path = _SCORES / f"{name}.json"
    path.write_text(json.dumps(scores, indent=1, sort_keys=True))
    return path


def load_scores(name: str) -> Scores:
    return json.loads((_SCORES / f"{name}.json").read_text())
