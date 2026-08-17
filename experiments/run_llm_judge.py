"""Produce results/rerank/llm_grades.json — the committed LLM-judge sweep.

Grades the dense top-50 pool for every judged query with a listwise 0-3
LLM judge running on headless Claude Code (subscription auth, no API key).
Checkpointed per query: safe to kill and re-run, resumes where it stopped.

Usage:
    python experiments/run_llm_judge.py            # full sweep (323 queries)
    python experiments/run_llm_judge.py --limit 3  # prototype run
"""
import argparse

from ragexp.data import load_nfcorpus
from ragexp.embed import Embedder
from ragexp.rerank import ClaudeCLIJudge, sweep
from ragexp.retrieve import dense

POOL_SIZE = 50
SWEEP_NAME = "llm_grades"


def main(limit: int | None = None) -> None:
    corpus = load_nfcorpus()
    emb = Embedder()
    doc_matrix = emb.encode([d.full for d in corpus.docs])

    queries = corpus.queries[:limit] if limit else corpus.queries
    pool = {
        q.query_id: dense(q.text, corpus.doc_ids, doc_matrix, emb)[:POOL_SIZE]
        for q in queries
    }
    query_texts = {q.query_id: q.text for q in queries}
    doc_texts = {d.doc_id: d.full for d in corpus.docs}

    judge = ClaudeCLIJudge(model="opus")

    def score_fn(query_text, docs):
        return {k: float(v) for k, v in judge.grade(query_text, docs).items()}

    done = sweep(SWEEP_NAME, pool, query_texts, doc_texts, score_fn)
    print(f"done: {len(done)} queries graded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
