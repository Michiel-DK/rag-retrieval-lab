"""Produce results/rerank/gliclass_scores.json — the zero-shot classifier sweep.

Scores the same dense top-50 pool as the LLM judge with
`knowledgator/gliclass-modern-base-v3.0` (zero-shot
classifier, query as the only label). Checkpointed per query.

Usage:
    python experiments/run_gliclass.py            # full sweep
    python experiments/run_gliclass.py --limit 3  # prototype run
"""
import argparse
import time

from ragexp.data import load_nfcorpus
from ragexp.embed import Embedder
from ragexp.rerank import GliclassReranker, sweep
from ragexp.retrieve import dense

POOL_SIZE = 50
SWEEP_NAME = "gliclass_scores"


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

    reranker = GliclassReranker()

    def score_fn(query_text, docs):
        scores = reranker.score(query_text, [t for _, t in docs])
        return {doc_id: s for (doc_id, _), s in zip(docs, scores)}

    start = time.perf_counter()
    done = sweep(SWEEP_NAME, pool, query_texts, doc_texts, score_fn)
    print(f"done: {len(done)} queries scored in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
