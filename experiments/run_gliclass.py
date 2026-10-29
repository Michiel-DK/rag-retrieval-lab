"""Produce results/rerank/gliclass_scores.json (v3) or gliclass_v4_scores.json (v4).

Scores the same dense top-50 pool as the LLM judge with a GLiClass zero-shot
classifier (query as the only label). Checkpointed per query.
--arm v3 (default): `knowledgator/gliclass-modern-base-v3.0`, no prompt.
--arm v4: `knowledgator/gliclass-v4-edge` with an instruction prompt.

Usage:
    python experiments/run_gliclass.py            # full v3 sweep
    python experiments/run_gliclass.py --arm v4   # full v4 sweep
    python experiments/run_gliclass.py --limit 3  # prototype run
"""
import argparse
import time

from ragexp.data import load_nfcorpus
from ragexp.embed import Embedder
from ragexp.rerank import GliclassReranker, sweep
from ragexp.retrieve import dense

POOL_SIZE = 50
ARMS = {
    "v3": {
        "model_name": "knowledgator/gliclass-modern-base-v3.0",
        "prompt": None,
        "add_prefix_space": True,
        "sweep_name": "gliclass_scores",
    },
    "v4": {
        "model_name": "knowledgator/gliclass-v4-edge",
        "prompt": "Does the passage answer the query?",
        "add_prefix_space": False,
        "sweep_name": "gliclass_v4_scores",
    },
}


def main(limit: int | None = None, arm: str = "v3") -> None:
    cfg = ARMS[arm]
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

    reranker = GliclassReranker(
        model_name=cfg["model_name"],
        prompt=cfg["prompt"],
        add_prefix_space=cfg["add_prefix_space"],
    )

    def score_fn(query_text, docs):
        scores = reranker.score(query_text, [t for _, t in docs])
        return {doc_id: s for (doc_id, _), s in zip(docs, scores)}

    start = time.perf_counter()
    done = sweep(cfg["sweep_name"], pool, query_texts, doc_texts, score_fn)
    print(f"done: {len(done)} queries scored in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arm", choices=sorted(ARMS), default="v3")
    args = ap.parse_args()
    main(args.limit, args.arm)
