"""Produce results/context/haiku_contexts.json — nb 07's LLM context lines.

One headless Claude Code call per NFCorpus document (subscription auth, no API
key). Checkpointed: safe to kill and re-run, resumes where it stopped.

Usage:
    python experiments/run_contexts.py             # all 3,633 docs
    python experiments/run_contexts.py --limit 20  # timing pilot
"""
import argparse
import time

from ragexp.contextualize import ClaudeCLIContextualizer, build_contexts
from ragexp.data import load_nfcorpus

NAME = "haiku_contexts"


def main(limit: int | None = None, workers: int = 8) -> None:
    corpus = load_nfcorpus()
    docs = corpus.docs[:limit] if limit else corpus.docs
    start = time.perf_counter()
    done = build_contexts(NAME, docs, ClaudeCLIContextualizer(model="haiku"), workers=workers)
    failed = sum(1 for d in docs if done.get(d.doc_id) is None)
    print(f"done: {len(docs)} docs, {failed} failed, {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.limit, args.workers)
