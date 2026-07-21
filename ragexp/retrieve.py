"""Retrieval — rank documents for a query. One function per method.

Each returns a ranked list of doc_ids (best first). They share a signature so
notebooks can swap them and re-score with the same ruler:

    ranked_ids = method(query, corpus, ...)

Methods:
- ``dense``      — cosine similarity over embeddings (the baseline)
- ``bm25``       — lexical / term-overlap
- ``rrf``        — Reciprocal Rank Fusion of two rankings (nb 03)
- ``pool_doc``   — title-only vs title+pooled-chunks embedding (nb 05)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .data import Corpus, Doc
from .embed import Embedder


# ── Dense ─────────────────────────────────────────────────────────────────


def dense(
    query: str,
    doc_ids: Sequence[str],
    doc_matrix: np.ndarray,
    embedder: Embedder,
) -> List[str]:
    """Rank doc_ids by cosine similarity to the query.

    ``doc_matrix`` is the (n, dim) unit-normalized doc embeddings aligned to
    ``doc_ids``. Query is embedded fresh; because both are normalized, the dot
    product is cosine. Pre-embedding the docs is what makes this fast enough to
    re-run per notebook.
    """
    q = embedder.encode([query])[0]
    scores = doc_matrix @ q
    order = np.argsort(-scores)
    return [doc_ids[i] for i in order]


# ── BM25 ──────────────────────────────────────────────────────────────────


def build_bm25(docs: Sequence[Doc]):
    """Build a BM25 index over ``docs`` (title + body). Returns (index, ids)."""
    from rank_bm25 import BM25Okapi

    tokenized = [d.full.lower().split() for d in docs]
    return BM25Okapi(tokenized), [d.doc_id for d in docs]


def bm25(query: str, index, doc_ids: Sequence[str]) -> List[str]:
    """Rank doc_ids by BM25 score against the query."""
    scores = index.get_scores(query.lower().split())
    order = np.argsort(-scores)
    return [doc_ids[i] for i in order]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    """Fuse multiple ranked lists by Reciprocal Rank Fusion.

    Each doc scores sum(1 / (k + rank)) across the lists it appears in
    (rank 1-indexed). k=60 is the standard constant. This is the method
    nb 03 interrogates — it's a *rank* fusion, so it discards the score
    magnitudes each retriever produced, which is precisely the property
    under suspicion.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


# ── Chunk pooling (nb 05) ─────────────────────────────────────────────────


def _chunks(text: str, size: int = 400, overlap: int = 80) -> List[str]:
    """Character-window chunks. Crude on purpose — nb 05 is about the pooling
    idea, not chunk-boundary tuning."""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def pool_doc_matrix(
    docs: Sequence[Doc],
    embedder: Embedder,
    mode: str = "title",
) -> np.ndarray:
    """Build the (n, dim) doc matrix under different representations.

    - ``title``  — embed the title only (the sparse-signal baseline)
    - ``full``   — embed title + body as one string
    - ``pool``   — embed body in chunks, mean-pool them, blend with the title

    nb 05's question: does pooling the body recover signal the title alone
    misses? ``pool`` is the analogue of the source system's portfolio-pooled
    firm vector.
    """
    if mode == "title":
        return embedder.encode([d.title or d.text[:80] for d in docs])
    if mode == "full":
        return embedder.encode([d.full for d in docs])
    if mode == "pool":
        title_vecs = embedder.encode([d.title or d.text[:80] for d in docs])
        pooled = np.zeros_like(title_vecs)
        for i, d in enumerate(docs):
            ch = _chunks(d.text)
            cv = embedder.encode(ch)
            pooled[i] = cv.mean(axis=0)
        blend = 0.25 * title_vecs + 0.75 * pooled
        norms = np.linalg.norm(blend, axis=1, keepdims=True)
        return blend / np.clip(norms, 1e-8, None)
    raise ValueError(f"unknown mode {mode!r}")
