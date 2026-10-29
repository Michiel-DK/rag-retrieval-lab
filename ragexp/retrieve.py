"""Retrieval — rank documents for a query. One function per method.

Each returns a ranked list of doc_ids (best first). They share a signature so
notebooks can swap them and re-score with the same ruler:

    ranked_ids = method(query, corpus, ...)

Methods:
- ``dense``      — cosine similarity over embeddings (the baseline)
- ``bm25``       — lexical / term-overlap
- ``rrf``        — Reciprocal Rank Fusion of two rankings (nb 03)
- ``pool_doc``   — title-only vs title+pooled-chunks embedding (nb 05); with
  per-chunk prefixes, the contextual-retrieval arms (nb 07)
- ``chunk_matrix`` + ``best_chunk_scores`` — every chunk ranked on its own, a
  doc scoring as its best chunk (nb 07's chunk-level arm)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    depth: Optional[int] = None,
) -> List[str]:
    """Fuse multiple ranked lists by Reciprocal Rank Fusion.

    Each doc scores sum(1 / (k + rank)) across the lists it appears in
    (rank 1-indexed). k=60 is the standard constant. This is the method
    nb 03 interrogates — it's a *rank* fusion, so it discards the score
    magnitudes each retriever produced, which is precisely the property
    under suspicion.

    ``depth`` truncates each input list before fusing. ``None`` fuses the
    full rankings (the original RRF formulation); production systems fuse
    top-k candidate lists instead, where a doc outside one retriever's
    depth gets zero contribution from it. The two differ near the tail —
    nb 03 measures by how much rather than assuming.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        if depth is not None:
            ranking = ranking[:depth]
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
    chunk_prefixes: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """Build the (n, dim) doc matrix under different representations.

    - ``title``  — embed the title only (the sparse-signal baseline)
    - ``full``   — embed title + body as one string
    - ``pool``   — embed body in chunks, mean-pool them, blend with the title

    nb 05's question: does pooling the body recover signal the title alone
    misses? ``pool`` is the analogue of the source system's portfolio-pooled
    firm vector.

    ``chunk_prefixes`` (``pool`` only, nb 07) maps a doc id to one prefix per
    chunk; each chunk is embedded as ``prefix + "\n" + chunk``. An empty prefix,
    or a doc missing from the map, leaves its chunks as they are.
    """
    if mode == "title":
        return embedder.encode([d.title or d.text[:80] for d in docs])
    if mode == "full":
        return embedder.encode([d.full for d in docs])
    if mode == "pool":
        title_vecs = embedder.encode([d.title or d.text[:80] for d in docs])
        # One encode call over every doc's chunks, then split back per doc:
        # per-doc calls wrote one cache file per document.
        chunk_vecs, bounds = _encode_chunks(docs, embedder, chunk_prefixes)
        pooled = np.zeros_like(title_vecs)
        for i in range(len(docs)):
            pooled[i] = chunk_vecs[bounds[i] : bounds[i + 1]].mean(axis=0)
        blend = 0.25 * title_vecs + 0.75 * pooled
        norms = np.linalg.norm(blend, axis=1, keepdims=True)
        return blend / np.clip(norms, 1e-8, None)
    raise ValueError(f"unknown mode {mode!r}")


def chunk_matrix(
    docs: Sequence[Doc],
    embedder: Embedder,
    chunk_prefixes: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Every chunk as its own row, plus ``bounds``: doc ``i`` owns rows
    ``bounds[i]:bounds[i + 1]``. Same chunks, prefixes and encode call as
    ``pool_doc_matrix(mode="pool")``, so the embedding cache is shared."""
    return _encode_chunks(docs, embedder, chunk_prefixes)


def best_chunk_scores(query_vecs: np.ndarray, chunk_vecs: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """(n_queries, n_docs): each doc scores as the cosine of its best chunk."""
    return np.maximum.reduceat(query_vecs @ chunk_vecs.T, bounds[:-1], axis=1)


def _encode_chunks(docs, embedder, chunk_prefixes):
    per_doc = [_chunks(d.text) for d in docs]
    if chunk_prefixes is not None:
        per_doc = [_prefixed(ch, chunk_prefixes.get(d.doc_id)) for d, ch in zip(docs, per_doc)]
    bounds = np.cumsum([0] + [len(ch) for ch in per_doc])
    return embedder.encode([c for ch in per_doc for c in ch]), bounds


def _prefixed(chunks: List[str], prefixes: Optional[List[str]]) -> List[str]:
    if prefixes is None:
        return chunks
    if len(prefixes) != len(chunks):
        raise ValueError(f"{len(prefixes)} prefixes for {len(chunks)} chunks")
    return [f"{p}\n{c}" if p else c for p, c in zip(prefixes, chunks)]
