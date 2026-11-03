"""The method interface: one object per retrieval method, looked up by name.

A method has a ``name`` (its results/scores/<name>.json), ``params()`` (what it
was run with, recorded later by the run manifest) and ``rank(corpus)`` (query_id
-> ranked doc_ids for every query). Scoring stays in ``runs.evaluate_run``, so a
method cannot change the ruler.

Ported so far: the three first-stage retrievers whose committed scores can be
rebuilt from the corpus and the embedding cache alone (bm25, dense, rrf).
Rerankers, chunking and LTR stay in their notebooks/experiments for now; their
inputs include LLM sweeps that are committed evidence and are not re-run.

The free functions in ``ragexp.retrieve`` remain the implementation; these
classes only fix the call that produced each committed file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from .data import Corpus

Rankings = Dict[str, List[str]]


@runtime_checkable
class Method(Protocol):
    name: str

    def params(self) -> Dict[str, object]: ...

    def rank(self, corpus: Corpus) -> Rankings: ...


@dataclass
class Bm25:
    """nb 03: BM25Okapi over title + body, lowercase whitespace tokens."""

    name: str = "bm25"

    def params(self) -> Dict[str, object]:
        return {}

    def rank(self, corpus: Corpus) -> Rankings:
        from .retrieve import bm25, build_bm25

        index, ids = build_bm25(corpus.docs)
        return {q.query_id: bm25(q.text, index, ids) for q in corpus.queries}


@dataclass
class Dense:
    """nb 02: cosine over unit-normalized embeddings of title + body."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "dense"

    def params(self) -> Dict[str, object]:
        return {"model": self.model}

    def rank(self, corpus: Corpus) -> Rankings:
        from .embed import Embedder
        from .retrieve import dense

        emb = Embedder(self.model)
        doc_matrix = emb.encode([d.full for d in corpus.docs])
        return {q.query_id: dense(q.text, corpus.doc_ids, doc_matrix, emb) for q in corpus.queries}


@dataclass
class Rrf:
    """nb 03: Reciprocal Rank Fusion of dense and bm25. The committed run is depth=1000."""

    k: int = 60
    depth: Optional[int] = 1000
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "rrf"

    def params(self) -> Dict[str, object]:
        return {"k": self.k, "depth": self.depth, "model": self.model, "parts": ["dense", "bm25"]}

    def rank(self, corpus: Corpus) -> Rankings:
        from .retrieve import rrf

        d = Dense(self.model).rank(corpus)
        b = Bm25().rank(corpus)
        return {q.query_id: rrf([d[q.query_id], b[q.query_id]], k=self.k, depth=self.depth) for q in corpus.queries}


REGISTRY: Dict[str, Callable[..., Method]] = {"bm25": Bm25, "dense": Dense, "rrf": Rrf}


def get_method(name: str, **params) -> Method:
    """Build a registered method. Unknown names list what exists."""
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; registered: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name](**params)


def describe(method: Method) -> Dict[str, object]:
    return {"name": method.name, "class": type(method).__name__, "params": method.params()}


__all__ = ["Method", "Rankings", "Bm25", "Dense", "Rrf", "REGISTRY", "get_method", "describe"]
