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


def maxsim_rank(query_vecs: "np.ndarray", doc_vecs: "np.ndarray", offsets: "np.ndarray", doc_ids: List[str]) -> List[str]:
    """Exhaustive late-interaction ranking over the whole corpus.

    ``doc_vecs`` is every document's token vectors stacked into one (n_tokens, dim)
    matrix; doc ``i`` owns rows ``offsets[i]:offsets[i+1]``. Score = sum over query
    vectors of the max dot product with that doc's vectors (plain MaxSim, fp32).
    Ties break on corpus order (stable sort), so the ranking is deterministic.
    """
    import numpy as np

    sims = query_vecs.astype(np.float32) @ doc_vecs.astype(np.float32).T  # (n_q_vecs, n_tokens)
    per_doc = np.maximum.reduceat(sims, offsets[:-1], axis=1)  # (n_q_vecs, n_docs)
    scores = per_doc.sum(axis=0)
    order = np.argsort(-scores, kind="stable")
    return [doc_ids[i] for i in order]


@dataclass
class Uibk:
    """UIBK-Embedding-v1 (arXiv 2610.27195) as the first stage alone, loaded through PyLate.

    Title + body for every doc, ``prompt_name`` "document"/"query" as the card requires,
    then exhaustive MaxSim over the whole corpus (no PLAID index). The model's 3 dense
    slot vectors sit in the same list, so plain MaxSim is the card's fused score.
    ``revision`` pins the HF commit whose remote code (trust_remote_code) was read before
    the run. CPU only; ``threads`` and ``batch_size`` are fields because they fix the numerics.
    Needs the ``uibk`` extra (pylate), in its own venv: pylate pins older torch/transformers.
    """

    model: str = "DataScience-UIBK/uibk-embedding-v1"
    revision: str = "b18df82e77306006a1dd0162199f6dfeacb5492e"
    batch_size: int = 16
    threads: int = 2
    name: str = "uibk-first-stage"

    def params(self) -> Dict[str, object]:
        return {"model": self.model, "revision": self.revision, "batch_size": self.batch_size,
                "threads": self.threads, "device": "cpu", "scoring": "exhaustive MaxSim"}

    def _load(self):
        import torch

        try:
            from pylate import models
        except ImportError:
            raise ValueError("uibk-first-stage needs pylate: pip install -e '.[uibk]' in its own venv (see pyproject)") from None

        torch.set_num_threads(self.threads)
        return models.ColBERT(self.model, revision=self.revision, trust_remote_code=True, device="cpu")

    def rank(self, corpus: Corpus) -> Rankings:
        import numpy as np

        model = self._load()
        docs = model.encode([d.full for d in corpus.docs], is_query=False, prompt_name="document",
                            batch_size=self.batch_size, show_progress_bar=True)
        lengths = np.array([len(e) for e in docs])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        doc_vecs = np.concatenate([np.asarray(e, dtype=np.float32) for e in docs])
        del docs
        queries = model.encode([q.text for q in corpus.queries], is_query=True, prompt_name="query",
                               batch_size=self.batch_size)
        ids = corpus.doc_ids
        return {q.query_id: maxsim_rank(np.asarray(qv, dtype=np.float32), doc_vecs, offsets, ids)
                for q, qv in zip(corpus.queries, queries)}


REGISTRY: Dict[str, Callable[..., Method]] = {"bm25": Bm25, "dense": Dense, "rrf": Rrf, "uibk-first-stage": Uibk}


def get_method(name: str, **params) -> Method:
    """Build a registered method. Unknown names list what exists."""
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; registered: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name](**params)


def describe(method: Method) -> Dict[str, object]:
    return {"name": method.name, "class": type(method).__name__, "params": method.params()}


__all__ = ["Method", "Rankings", "Bm25", "Dense", "Rrf", "Uibk", "maxsim_rank", "REGISTRY", "get_method", "describe"]
