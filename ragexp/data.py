"""NFCorpus loader — the dataset the whole lab runs on.

Why NFCorpus (BEIR's bio-medical retrieval set):

- **Small.** 3,633 documents. The whole corpus embeds in under a minute on a
  laptop, so every notebook re-runs from scratch instead of asking you to
  trust a cached number.
- **Graded, not binary.** 12,334 test judgments at levels 1 and 2 (plus
  implicit 0). Most BEIR sets are binary — SciFact, for example, has 339
  qrels all scored 1, which makes nDCG degenerate and learning-to-rank
  impossible. Graded labels are what let notebooks 04 and 06 exist.
- **Densely judged.** ~38 judged docs per query across 323 queries. Sparse
  judgment sets (~1 relevant doc/query) can't distinguish "not retrieved"
  from "not judged".
- **Docs have structure.** Every doc is ``title`` + ``text`` (a short title
  and a ~1.7k-char abstract). Notebook 05 needs that: title-embedding vs
  chunk-pooling is the same question as summary-vs-body in a real RAG stack.

The relevance scale is 1 and 2 only — a doc that isn't judged is implicitly
0. That sparse convention matters: ``grades.get(doc_id, 0)`` is correct,
``grades[doc_id]`` is a KeyError waiting to happen.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from datasets import load_dataset

_HF_NAME = "BeIR/nfcorpus"


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str

    @property
    def full(self) -> str:
        """Title + body — the naive "just embed the whole thing" baseline."""
        return f"{self.title}\n\n{self.text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Corpus:
    docs: List[Doc]
    queries: List[Query]
    qrels: Dict[str, Dict[str, int]]  # query_id -> {doc_id: grade}

    @property
    def doc_ids(self) -> List[str]:
        return [d.doc_id for d in self.docs]

    def relevant_ids(self, query_id: str, min_grade: int = 1) -> List[str]:
        """Doc ids graded >= min_grade for this query."""
        return [
            d for d, g in self.qrels.get(query_id, {}).items() if g >= min_grade
        ]

    def summary(self) -> str:
        n_judged = sum(len(v) for v in self.qrels.values())
        per_q = n_judged / len(self.queries) if self.queries else 0
        grades: Dict[int, int] = {}
        for v in self.qrels.values():
            for g in v.values():
                grades[g] = grades.get(g, 0) + 1
        return (
            f"NFCorpus: {len(self.docs)} docs, {len(self.queries)} queries, "
            f"{n_judged} judgments ({per_q:.1f}/query), "
            f"grade distribution {dict(sorted(grades.items()))}"
        )


@lru_cache(maxsize=1)
def load_nfcorpus(split: str = "test") -> Corpus:
    """Load NFCorpus, keeping only queries that actually have judgments.

    The ``queries`` split ships 3,237 queries but the ``test`` qrels only
    cover ~323 of them. Scoring the unjudged ones would produce a pile of
    undefined metrics — see ``metrics.recall_at_k`` returning None. We drop
    them here rather than downstream, so the query count you see is the
    query count you score.
    """
    corpus_rows = load_dataset(_HF_NAME, "corpus", split="corpus")
    query_rows = load_dataset(_HF_NAME, "queries", split="queries")
    qrel_rows = load_dataset(f"{_HF_NAME}-qrels", split=split)

    qrels: Dict[str, Dict[str, int]] = {}
    for r in qrel_rows:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])

    docs = [
        Doc(doc_id=str(r["_id"]), title=r["title"] or "", text=r["text"] or "")
        for r in corpus_rows
    ]
    queries = [
        Query(query_id=str(r["_id"]), text=(r["text"] or r["title"] or ""))
        for r in query_rows
        if str(r["_id"]) in qrels
    ]

    return Corpus(docs=docs, queries=queries, qrels=qrels)
