"""Reranking — reorder a fixed candidate pool. Two rerankers, one contract.

Ported (in shape) from a production system's listwise 0–3 grader; the LLM
client is swapped for a pluggable one. The contract every reranker obeys:

    it receives a *fixed* candidate pool and returns a permutation of it.

That contract is the whole point of notebook 04: a permutation cannot change
which documents are in the pool, so recall@pool-size is structurally immune
to whatever the reranker believes — including an LLM judge agreeing with
itself.

Rerankers:
- ``ClaudeCLIJudge`` — listwise LLM grader via headless Claude Code
  (``claude -p``), so it runs on a subscription with no API key. Grades are
  checkpointed per query to ``results/rerank/`` — the sweep resumes where it
  stopped, and downstream readers reproduce from the committed file without
  any LLM at all.
- ``ZerankReranker`` — ``zeroentropy/zerank-1-small``, an off-the-shelf
  cross-encoder. Scores checkpointed the same way.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_RERANK_DIR = Path(__file__).resolve().parent.parent / "results" / "rerank"

Pool = Dict[str, List[str]]  # query_id -> candidate doc_ids (ranked)


# ── The contract ──────────────────────────────────────────────────────────


def rerank_by_scores(pool_ids: Sequence[str], scores: Dict[str, float]) -> List[str]:
    """Permute ``pool_ids`` by score, descending; ties keep original order.

    Stable sort on the negated score means a reranker that assigns every doc
    the same score returns the pool unchanged — the identity permutation is
    the degenerate case, not an error.
    """
    return sorted(pool_ids, key=lambda d: -scores.get(d, float("-inf")))


# ── LLM judge (listwise 0–3 grader) ──────────────────────────────────────

_PROMPT = """You are grading search results for relevance.

Query: {query}

Below are {n} documents, numbered 1..{n}. Grade each document's relevance to
the query on this scale:
  3 = directly answers or centrally addresses the query
  2 = substantially relevant (on-topic evidence, partial answer)
  1 = marginally relevant (related topic, tangential)
  0 = not relevant

Documents:
{docs}

Reply with ONLY a JSON object mapping each document number to its grade, like
{{"1": 2, "2": 0, ...}}. Every number 1..{n} must appear. No other text."""


class ClaudeCLIJudge:
    """Listwise grader backed by headless Claude Code (`claude -p`).

    Runs on the local `claude` login (subscription) — no API key. One call
    grades a whole candidate pool for one query.
    """

    def __init__(self, model: str = "opus", timeout: int = 600, max_chars: int = 350):
        # "opus" is an unversioned CLI alias. The committed grades
        # (results/rerank/llm_grades.json, 2026-08-15) came through it; the exact
        # model version was not recorded.
        self.model = model
        self.timeout = timeout
        self.max_chars = max_chars  # per-doc text budget keeps the prompt sane

    def grade(self, query: str, docs: Sequence[Tuple[str, str]]) -> Dict[str, int]:
        """Grade ``docs`` (doc_id, text) against ``query`` -> {doc_id: 0..3}."""
        listing = "\n".join(
            f"{i}. {text[: self.max_chars]}" for i, (_, text) in enumerate(docs, start=1)
        )
        prompt = _PROMPT.format(query=query, n=len(docs), docs=listing)
        raw = self._call(prompt)
        by_index = self._parse(raw, n=len(docs))
        return {doc_id: by_index[i] for i, (doc_id, _) in enumerate(docs, start=1)}

    def _call(self, prompt: str) -> str:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {proc.stderr[:500]}")
        return json.loads(proc.stdout)["result"]

    @staticmethod
    def _parse(raw: str, n: int) -> Dict[int, int]:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object in judge reply: {raw[:200]}")
        parsed = json.loads(m.group(0))
        grades = {int(k): int(v) for k, v in parsed.items()}
        missing = set(range(1, n + 1)) - set(grades)
        if missing:
            raise ValueError(f"judge reply missing indices {sorted(missing)[:5]}...")
        bad = [v for v in grades.values() if not 0 <= v <= 3]
        if bad:
            raise ValueError(f"grades out of range: {bad[:5]}")
        return grades


# ── zerank-1-small (off-the-shelf cross-encoder) ─────────────────────────


class ZerankReranker:
    """`zeroentropy/zerank-1-small` scoring (query, doc) pairs. Lazy-loaded.

    ``batch_size`` and ``max_chars`` keep peak memory inside what Apple MPS
    allows for a 1.7B cross-encoder — a batch of 50 full abstracts OOMs.
    """

    def __init__(
        self,
        model_name: str = "zeroentropy/zerank-1-small",
        batch_size: int = 8,
        max_chars: int = 1200,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, trust_remote_code=True)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        import torch

        model = self._load()
        pairs = [(query, t[: self.max_chars]) for t in texts]
        out: List[float] = []
        # inference_mode matters twice over: the model's custom predict()
        # doesn't disable autograd itself, so without this every forward
        # builds a graph — ~2.6x slower and enough extra memory that macOS
        # kills the process mid-sweep.
        with torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                out.extend(
                    float(s) for s in model.predict(pairs[i : i + self.batch_size])
                )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return out


# ── GLiClass (zero-shot classifier as a reranker) ────────────────────────


class GliclassReranker:
    """`knowledgator/gliclass-modern-base-v3.0` scoring (query, doc) pairs. Lazy-loaded.

    A zero-shot classifier, not a ranker: each doc is the text and the query is
    its only label, so a doc's score is the sigmoid probability that the query
    applies to it. ``max_chars`` is zerank's cut, so both arms see the same input.
    """

    def __init__(
        self,
        model_name: str = "knowledgator/gliclass-modern-base-v3.0",
        batch_size: int = 8,
        max_chars: int = 1200,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._pipeline = ZeroShotClassificationPipeline(
                GLiClassModel.from_pretrained(self.model_name),
                AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True),
                classification_type="multi-label",
                device=device,
                progress_bar=False,
            )
        return self._pipeline

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        pipeline = self._load()
        # threshold 0.0 keeps every label, so each result holds exactly one score
        results = pipeline(
            [t[: self.max_chars] for t in texts],
            [query],
            threshold=0.0,
            batch_size=self.batch_size,
        )
        return [float(r[0]["score"]) for r in results]


# ── Checkpointed sweeps ──────────────────────────────────────────────────


def _load_checkpoint(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(path.read_text()) if path.exists() else {}


def sweep(
    name: str,
    pool: Pool,
    query_texts: Dict[str, str],
    doc_texts: Dict[str, str],
    score_fn: Callable[[str, Sequence[Tuple[str, str]]], Dict[str, float]],
    verbose: bool = True,
    pool_name: str = "dense_pool",
) -> Dict[str, Dict[str, float]]:
    """Run ``score_fn`` over every query's pool, checkpointing per query.

    ``score_fn(query_text, [(doc_id, doc_text), ...]) -> {doc_id: score}``.
    Results persist to ``results/rerank/<name>.json`` after every query, so a
    killed or rate-limited sweep resumes instead of restarting. Already-scored
    queries are never re-run — delete the file to force a fresh sweep.

    The grade file is keyed by doc id, so it forgets the order the pool was
    in, and ``rerank_by_scores`` breaks ties by that order. So each query's
    pool is recorded, in the order given, to ``results/rerank/<pool_name>.json``
    before it is graded. A query already recorded there with a different
    order raises: its committed grades would no longer rebuild the same
    ranking. A query already graded but not recorded also raises, since the
    order its grades were made against is unknown.
    """
    _RERANK_DIR.mkdir(parents=True, exist_ok=True)
    path = _RERANK_DIR / f"{name}.json"
    done = _load_checkpoint(path)
    recorded = load_pool(pool_name)
    for qid, ids in pool.items():
        if qid in recorded and recorded[qid] != list(ids):
            raise ValueError(
                f"pool for {qid} differs from the one recorded in {pool_name}.json"
            )
        if qid in done and qid not in recorded:
            raise ValueError(
                f"{qid} is graded in {name}.json but has no pool in {pool_name}.json:"
                " the order its grades were made against is unknown. Restore"
                f" {pool_name}.json, or delete {qid} from {name}.json to re-grade it"
            )

    todo = [qid for qid in pool if qid not in done]
    for i, qid in enumerate(todo):
        # pool first: a kill between the two writes must not leave grades
        # whose order was never recorded. Rewriting the whole file per query
        # is ~200 KB at full size, milliseconds next to the grading call it
        # precedes, and the grade file is already rewritten the same way.
        recorded[qid] = list(pool[qid])
        save_pool(recorded, pool_name)
        docs = [(d, doc_texts[d]) for d in pool[qid]]
        done[qid] = score_fn(query_texts[qid], docs)
        path.write_text(json.dumps(done, indent=0, sort_keys=True))
        if verbose:
            print(f"[{len(done)}/{len(pool)}] {qid}", flush=True)
    return done


def load_sweep(name: str) -> Dict[str, Dict[str, float]]:
    """Read a committed sweep — the no-LLM reproduction path."""
    return _load_checkpoint(_RERANK_DIR / f"{name}.json")


def save_pool(pool: Pool, name: str = "dense_pool") -> None:
    """Write a candidate pool, each query's doc list in rank order."""
    _RERANK_DIR.mkdir(parents=True, exist_ok=True)
    # sort_keys orders the query ids only; the doc lists keep their order.
    (_RERANK_DIR / f"{name}.json").write_text(json.dumps(pool, indent=0, sort_keys=True))


def load_pool(name: str = "dense_pool") -> Pool:
    """Read a committed pool: the tie-break order ``rerank_by_scores`` needs."""
    path = _RERANK_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}
