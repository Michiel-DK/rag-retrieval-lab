# Pickup note — Contextual Retrieval, for analysis later (from repo-radar, 2026-09-30)

> **Status: IDEA, not scheduled.** Verdict row: repo-radar `PAPER_VERDICTS.md`, section
> "2026-09-28 · Anthropic, Contextual Retrieval". Link there, don't copy numbers here.

## The technique

[Anthropic, "Contextual Retrieval"](https://www.anthropic.com/engineering/contextual-retrieval)
(2024-09-19): before indexing, an LLM reads the whole document and writes 50–100 tokens situating
each chunk ("this is from X, about Y"). That text is prepended to the chunk for **both** the
embedding and BM25. Reported top-20 retrieval failure (1−recall@20): 5.7% → 3.7% (contextual
embeddings) → 2.9% (+ contextual BM25) → 1.9% (+ reranking the top-150). Their cost figure
($1.02 per million document tokens with prompt caching) is 2024 pricing, so re-price before use.

## Why it doesn't apply to notebooks 02–04

They retrieve NFCorpus documents **whole** (title + ~1.7k-char abstract, `ragexp/data.py`), so no
chunk has lost its document and the context has nothing to add. The lab's own results also
cut against two of its three layers: BM25 **lost** (−0.049 nDCG@10) and RRF fusion was **null**;
only reranking won (+0.0885), which matches the post's biggest single step being the reranker.

## Where it could apply: the chunked notebook

`ragexp/retrieve.py` splits text into 400-char windows (80 overlap) for "chunk pooling (nb 05)".
That's the only place in the lab where a chunk can lose its document.

⚠️ **Resolve first:** `SHOWCASE.md` ("What is honestly open") describes notebook 05's registered
prediction as a query→document **adapter** (predicted null), while `ragexp/retrieve.py` labels
**chunk pooling** as nb 05. Decide which notebook this belongs to before pre-registering anything.

## If picked up — pre-register three arms, cheapest first

1. Chunks as they are (the current pooling baseline).
2. **Title prepended to each chunk (no LLM, $0).** On NFCorpus the missing context *is* mostly the
   title, so this is the arm to beat.
3. LLM-written context per chunk (the post's method).

The paired bootstrap from notebook 01, as always. **Prediction to register:** arm 3 beats arm 1,
but its margin over arm 2 sits inside the CI on short abstracts. The LLM context should only pay on
long multi-section documents. If it pays here anyway, that's the finding.

## Also from the post

Under ~200k tokens (~500 pages), the post says skip retrieval and put the whole knowledge base in
the prompt. Worth a sentence in `LESSONS.md` as the "when not to build RAG at all" boundary.

## 2026-10-21 addendum — retrieval items from the repo-radar feed (verdicts: repo-radar `PAPER_VERDICTS.md` 10-21)

- **CoverSynth (2610.15409), coverage-guided retrieval eval sets.** From an unannotated corpus, target coverage across retrieval demands and system capabilities; deterministic checks locate evidence quotes in sources, model checks gate semantic validity. On RepLiQA **91% of human-written queries are answered by all 12 systems in the panel vs 12% of generated ones**; 12/12 within-harness model contrasts significant on generated sets vs 4/12 on human sets; surfaces underspecified/ambiguous-request failures. For this lab: the eval sets used in the notebooks are human/benchmark sets and may not separate configurations for the same reason. CANDIDATE; reopen when a config comparison comes back "within noise".
- **Simple Linear Corrections for Compression in Late Interaction (2610.14196)** and **Tail-Aware Retrieval Evaluation (2610.17986)**: skipped by title in the triage (multi-vector index compression; image instance retrieval with 100M distractors). Listed so they are not re-found as new.
