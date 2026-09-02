# Predict first, then run. Nulls included.

> **Frozen write-up, 2026-08-24.** The repo keeps moving; this page doesn't. Every number
> below traces to a committed file (named inline) — none of it is read off a notebook
> cell, and the recomputation source is the per-query score files. Where a widely-taught
> technique failed here, the failure is the content, not an embarrassment to edit out.

## The one-paragraph version

Four notebooks run the standard RAG retrieval playbook — dense baseline, BM25, RRF
fusion, reranking — on [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) (323
queries, human relevance labels), with **the prediction and its mechanism written down
before each experiment runs**. The scoring instrument is built and calibrated *first*
(notebook 01), every system's per-query scores are committed (`results/scores/*.json`),
and the closing ledger (`LESSONS.md`) recomputes every headline number from those files
rather than trusting the notebook that printed them. Two of the three celebrated
techniques did nothing or worse here, and the repo reports that with the same confidence
intervals as the one that worked.

## The scoreboard

From `LESSONS.md`, all deltas vs the dense baseline (MiniLM-L6), paired bootstrap over
323 queries:

| intervention | nDCG@10 | verdict |
|---|---|---|
| dense baseline (nb 02) | 0.3167 | the bar — per-query scores committed so later systems can be *paired* against it |
| BM25 (nb 03) | 0.2678 (−0.049, p ≈ 0) | **loses everywhere** — "lexical is strong on domain terminology" did not survive contact |
| RRF fusion (nb 03) | 0.3093 (every CI straddles zero) | **null** — fusion only beat its *weaker* parent, the trivial direction |
| LLM-judge rerank (nb 04) | **0.4052** (+0.0885, CI +0.072 – +0.106) | the only intervention that beat the baseline |
| zerank-1-small, 1.7B (nb 04) | 0.3881 (+0.0714) | ~80% of the LLM's lift with no per-query API cost (runs locally on a laptop; tens of seconds per query) |

## Four receipts

### 1. The ruler was calibrated before anything was measured

Notebook 01 builds the instrument and demonstrates two ways it can lie, on purpose:
scoring undefined queries as `0.0` instead of dropping them understated a demo mean by
~40% (undefined is `None`, never zero), and on the same data with a true Δ of +0.03, the
independent-bootstrap CIs overlap while the **paired** bootstrap says p ≈ 0 — the choice
of statistical test changes the verdict, so the repo standardizes on paired before any
system comparison happens. Every later claim inherits that calibration.

### 2. A prediction that held to the last bit

Notebook 04's pre-registered prediction: *nDCG rises, recall@50 does not move* — a
reranker permutes a fixed 50-doc pool; it cannot change what's in the pool. Result: nDCG
moved (+0.0885, p ≈ 0) and recall@50 stayed **bit-for-bit identical at 0.2508 across all
three systems** — per-query max |Δ| exactly 0.0. The invariant is now the repo's built-in
bug detector: if that metric ever moves under a reranker, something is broken, not
improved.

### 3. The perfect score containing zero information

Scoring the LLM-reranked ranking against the *LLM judge's own* relevance grades yields
**nDCG@10 = 1.0000** — against the human labels, the same ranking scores 0.4052. Both
numbers are in `LESSONS.md`, side by side, because the pair is the lesson: when the
labeler and the reranker share a model, the metric measures self-agreement, not quality.
Anchor evaluation in a measurement the evaluated system cannot influence.

### 4. Folklore, tested: "RRF helps on BEIR-type data" — not here

The notebook 03 prediction was registered as **genuinely unsure**, with the mechanism
arguments for both directions written down (BM25 should like bio-medical terminology;
the source system this repo generalizes had seen RRF *hurt*). The run resolved it: BM25
loses to dense on every metric (recall@50 −0.072, p ≈ 0 — NFCorpus queries are
natural-language questions, not keyword queries), and RRF hands back roughly the stronger
parent (null on all three metrics). A secondary measurement made the fusion-depth choice
empirical rather than conventional: depth=1000 is indistinguishable from full rankings;
depth=100 measurably clips recall@50 (−0.007, p = 0.03).

## What is honestly open

- **Notebooks 05–06 have not run.** Notebook 05's prediction is pre-registered and
  falsifiable — an adapter trained between query and document space should be a **null**
  here, because both are embedded by the same model and there is no cross-space
  misalignment to fix. It is listed as open, not assumed.
- **One corpus, one embedder.** Every verdict above is a fact about NFCorpus × MiniLM-L6
  with human qrels — the README says explicitly that the source production system's
  results did *not* transfer wholesale, and that the difference is the lesson. No claim
  here should be quoted without that scope.
- **The LLM judge's gain costs per-query inference.** The ledger prices the trade: the
  1.7B cross-encoder keeps ~80% of the lift with no API cost (local inference, tens of seconds per query).

## Why this transfers

Nothing above depends on NFCorpus. The portable part is the order of operations:
calibrate the ruler before measuring; write the prediction and its mechanism before the
run; commit per-query scores so comparisons can be paired; pick one metric the
intervention *cannot* legitimately move and treat any movement in it as a bug report;
and never score a system against labels it produced. The notebooks are the demonstration
that this ordering is cheap — four experiments, one laptop, and the nulls cost nothing
but honesty.

---

*Numbers: `LESSONS.md` (recomputed from `results/scores/*.json`). Method and scope:
`README.md`. Per-experiment detail: `notebooks/01–04`.*
