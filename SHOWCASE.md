# Predict first, then run. Nulls included.

> **Write-up of notebooks 01-04, dated 2026-08-24.** It has been corrected since where the
> 2026-09 audit (`docs/defects-from-radar-2026-09-04.md`) found errors. Notebooks 05 and 06
> are noted in the open section. Notebook 07 and the GLiClass arms of notebook 04 are
> covered in `README.md` and `LESSONS.md`, not here. Every number
> below traces to a committed file, named inline. None of it is read off a notebook cell.
> The recomputation source is the per-query score files. Where a widely-taught technique
> failed here, the failure is the content, not an embarrassment to edit out.

## The one-paragraph version

Four notebooks run the standard RAG retrieval playbook on
[NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus): dense baseline, BM25, RRF
fusion, reranking. The corpus ships 323 queries with human relevance labels. **The
prediction and its mechanism are written down before each experiment runs.**

Notebook 01 builds and calibrates the scoring instrument *first*. Every system's
per-query scores are committed to `results/scores/*.json`. The closing ledger
(`LESSONS.md`) recomputes every headline number from those files rather than trusting
the notebook that printed them.

Two of the three celebrated techniques did nothing or worse here. The repo reports that
with the same confidence intervals as the one that worked.

## The scoreboard

From `LESSONS.md`, all deltas vs the dense baseline (MiniLM-L6), paired bootstrap over
323 queries:

| intervention | nDCG@10 | verdict |
|---|---|---|
| dense baseline (nb 02) | 0.3167 | the bar: per-query scores committed so later systems can be *paired* against it |
| BM25 (nb 03) | 0.2678 (−0.049, p ≈ 0) | **loses everywhere**: "lexical is strong on domain terminology" did not survive contact |
| RRF fusion (nb 03) | 0.3093 (every CI straddles zero) | **null**: fusion only beat its *weaker* parent, the trivial direction |
| LLM-judge rerank (nb 04) | **0.4052** (+0.0885, CI +0.071 – +0.106) | the only intervention that beat the baseline |
| zerank-1-small, 1.7B (nb 04) | 0.3881 (+0.0714) | ~80% of the LLM's lift with no per-query API cost (runs locally on a laptop; tens of seconds per query) |

## Four receipts

### 1. The ruler was calibrated before anything was measured

Notebook 01 builds the instrument and demonstrates two ways it can lie, on purpose.

Scoring undefined queries as `0.0` instead of dropping them understated a demo mean by
~40%. Undefined is `None`, never zero.

On the same data with a true Δ of +0.03, the **paired** bootstrap says p ≈ 0 while the
independent-bootstrap CIs overlap. The choice of statistical test changes the verdict.
The repo therefore standardizes on paired before any system comparison happens, and
every later claim inherits that calibration.

### 2. A prediction that held to the last bit

Notebook 04's pre-registered prediction: *nDCG rises, recall@50 does not move*. A
reranker permutes a fixed 50-doc pool, so it cannot change what's in the pool.

The result: nDCG moved (+0.0885, p ≈ 0) and recall@50 stayed **bit-for-bit identical at
0.2508 across all three systems**, per-query max |Δ| exactly 0.0. The invariant is now
the repo's built-in bug detector. If that metric ever moves under a reranker, something
is broken rather than improved.

### 3. The perfect score containing zero information

Scoring the LLM-reranked ranking against the *LLM judge's own* relevance grades yields
**nDCG@10 = 1.0000** (a mean over 317 of the 323 queries: 6 got all-zero judge grades and score as undefined). Against the human labels, the same ranking scores 0.4052.

Both numbers sit side by side in `LESSONS.md`, because the pair is the lesson. When the
labeler and the reranker share a model, the metric measures self-agreement, not quality.
Anchor evaluation in a measurement the evaluated system cannot influence.

### 4. Folklore, tested: "RRF helps on BEIR-type data" — not here

The notebook 03 prediction was registered as **genuinely unsure**, with the mechanism
arguments for both directions written down. BM25 should like bio-medical terminology.
The source system this repo generalizes had seen RRF *hurt*.

The run resolved it. BM25 loses to dense on every metric (recall@50 −0.072, p ≈ 0).
The loss is not uniform. On the 103 single-term queries BM25 ties dense (+0.0008
nDCG@10, p = 0.95); on the 96 queries of five or more words it loses by 0.104. That
split was run afterwards, so it is exploratory. RRF hands back roughly the stronger
parent, a null on all three metrics.

A secondary measurement made the fusion-depth choice empirical rather than
conventional.

- depth=1000 is indistinguishable from full rankings.
- depth=100 measurably clips recall@50 (−0.007, p = 0.03).

## What is honestly open

- **Notebooks 05 and 06 have run.** In notebook 05, pooling beat title-only (nDCG@10
  +0.0255, p=0.0002) but nulled against title+text over 323 queries. The adapter nulled
  on 107 held-out queries (nDCG@10 -0.0063, p=0.28). In notebook 06, on MSLR-WEB10K
  Fold 1, LambdaMART beat the best single feature: 0.327 to 0.492, +0.165,
  CI [+0.155, +0.175], over 1,928 defined test queries. Two things stay open there. MSLR
  has no text to relabel, so the LLM-label question is untested. The run covers one fold
  with fixed settings.
- **One corpus, one embedder.** Every verdict above is a fact about NFCorpus × MiniLM-L6
  with human qrels. The README says explicitly that the source production system's
  results did *not* transfer wholesale, and that the difference is the lesson. No claim
  here should be quoted without that scope.
- **The LLM judge's gain costs per-query inference.** The ledger prices the trade: the
  1.7B cross-encoder keeps ~80% of the lift with no API cost (local inference, tens of seconds per query).

## Why this transfers

Nothing above depends on NFCorpus. The portable part is the order of operations.

- Calibrate the ruler before measuring.
- Write the prediction and its mechanism before the run.
- Commit per-query scores so comparisons can be paired.
- Pick one metric the intervention *cannot* legitimately move, and treat any movement
  in it as a bug report.
- Never score a system against labels it produced.

The notebooks show that this ordering is cheap: four experiments, one laptop, and the
nulls reported alongside the win.

---

*Numbers: `LESSONS.md` (recomputed from `results/scores/*.json`). Method and scope:
`README.md`. Per-experiment detail: `notebooks/01–04`.*
