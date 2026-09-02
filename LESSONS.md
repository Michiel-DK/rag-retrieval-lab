# LESSONS.md — every experiment: what / number / verdict / lesson

The closing artifact for notebooks 01–04 (the stop-and-ship set). Every number below is
**recomputed from the committed per-query score files** (`results/scores/*.json`, n = 323
NFCorpus queries, scored against the dataset's human qrels) — not read off a notebook cell.
Format follows `docs/technical/RETRIEVAL_LEDGER.md`. Notebooks 05–06 are optional depth;
their rows get added if they run.

## Predictions → results

| # | Notebook | Prediction (as pre-registered) | Result |
|---|---|---|---|
| 01 | `the-ruler` | *No prediction — builds the instrument* | ✅ Both calibration demos land: zero-poisoning understates the mean ~40%; paired bootstrap finds p ≈ 0 where independent CIs overlap |
| 02 | `dense-baseline` | *No prediction — sets the bar* | ✅ nDCG@10 **0.3167** · recall@10 0.1550 · recall@50 0.2508; per-query scores committed |
| 03 | `lexical-and-fusion` | **Genuinely unsure** — BM25 might be strong on bio-medical terminology; RRF widely reported to help on BEIR | ❌ Both mechanisms failed: BM25 loses to dense everywhere (nDCG@10 0.2678, −0.049, p ≈ 0); RRF vs dense is a null (0.3093, every CI straddles zero) |
| 04 | `reranking` | **nDCG rises, recall@50 does not move** — a reranker reorders a fixed pool, it cannot change what's in it | ✅ **Held, both halves, both rerankers**: nDCG@10 0.3167 → **0.4052** (LLM judge, p ≈ 0) / **0.3881** (zerank-1-small); recall@50 **frozen bit-for-bit at 0.2508** for all three systems |

## The ledger

| what | number | verdict | lesson |
|---|---|---|---|
| zero-poisoning undefined queries (40/100 undefined in the demo) | mean understated ~40% | silent bug | undefined is `None`, never `0.0` — drop, don't zero |
| paired vs independent bootstrap, same data, true Δ = +0.03 | independent CIs overlap; paired p ≈ 0 | instrument choice changes the verdict | compare systems with the **paired** bootstrap; CI-overlap eyeballing is a weaker test |
| dense baseline (MiniLM-L6, title+text) | nDCG@10 0.3167 · recall@10 0.1550 · recall@50 0.2508 | the bar is set | a baseline is only useful if its **per-query** scores are saved — the mean alone can't be paired against |
| BM25 vs dense | nDCG@10 −0.049 · recall@50 −0.072, both p ≈ 0 | lexical loses | NFCorpus queries are natural-language questions, not keyword queries — "rare exact terms" didn't materialize; lexical signal is real but strictly weaker than the embedder here |
| RRF (dense + BM25) vs dense | null on all three metrics (every CI straddles zero) | fusion is not a free lunch | when one retriever is clearly stronger, rank-averaging hands back roughly the stronger one; RRF only beats the **weaker** parent (+0.042 vs BM25) — the trivial direction. "Helps on BEIR" did not survive contact with this corpus + this embedder |
| fusion depth | depth=1000 indistinguishable from full rankings; depth=100 clips recall@50 (−0.007, p = 0.03) | convention adopted | fuse at `depth=1000` — production-shaped, measurably lossless at the ks we score |
| LLM-judge rerank vs dense | nDCG@10 +0.0885, CI [+0.072, +0.106], p ≈ 0; recall@10 +0.032 | first method to beat the baseline | reranking finds gains where BM25 and RRF couldn't — the pool was fine, the *order* was the problem |
| zerank-1-small (1.7B cross-encoder) vs dense | nDCG@10 +0.0714, CI [+0.055, +0.089], p ≈ 0 | most of the LLM's gain with no per-query API cost (runs locally on a laptop; tens of seconds per query) | an off-the-shelf cross-encoder captures ~80% of the LLM-judge lift; the LLM's edge is real but pay-per-query |
| recall@50 under both rerankers | per-query max |Δ| = **exactly 0.0** | structural invariant confirmed | a permutation of a 50-doc pool cannot change what's in the pool — if this metric ever moves, you've found a bug, not an improvement |
| LLM rerank scored against the judge's **own** grades | nDCG@10 = 1.0000 (vs 0.4052 against human labels) | a perfect score containing zero information | when the labeler and the reranker share a model, the metric is self-agreement, not quality — **anchor evaluation in a measurement the evaluated system cannot influence** |

## The one-line version

Reranking was the only intervention that beat the dense baseline, the gain is real against
human labels and worthless against the judge's own — and the metric that couldn't move,
didn't, to the last bit. That last fact is the repo's spine: pick your anchored metric
before you optimize, and treat any movement in it as a bug report.
