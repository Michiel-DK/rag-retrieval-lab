# LESSONS.md — every experiment: what / number / verdict / lesson

The closing artifact for all seven notebooks, and all seven ran. Except where named below, every
number is **recomputed from the committed per-query score files**, not read off a notebook cell.
The source is `results/scores/*.json`: n = 323 NFCorpus queries, scored against the
dataset's human qrels. The notebook 06 rows are one exception: they use MSLR-WEB10K Fold 1
test queries (n = 1,928 with a relevant document), from `results/scores/ltr_*.json`.

Three notebook 05 ledger rows are quoted from notebook output, not recomputed from score
files: the encoder-truncation share (78.8%), the truncation-tercile split and the adapter
train-vs-eval row. Each needs the encoder or the training run.

## Predictions → results

| # | Notebook | Prediction (as pre-registered) | Result |
|---|---|---|---|
| 01 | `the-ruler` | *No prediction; builds the instrument* | ✅ Both calibration demos land: zero-poisoning understates the mean ~40%; paired bootstrap finds p ≈ 0 where independent CIs overlap |
| 02 | `dense-baseline` | *No prediction; sets the bar* | ✅ nDCG@10 **0.3167** · recall@10 0.1550 · recall@50 0.2508; per-query scores committed |
| 03 | `lexical-and-fusion` | **Genuinely unsure**: BM25 might be strong on bio-medical terminology; RRF widely reported to help on BEIR | ❌ Both mechanisms failed: BM25 loses to dense everywhere (nDCG@10 0.2678, −0.049, p ≈ 0); RRF vs dense is a null (0.3093, every CI straddles zero) |
| 04 | `reranking` | **nDCG rises, recall@50 does not move**: a reranker reorders a fixed pool, it cannot change what's in it | ✅ **Held, both halves, both rerankers**: nDCG@10 0.3167 → **0.4052** (LLM judge, p ≈ 0) / **0.3881** (zerank-1-small); recall@50 **frozen bit-for-bit at 0.2508** for all three systems |
| 05 | `pooling-vs-summary` | **Pooling helps** (the body carries signal the title doesn't). **Adapter nulls**: same encoder, nothing to align | ✅ **Split**: pooling helps held against title-only (nDCG@10 +0.0255, p=0.0002) and failed against `dense` (title+text): recall@10 -0.0013, recall@50 -0.0053, nDCG@10 -0.0028, all three null over 323 queries. Adapter nulls held on the 107 held-out queries: nDCG@10 -0.0063 (p=0.28), recall@10 -0.0031 (p=0.39), with recall@50 -0.0245 (p=0.0112) a significant loss |
| 06 | `learning-to-rank` | **LambdaMART beats the best single feature** on Fold 1 test nDCG@10 (paired-bootstrap CI excludes 0). Refined on 2026-10-29, before running, from the first wording "LambdaMART wins", which named no baseline. A published null exists ([Elsevier](https://github.com/elsevierlabs-os/build-ltr-models-using-llm)) but it used LLM-generated labels; MSLR has no text to relabel, so that question stays a caveat | ✅ **Held**: test nDCG@10 0.327 (best feature, picked on vali) → **0.492** (LambdaMART), +0.165, CI [+0.155, +0.175], p ≈ 0, 1,928 defined test queries |
| 07 | `contextual-retrieval` | **The LLM context line beats plain chunks, but not the title**: arm 3 vs arm 1 CI above 0, arm 3 vs arm 2 CI includes 0 | ❌ **First half failed, second held**: arm 3 vs arm 1 nDCG@10 +0.0051, CI [−0.0011, +0.0117]; vs arm 2 +0.0025, CI includes 0. All three chunk arms tie title+text dense |

## The ledger

| what | number | verdict | lesson |
|---|---|---|---|
| zero-poisoning undefined queries (40/100 undefined in the demo) | mean understated ~40% | silent bug | undefined is `None`, never `0.0`; drop, don't zero |
| paired vs independent bootstrap, same data, true Δ = +0.03 | independent CIs overlap; paired p ≈ 0 | instrument choice changes the verdict | compare systems with the **paired** bootstrap; CI-overlap eyeballing is a weaker test |
| dense baseline (MiniLM-L6, title+text) | nDCG@10 0.3167 · recall@10 0.1550 · recall@50 0.2508 | the bar is set | a baseline is only useful if its **per-query** scores are saved: the mean alone can't be paired against |
| BM25 vs dense | nDCG@10 −0.049 · recall@50 −0.072, both p ≈ 0 | lexical loses | the loss sits in long queries: on the 103 single-term queries BM25 ties dense (+0.0008, p = 0.95), on the 96 queries of 5+ words it loses −0.104 (post-hoc split). The first post-mortem blamed "natural-language queries" without checking; the median query is 2 words. Check a post-mortem against the data before it becomes a lesson |
| RRF (dense + BM25) vs dense | null on all three metrics (every CI straddles zero) | fusion is not a free lunch | when one retriever is clearly stronger, rank-averaging hands back roughly the stronger one; RRF only beats the **weaker** parent (+0.042 vs BM25), the trivial direction. "Helps on BEIR" did not survive contact with this corpus + this embedder |
| fusion depth | depth=1000 indistinguishable from full rankings; depth=100 clips recall@50 (−0.007, p = 0.03) | convention adopted | fuse at `depth=1000`: production-shaped, measurably lossless at the ks we score |
| LLM-judge rerank vs dense | nDCG@10 +0.0885, CI [+0.071, +0.106], p ≈ 0; recall@10 +0.032 | first method to beat the baseline | reranking finds gains where BM25 and RRF couldn't; the pool was fine, the *order* was the problem |
| zerank-1-small (1.7B cross-encoder) vs dense | nDCG@10 +0.0714, CI [+0.055, +0.089], p ≈ 0 | most of the LLM's gain with no per-query API cost (runs locally on a laptop; tens of seconds per query) | an off-the-shelf cross-encoder captures ~80% of the LLM-judge lift; the LLM's edge is real but pay-per-query |
| GLiClass modern-base v3.0 (zero-shot classifier, query as the label) vs dense | nDCG@10 −0.0386, CI [−0.0601, −0.0177], p = 0.0004; vs zerank −0.1100 | predicted: beats dense, loses to zerank. Half failed | a reranker can make a pool worse. GLiClass is far above a random shuffle (0.114), so it carries signal, just less than the dense order it replaced. A classifier's "does this label apply" probability is not a relevance score; check a reranker against the order it starts from, not against zero |
| recall@50 under both rerankers | per-query max |Δ| = **exactly 0.0** | structural invariant confirmed | a permutation of a 50-doc pool cannot change what's in the pool; if this metric ever moves, you've found a bug, not an improvement |
| LLM rerank scored against the judge's **own** grades | nDCG@10 = 1.0000, a mean over 317 of the 323 queries (vs 0.4052 against human labels, over all 323) | a perfect score containing zero information | when the labeler and the reranker share a model, the metric is self-agreement, not quality; **anchor evaluation in a measurement the evaluated system cannot influence** |
| `title-only` vs `dense` (title+text) (nb 05) | nDCG@10 -0.0282 (p=0.0016), recall@10 -0.0195 (p=0.0276), recall@50 -0.0130 (p=0.0954) | prediction held on nDCG@10 and recall@10, recall@50 null | embedding a summary in place of the document costs ranking quality first and recall later |
| `chunk-pool` vs `title-only` (nb 05) | nDCG@10 +0.0255 (p=0.0002), recall@10 +0.0181 (p=0.0138), recall@50 +0.0077 (null) | prediction held | mean-pooled body chunks recover most of what the title drops |
| `chunk-pool` vs `dense` (title+text) (nb 05) | recall@10 -0.0013, recall@50 -0.0053, nDCG@10 -0.0028, all three null | prediction failed across the full 323 queries | pooling buys nothing on average when the baseline string already contains the body: the source system's lift came from its firm vector holding no portfolio text |
| same comparison, queries split by encoder truncation (nb 05) | nDCG@10 -0.0261 (p=0.0100) in the least-clipped tercile, +0.0265 (p=0.0058) in the most-clipped | two effects cancelling | a null over a heterogeneous query set is the sum of its subgroups, so split before concluding a mechanism is absent |
| encoder truncation (nb 05) | 78.8% of docs exceed `max_seq_length=256` in `d.full` | measured, and it matters unevenly | four documents in five lose their tail before the model reads them, which changes the ranking only for queries whose relevant documents are the long ones |
| InfoNCE adapter vs `W=I`, 107 held-out queries (nb 05) | recall@10 -0.0031 (p=0.39), nDCG@10 -0.0063 (p=0.28), recall@50 -0.0245 (p=0.0112) | prediction held | a linear adapter fixes cross-space misalignment, and one encoder for both sides leaves none to fix |
| adapter train vs eval (nb 05) | nDCG@10 +0.0072 on 215 train queries, -0.0063 on 107 held-out | overfit, as expected | 147,456 parameters over 215 queries memorise the split; only the held-out number is evidence |
| LambdaMART vs best single feature (MSLR-WEB10K Fold 1, nb 06) | nDCG@10 0.327 → 0.492, +0.165, CI [+0.155, +0.175], p ≈ 0 | learned ranker wins, as predicted | on features designed to be combined, a listwise ranker beats the strongest single signal by a wide margin; pick the baseline feature on vali, never on test |
| Haiku context line per chunk vs plain chunks (nb 07) | nDCG@10 +0.0051, CI [−0.0011, +0.0117], p = 0.11; recall@10 and recall@50 null | predicted to win, did not | 18,835 LLM-written lines bought nothing measurable on short abstracts. Context restores what a chunk lost from its document, and a ~1.7k-char abstract in five chunks loses little |
| title prefixed to every chunk vs plain chunks (nb 07) | nDCG@10 +0.0026, CI [−0.0070, +0.0120], p = 0.60 | the free control nulled too | the pooled vector already blends in the title vector, so a per-chunk title repeats it |
| every chunk arm vs `dense` (title+text) (nb 07) | nDCG@10 −0.0001 (title arm), +0.0024 (context arm), both p > 0.7 | null | on short documents, embedding the whole document once is as good as any chunking scheme tested here |

## The one-line version

Reranking was the only intervention that beat the dense baseline. The gain is real
against human labels and worthless against the judge's own. The metric that could not
move did not move, to the last bit.

That last fact is the repo's spine. Pick the anchored metric before optimizing, and
treat any movement in it as a bug report.
