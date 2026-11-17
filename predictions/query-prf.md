# query-prf

origin: {"kind": "inbox", "url": "repo-radar:docs/inbox/2026-11-17-coverage-gaps-four-lab-ideas.md", "label": "[retrieval] gap 2 of 4, query rewriting: no query-side run"}

Retrieval lane, written 2026-11-17 by the radar's daily triage (rrd-triage-2026-11-17), before any run. Reader question: **C5** (does the trick everyone talks about hold on office work?), at 8 ⚠ in today's count. decider: operator, whether the model-rewrite arm (HyDE / Query2doc shape, a key or the GPU) gets a go as the next rag lane; flips when the PRF Δ nDCG@10 or Δ recall@50 CI lower bound is above 0 (query-side expansion moves this corpus, so the paid arm is worth one run), or both CIs straddle or sit below 0 (query rewriting goes into LESSONS as a null on NFCorpus and the paid arm stays parked).

## Prediction
Metric: nDCG@10 (primary), recall@50 and recall@300, paired bootstrap Δ against `dense` (seed 42, 10,000 resamples), on the 323 NFCorpus test queries in `results/query_order.json`, human qrels. Baseline: the committed dense run (MiniLM-L6, title+text; nDCG@10 0.3167, recall@50 0.2508, recall@300 0.4242, LESSONS rows 02 and pool-depth). Method `dense-prf`: q' = (q + d1 + d2 + d3) / 4 over the unit-normalised vectors of the query's top-3 dense hits, re-normalised, then the same exact inner-product search; one pre-registered setting (k = 3, equal weights), no tuning.
Direction and size: **unsure, leaning null on nDCG@10.** Δ nDCG@10 point +0.005, range −0.015 to +0.020, CI straddling 0. Δ recall@50 point +0.010, range −0.005 to +0.030. Far below the LLM rerank: `dense-prf` nDCG@10 stays under 0.34 against `rerank_llm` 0.4052.
Kill clause: Δ nDCG@10 CI upper bound below 0 (PRF hurts). If it fires, the post says so with the per-query split below.
Pre-registered split (post-hoc rows are labelled as such): the 63 queries with dense recall@50 = 0 (counted in rerank-gate-oracle) against the other 260; and single-term vs 5+-word queries (the BM25 split in LESSONS).

## Mechanism
Expansion from feedback documents helps when the top hits are relevant and the query is short: NFCorpus has a median query of 2 words and 38.2 judged-relevant documents per query, which favours it. It hurts when the top hits are wrong (query drift): on the 63 queries where dense finds nothing relevant in the top 50, the top 3 are off-topic and should pull the query further away, so the gain, if any, sits in the other 260. A reranker reorders a fixed pool; PRF changes the pool, so recall@50 can move here where it could not under reranking.
