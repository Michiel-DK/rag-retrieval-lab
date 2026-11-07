# pool-depth

Retrieval lane, written 2026-11-07 by the radar's parked review, before any run. Origin: radar verdict "2026-11-07 · Parked review", item Beyond Recall Saturation, https://arxiv.org/abs/2610.24478; this run tests its first claim (embedding recall saturates by top-300) on NFCorpus, not its reranker-training result. Question: on NFCorpus, does first-stage recall saturate by top-300, i.e. is widening the 50-doc rerank pool a lever or not?

## Prediction
Metric: recall@50, @100, @200, @300 per first stage (dense, bm25, rrf; plus `uibk-first-stage` if that run is committed). The headline is the paired-bootstrap Δ recall@300 − recall@50 per method. Case set: the same 323 NFCorpus test queries. Baseline: `results/scores/dense.json` recall@50 0.2508, `results/scores/rrf.json` 0.2521, `results/scores/bm25.json` 0.1792.
Direction and size: **no saturation on NFCorpus.** Dense recall@300 rises by ≥ 0.10 over recall@50; point 0.40 (0.33 to 0.48), unsure on the size, confident on the direction.
Kill clause: if dense recall@300 − recall@50 < 0.05 (CI upper bound below 0.05), pool depth is not a lever on NFCorpus. The post says the reranker, not the pool, is the target, and no deeper-pool rerank run is proposed.

## Mechanism
A reranker can only reorder what the first stage hands it, and the committed recall@50 of 0.25 means about three quarters of the judged-relevant documents never reach any of the lab's rerankers. The paper measured saturation on Chinese legal retrieval, with its own candidate pools; NFCorpus qrels are graded and broad, so a query has many relevant documents spread across the corpus (⚠️ from memory: count qrels per query before the run and put the median in the post), and the curve should keep climbing past 50. If it does, the 0.4052 LLM-judge result is capped by the pool, and the next lever is depth, not a better judge.
