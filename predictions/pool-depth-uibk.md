# pool-depth-uibk

Retrieval lane, written 2026-11-07 by the follow-up scope job, before any run. Origin: the pool-depth lane (PR #47) measured dense,
bm25 and rrf at recall@50/100/200/300 and left the uibk arm unrun because `uibk-first-stage` was not yet on master. Question: does
UIBK-Embedding-v1's recall lead over dense (+0.0468 at @50) persist to @300, or do the two curves converge?

## Prediction
Metric: recall@50, @100, @200, @300 for `uibk-first-stage` (run name `pool-depth-uibk`), paired bootstrap against
`results/scores/pool-depth-dense.json` at each depth and against its own recall@50. Case set: the same 323 NFCorpus test queries.
Baselines: `results/scores/uibk-first-stage.json` recall@50 0.2976; `results/scores/pool-depth-dense.json` recall@50 0.2508,
@100 0.3115, @200 0.3833, @300 0.4242.
Direction and size: **no saturation for uibk either**: recall@300 − recall@50 ≥ +0.10, point recall@300 0.46 (0.42 to 0.52).
**The lead over dense shrinks with depth but does not close**: Δ vs dense +0.0468 at @50 → about +0.03 at @300 (0.01 to 0.05), CI
still excluding 0 at @300; unsure on the size at @100 and @200, confident the sign stays positive at every depth.
nDCG@10 and recall@10 of the new run equal the committed `uibk-first-stage` values exactly (same method, same weights, same revision):
a drift there is a bug, not a result.
Kill clause: if the Δ vs dense at recall@300 CI includes 0 (the two first stages converge), the post says the uibk lead is a top-of-list
effect and a deeper-pool rerank should start from dense, not uibk. If recall@300 − recall@50 CI upper bound is below 0.05, uibk
saturates where dense does not; post that as the surprise.

## Mechanism
Late-interaction MaxSim scores token pairs, so it separates near-duplicates and topical neighbours better than one pooled MiniLM vector:
that is where its +0.07 nDCG@10 comes from, and it is a top-of-list effect. Deep in the list both models are ranking documents that
share only loose topical overlap with the query, and token-level matching has less to add there, so the gap should narrow. It should not
close, because the uibk encoder is also the stronger model on BEIR-style corpora at every cut the card reports. The curve keeps
rising to 300 for the same reason dense's did: NFCorpus queries have a median of 16 judged-relevant docs (mean 38.2) spread across the
corpus, so no first stage has found them all by rank 50.
