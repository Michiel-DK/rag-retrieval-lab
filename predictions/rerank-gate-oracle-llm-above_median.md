# rerank-gate-oracle-llm-above_median

One run of the rerank-gate-oracle lane: judge `llm`, gate feature `above_median`, `ragexp run gated-rerank --param judge=llm --param feature=above_median` (5-fold CV over queries, seed 42, min training skip rate 0.3). The lane's prediction is `predictions/rerank-gate-oracle.md`, committed 7805aef before any run; this file only names the run so the gate can find it, and adds nothing the lane prediction does not say. All six run files were written together, before any gate was scored.

## Prediction
As `predictions/rerank-gate-oracle.md`, part (2): the gated system at a ≥ 30% skip rate costs 0.005 to 0.015 nDCG@10 against `rerank_llm`, a CI that includes 0 is possible, and the gate recovers under a quarter of the oracle ceiling. recall@50 equals `dense` on every query (the gated ranking is the pool in one of two orders): a drift there is a harness bug. Known before the run: with 50 distinct pool scores, the count above the median is 25 for every query, so this feature cannot split queries; the fit can only skip all (gated = dense) and the run is the negative kept on the record.

## Mechanism
As `predictions/rerank-gate-oracle.md`: pool statistics describe the first-stage list, not whether the judge will disagree with it, so a one-feature gate separates safe skips from harmful ones poorly.
