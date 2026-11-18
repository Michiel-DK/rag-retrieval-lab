# query-rewrite-q2d

One run of the query-rewrite lane: `ragexp run dense-q2d --name query-rewrite-q2d --recall-k 10 --recall-k 50 --recall-k 300`, reading only the committed `results/query-rewrite/generations.json`. The lane's prediction is `predictions/query-rewrite.md`, committed in the same commit as this file, before any generation. This file only names the run so the gate can find it, and adds nothing the lane prediction does not say.

## Prediction
As `predictions/query-rewrite.md`, the `dense-q2d` bullet: Δ nDCG@10, recall@50 and recall@300 vs `dense` / `pool-depth-dense` inside the stated ranges, leaning null; below `rerank_llm` by more than 0.06 nDCG@10. Kill: Δ nDCG@10 CI upper bound below 0 (hurts); CI lower bound above 0 refuses the lean-null the other way (helps).

## Mechanism
As `predictions/query-rewrite.md`: a frozen MiniLM-L6 encoder never trained on expanded queries, one zero-shot passage against 38.2 relevant abstracts per query.
