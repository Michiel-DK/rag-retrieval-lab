# Literature

Only what this lab ran against. Each row: the paper or repo, the claim tested, the run that tested it, what held.
The radar (private) holds what exists and its verdicts; this file holds what was reproduced here.
Full table, comparison and the "may / may not claim" list: `docs/literature-2026-11-16.md`. Every row below is rank [F] (full text read).

| source | claim tested | run | held? |
|---|---|---|---|
| UIBK-Embedding-v1, arXiv 2610.27195, Table 2 (NFCorpus nDCG@10 38.79) | the model as first stage alone reaches the zerank rerank's nDCG@10 on NFCorpus | `uibk-first-stage` (0.3894, `predictions/uibk-first-stage.md`) | Number reproduces within 0.0015. Not a held-out number: the paper's ablations are scored on BEIR averages. Document length of our run not pinned (unchecked). |
| "Beyond Recall Saturation", arXiv 2610.24478, Table II | embedding recall saturates by top-300 | `pool-depth-dense`, `-bm25`, `-rrf`, `pool-depth-uibk` | Not a conflict. The paper shows diminishing returns at 86 to 95% recall with 1.9 positives per query and disclaims anything past 300; NFCorpus (38.2 relevant per query) reaches 0.42. Per-100-docs gain 200 to 300 is 3.4 to 5.8 points here vs 1.3 to 5.0 there. |
| "Per-Query Gating of LLM Rerankers", arXiv 2609.22880, Tables 3 to 5 | a per-query gate can skip the reranker where it will not help | `rerank-gate-oracle-{llm,zerank}-{top1,gap,above_median}` | Same qualitative result (weakly discriminative signals; random gate at matched skip rate is the baseline). The paper has no AUC 0.65 line; the prediction and LESSONS row cite one. |
| Bruch et al., arXiv 2210.11934, Table 2 | RRF of BM25 and MiniLM dense beats dense (the "RRF widely reported to help on BEIR" line in the nb 03 prediction, README) | `rrf`, `pool-depth-rrf` (nb 03) | **Opposite sign.** Same dense model, same 323 queries: their RRF recall@100 0.327 vs dense 0.320 (p<0.01); ours 0.2989 vs 0.3115. Their BM25 is stemmed, k1=0.9, b=0.4 (recall@100 0.255); ours is not (0.2187). Hypothesis, untested: the weaker BM25. |
| BEIR, arXiv 2104.08663, Tables 1, 2, 9 | BM25 loses to dense on NFCorpus | `bm25`, `dense` (nb 02, 03) | Not comparable. BEIR BM25 0.325 nDCG@10 vs ours 0.2678; BEIR has BM25 above TAS-B (0.319). Gain function of BEIR's nDCG not stated. |
| RankGPT, arXiv 2304.09542, Table 1 | LLM reranking lifts NFCorpus nDCG@10 | `rerank_llm` (nb 04) | Same direction and order of size (+4.9 gpt-3.5, +7.7 gpt-4 over BM25 vs our +8.9 over dense). First stage, pool and model differ, so no matches/beats claim. |
| ConTEB, arXiv 2505.24782, Table 2 | context lines help where chunks lose their document, not where they do not | `chunk_context`, `chunk_max_context` (nb 07) | Consistent. 52.0 → 72.4 on context-dependent sets, 63.2 = 63.2 on unchunked NanoBEIR; magnitudes not comparable (abstracts, about five chunks). |
