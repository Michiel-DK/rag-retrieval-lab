# Defects surfaced by the repo-radar scoping pass — 2026-09-04

> Read-only audit against `b05b848`. Nothing fixed. Verify file:line before acting; the source is
> `repo-radar/docs/training-module-scope.md` Appendix A. Filed on branch `radar/defects-2026-09-04`.

1. **`ragexp/embed.py:39-46` — cache key omits `normalize`.** `encode(texts, normalize=False)` after a normalized call silently returns normalized vectors. Latent today; nb 05's `pool` mode is exactly the caller that would trip it. Fix: include the flag in `_key`.
2. **`ragexp/rerank.py:195` — `sort_keys=True` destroys pool order.** `results/rerank/llm_grades.json` alone cannot rebuild the 0.4052 ranking (doc-id order gives 0.3883); nb 04 recovers it only via the gitignored embedding cache. Persist the pool order or the reranked list.
3. **nb 03 cell 10 / `LESSONS.md:25` / `SHOWCASE.md:66` — BM25 post-mortem contradicts the corpus.** Median query is 2 words, 103/323 single terms; BM25 ties dense on 1-word queries (+0.0008, p=0.95) and loses only on 5–11-word ones (−0.104). The pre-registered mechanism was right; the after-the-fact explanation was never checked. Rewrite the three passages.
4. **nb 04 cell 8 — self-agreement table is a mean over 317, not 323.** Six all-zero-grade queries drop as `None`. Undocumented; say so under the table.
5. **`retrieve.py:53,59` — no punctuation stripping.** 97/1,064 query tokens carry attached punctuation and can never match. README owns stemming/stopwords, not this. Count note (2026-10-30): 97 is the number of tokens (323 test queries, `.lower().split()`, 1,064 tokens) containing any non-alphanumeric character; README's 73 counts only the tokens that start or end with one. The other 24 have it only inside (`x-rays`, `didn't`) and match only an identically punctuated document token.
6. **`retrieve.py:107-135` — `pool_doc_matrix` and `_chunks` have zero test coverage**, and `:131` calls `encode` per document inside the loop (~3,633 cache files). Test before nb 05 runs.
7. **`LESSONS.md:6` dangling cross-repo reference** to `docs/technical/RETRIEVAL_LEDGER.md` (lives in mast). `BUILD.md:71` "2B" vs `rerank.py:124` "1.7B" for zerank.
8. **`metrics.py:104,126` — bootstrap CI depends on query-list order** (fixed seed, positional resampling). Notebooks use corpus order, `make_readme_chart.py:41` uses sorted ids: third-decimal CI differences. Document, or sort inside `paired_bootstrap`.
9. **Framing:** nb 05's "adapter nulls" is a replication of mast's Track A1 result (`RETRIEVAL_LEDGER.md:206-211`), not an a-priori bet; and its "pooling helps" half is the ¼/¾ blend from mast transplanted (`retrieve.py:133`). `README.md:55-58` should say so.
