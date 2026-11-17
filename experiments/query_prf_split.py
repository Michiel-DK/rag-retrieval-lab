"""Lane query-prf: the pre-registered per-query split of dense-prf vs dense.

Read-only. Reads the committed results/scores/{dense,pool-depth-dense,query-prf}.json and the
query texts (dataset cache). Two splits, both written in predictions/query-prf.md before the run:
(1) dense recall@50 = 0 (the 63 queries counted in rerank-gate-oracle) vs the rest;
(2) query length in whitespace words: 1 word vs 5+ words (the nb 03 BM25 split; counts 103 / 96).
Paired bootstrap per subgroup, seed 42, 10,000 resamples, as `ragexp diff`. Run from the repo root:
    python experiments/query_prf_split.py
"""
from ragexp import runs
from ragexp.data import load_nfcorpus
from ragexp.metrics import paired_bootstrap

c = load_nfcorpus()
words = {q.query_id: len(q.text.split()) for q in c.queries}
dense, deep, prf = runs.load_scores("dense"), runs.load_scores("pool-depth-dense"), runs.load_scores("query-prf")
zero = {q for q in dense if dense[q]["recall@50"] == 0}
print(f"dense recall@50 = 0: {len(zero)} of {len(dense)}; 1 word: {sum(v == 1 for v in words.values())}; "
      f"5+ words: {sum(v >= 5 for v in words.values())}")

groups = {
    "dense recall@50 = 0": lambda q: q in zero,
    "dense recall@50 > 0": lambda q: q not in zero,
    "1 word": lambda q: words[q] == 1,
    "5+ words": lambda q: words[q] >= 5,
}
for metric, base in (("ndcg@10", dense), ("recall@50", dense), ("recall@300", deep)):
    for label, keep in groups.items():
        ids = sorted(q for q in prf if keep(q))
        d, ci, p = paired_bootstrap([base[q][metric] for q in ids], [prf[q][metric] for q in ids],
                                    n_resamples=10_000, seed=42)
        up = sum(prf[q][metric] > base[q][metric] for q in ids)
        down = sum(prf[q][metric] < base[q][metric] for q in ids)
        print(f"{metric:>10}  {label:<20} n={len(ids):>3}  delta {d:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"p={p:.4f}  up {up} / down {down}")
