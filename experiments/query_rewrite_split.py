"""Lane query-rewrite: the pre-registered per-query splits and the generation counts.

Read-only. Reads results/scores/{dense,pool-depth-dense,query-rewrite-hyde,query-rewrite-q2d}.json,
results/query-rewrite/generations.json and the query texts (dataset cache). Splits as
predictions/query-rewrite.md: dense recall@50 = 0 (63) vs the rest; 1 word (103) vs 5+ words (96).
Paired bootstrap per subgroup, seed 42, 10,000 resamples, as `ragexp diff`. Also: null passages,
the pre-registered clarification/refusal rule, tokens and cost. Run from the repo root:
    PYTHONPATH=. python experiments/query_rewrite_split.py
"""
import json
import re

from ragexp import runs
from ragexp.data import load_nfcorpus
from ragexp.metrics import paired_bootstrap
from ragexp.paths import RESULTS_DIR

data = json.loads((RESULTS_DIR / "query-rewrite" / "generations.json").read_text())
g = data["generations"]
ok = {q: v for q, v in g.items() if v}


def non_passage(p: str) -> bool:
    # predictions/query-rewrite.md: "?" in the first sentence, or starts with I / I'm / Could you / Please
    first = re.split(r"(?<=[.!?])\s", p, maxsplit=1)[0]
    return "?" in first or bool(re.match(r"(I |I'm|Could you|Please)", p))


flagged = sorted(q for q, v in ok.items() if non_passage(v["passage"]))
tok = {k: sum(v[k] for v in ok.values()) for k in ("input_tokens", "cache_creation_input_tokens",
                                                     "cache_read_input_tokens", "output_tokens", "thinking_tokens")}
words = [len(v["passage"].split()) for v in ok.values()]
print(f"generations: {len(g)} queries, {len(g) - len(ok)} null, {data['attempts']} attempts, "
      f"models {sorted({v['model'] for v in ok.values()})}")
print(f"tokens {tok}; cost ${data['spent_usd']:.4f} (claude -p total_cost_usd, costBasis list)")
print(f"passage words: median {sorted(words)[len(words) // 2]}, max {max(words)}, over 100: {sum(w > 100 for w in words)}")
print(f"non-passage rule: {len(flagged)} flagged {flagged}")
for q in flagged:
    print(f"  {q}: {ok[q]['passage'][:160]}")

c = load_nfcorpus()
nw = {q.query_id: len(q.text.split()) for q in c.queries}
dense, deep = runs.load_scores("dense"), runs.load_scores("pool-depth-dense")
zero = {q for q in dense if dense[q]["recall@50"] == 0}
groups = {
    "all": lambda q: True,
    "dense recall@50 = 0": lambda q: q in zero,
    "dense recall@50 > 0": lambda q: q not in zero,
    "1 word": lambda q: nw[q] == 1,
    "5+ words": lambda q: nw[q] >= 5,
}
for arm in ("query-rewrite-hyde", "query-rewrite-q2d"):
    s = runs.load_scores(arm)
    print(f"\n== {arm} vs dense (recall@300 vs pool-depth-dense)")
    for metric, base in (("ndcg@10", dense), ("recall@50", dense), ("recall@300", deep)):
        for label, keep in groups.items():
            ids = sorted(q for q in s if keep(q))
            d, ci, p = paired_bootstrap([base[q][metric] for q in ids], [s[q][metric] for q in ids],
                                        n_resamples=10_000, seed=42)
            up = sum(s[q][metric] > base[q][metric] for q in ids)
            down = sum(s[q][metric] < base[q][metric] for q in ids)
            print(f"{metric:>10}  {label:<20} n={len(ids):>3}  delta {d:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                  f"p={p:.4f}  up {up} / down {down}")
