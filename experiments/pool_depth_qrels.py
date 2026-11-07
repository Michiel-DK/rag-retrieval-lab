"""Pool-depth lane: judged-relevant docs per query, and the dense Δ split by that count.

Read-only. Reads the qrels (same definition as recall: grade >= 1) and the committed
results/scores/pool-depth-dense.json. Run from the repo root:
    python experiments/pool_depth_qrels.py
"""
import statistics as st

from ragexp import runs
from ragexp.data import load_nfcorpus

c = load_nfcorpus()
n = {q.query_id: len(c.relevant_ids(q.query_id)) for q in c.queries}
v = list(n.values())
print(f"queries {len(v)}  median {st.median(v)}  mean {st.mean(v):.1f}  max {max(v)}  >50: {sum(x > 50 for x in v)}")
d = runs.load_scores("pool-depth-dense")
for label, keep in (("<=50 relevant", lambda x: x <= 50), (">50 relevant", lambda x: x > 50)):
    ids = [q for q in d if keep(n[q])]
    gain = st.mean(d[q]["recall@300"] - d[q]["recall@50"] for q in ids)
    print(f"dense recall@300 - recall@50, {label}: n={len(ids)}  mean {gain:+.4f}  (post-hoc)")
