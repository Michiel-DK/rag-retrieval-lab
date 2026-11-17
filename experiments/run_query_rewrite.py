"""Produce results/query-rewrite/generations.json: one Haiku passage per NFCorpus test query.

Lane query-rewrite. One ``claude -p`` call per query in ``results/query_order.json`` order,
sequential, checkpointed after every call, a $2 ceiling on the CLI's list-price cost.
Prompt and route: ``ragexp/rewrite.py``, as committed in ``predictions/query-rewrite.md``.
Safe to kill and re-run; stored queries are never re-run.

Usage (repo root):
    python experiments/run_query_rewrite.py
"""
import json
import sys

from ragexp.data import load_nfcorpus
from ragexp.paths import RESULTS_DIR
from ragexp.rewrite import GENERATIONS_FILE, BudgetExceeded, ClaudeCLIRewriter, build_generations, sha256_file

order = json.loads((RESULTS_DIR / "query_order.json").read_text())
text = {q.query_id: q.text for q in load_nfcorpus().queries}
try:
    data = build_generations([(q, text[q]) for q in order], ClaudeCLIRewriter(model="haiku"), ceiling_usd=2.0)
except BudgetExceeded as e:
    print(f"HOLD: {e}")
    sys.exit(3)
g = data["generations"]
ok = [v for v in g.values() if v]
tot = {k: sum(v[k] for v in ok) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                                          "output_tokens", "thinking_tokens")}
print(f"done: {len(g)} queries, {len(g) - len(ok)} null, {data['attempts']} attempts, "
      f"models {sorted({v['model'] for v in ok})}, tokens {tot}, spent ${data['spent_usd']:.4f}")
print(f"sha256 {sha256_file(GENERATIONS_FILE)}")
