"""Lane rerank-gate-oracle: the oracle split, the per-fold gates, skip rates, AUCs and paired CIs.

    python experiments/rerank_gate.py

Reads committed files and the embedding cache only (no model call, no LLM). Writes
nothing: the committed per-query scores come from `ragexp run gated-rerank`, and this
script re-derives every number it prints from the same `GatedRerank.gate` call.

Prediction: predictions/rerank-gate-oracle.md (committed 7805aef, before any run).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ragexp.data import load_nfcorpus
from ragexp.gate import FEATURES, auc
from ragexp.metrics import paired_bootstrap
from ragexp.methods import GatedRerank
from ragexp.paths import RESULTS_DIR
from ragexp.runs import evaluate_run, load_scores


def main() -> int:
    corpus = load_nfcorpus()
    qids = [q.query_id for q in corpus.queries]
    order = json.loads((RESULTS_DIR / "query_order.json").read_text())
    assert qids == order, "corpus order differs from results/query_order.json"
    dense = load_scores("dense")
    zero_pool = {q for q in qids if dense[q]["recall@50"] == 0}

    for judge in ("llm", "zerank"):
        committed = load_scores(f"rerank_{judge}")
        full = np.array([committed[q]["ndcg@10"] for q in qids])
        base = np.array([dense[q]["ndcg@10"] for q in qids])
        print(f"\n=== judge {judge}: dense {base.mean():.4f} -> full rerank {full.mean():.4f} "
              f"(pooled Δ {full.mean() - base.mean():+.4f}) ===")
        gates = {f: GatedRerank(judge=judge, feature=f).gate(corpus) for f in FEATURES}
        gains = gates["top1"]["gains"]
        d = np.array([gains[q] for q in qids])
        # the gate's own nDCG must equal the committed files exactly
        assert np.allclose(d, full - base, atol=1e-12, rtol=0), "gate gains differ from committed scores"
        n = len(qids)
        pos, zer, neg = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
        zero_in_zero = sum(1 for q in qids if q in zero_pool and gains[q] == 0)
        ceiling = float(np.maximum(-d, 0).sum() / n)
        print(f"oracle split: Δ>0 {pos}/{n} ({pos / n:.1%})  Δ=0 {zer}/{n} ({zer / n:.1%}; "
              f"{zero_in_zero} of them have dense recall@50 = 0, of {len(zero_pool)} such queries)  "
              f"Δ<0 {neg}/{n} ({neg / n:.1%})")
        print(f"oracle ceiling (skip every Δ<0 query): +{ceiling:.4f} nDCG@10 over the full rerank "
              f"-> {full.mean() + ceiling:.4f}; harmed mean damage {(-d[d < 0]).mean():.4f}")
        harm = [gains[q] < 0 for q in qids]
        for f in FEATURES:
            g = gates[f]
            feats = np.array([g["features"][q] for q in qids])
            a = auc(feats, harm) if len(set(feats)) > 1 else 0.5
            call = np.array([g["call"][q] for q in qids])
            gated = np.where(call, full, base)
            rankings = {q: (g["judge_order"][q] if g["call"][q] else g["dense_order"][q]) for q in qids}
            r50 = evaluate_run(corpus, rankings, recall_ks=(50,), ndcg_ks=())
            assert all(r50[q]["recall@50"] == dense[q]["recall@50"] for q in qids), "recall@50 moved: broken harness"
            delta, ci, p = paired_bootstrap(list(full), list(gated))
            skip = 1 - call.mean()
            gain_full = full.mean() - base.mean()
            kept = (gated.mean() - base.mean()) / gain_full
            print(f"\n  feature {f}: AUC(feature high -> Δ<0) {a:.3f}  (orientation-free {max(a, 1 - a):.3f})")
            for i, fit in enumerate(g["fits"]):
                print(f"    fold {i}: skip when {f} {'>' if fit.direction == 'high' else '<'} {fit.threshold:.4f}  "
                      f"train skip {fit.train_skip_rate:.1%}")
            fold_rates = []
            from ragexp.gate import assign_folds
            fold_of = assign_folds(qids, 5, 42)
            for i in range(5):
                idx = [k for k, q in enumerate(qids) if fold_of[q] == i]
                fold_rates.append((1 - call[idx].mean(), gated[idx].mean() - full[idx].mean()))
            print("    held-out per fold (skip, Δ vs full): "
                  + "  ".join(f"{s:.0%}/{x:+.4f}" for s, x in fold_rates))
            print(f"    held-out skip rate {skip:.1%}; gated nDCG@10 {gated.mean():.4f} vs full {full.mean():.4f}: "
                  f"Δ {delta:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], p={p:.4f}; keeps {kept:.1%} of the rerank gain; "
                  f"recovers {max(gated.mean() - full.mean(), 0) / ceiling:.1%} of the oracle ceiling")
            print(f"    kill test (skip ≥ 30% and CI lower bound ≥ −0.01): "
                  f"{'PASS' if skip >= 0.3 and ci[0] >= -0.01 else 'fail'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
