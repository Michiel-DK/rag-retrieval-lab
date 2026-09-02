"""Recompute the README results table and render docs/img/ndcg10-by-method.png.

Every number comes from the committed per-query score files under
``results/scores/*.json`` (n = 323 NFCorpus queries, human qrels), the same
source LESSONS.md recomputes from. Nothing is read off a notebook cell.

    python experiments/make_readme_chart.py

Prints the table (mean nDCG@10, 95% bootstrap CI of that mean, paired delta vs
dense with its CI, mean recall@10, mean recall@50) and writes the chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragexp.metrics import independent_ci, paired_bootstrap  # noqa: E402
from ragexp.runs import load_scores, metric_vector, summarize  # noqa: E402

SYSTEMS = [
    ("dense", "dense\n(MiniLM-L6)"),
    ("bm25", "BM25\n(untuned)"),
    ("rrf", "RRF\n(dense+BM25)"),
    ("rerank_zerank", "zerank-1-small\nrerank"),
    ("rerank_llm", "LLM-judge\nrerank"),
]

OUT = ROOT / "docs" / "img" / "ndcg10-by-method.png"


def main() -> None:
    scores = {name: load_scores(name) for name, _ in SYSTEMS}
    qids = sorted(scores["dense"])
    dense_vec = metric_vector(scores["dense"], "ndcg@10", qids)

    rows = []
    for name, _ in SYSTEMS:
        vec = metric_vector(scores[name], "ndcg@10", qids)
        mean, ci = independent_ci(vec)
        summ = summarize(scores[name])
        if name == "dense":
            delta, dci, p = 0.0, (0.0, 0.0), 1.0
        else:
            delta, dci, p = paired_bootstrap(dense_vec, vec)
        rows.append((name, mean, ci, delta, dci, p, summ["recall@10"], summ["recall@50"]))

    print(f"{'system':14s} {'nDCG@10':>8s} {'95% CI (mean)':>18s} {'d vs dense':>11s} "
          f"{'95% CI (paired)':>20s} {'p':>6s} {'r@10':>7s} {'r@50':>7s}")
    for name, mean, ci, delta, dci, p, r10, r50 in rows:
        print(f"{name:14s} {mean:8.4f} [{ci[0]:.4f}, {ci[1]:.4f}] {delta:+11.4f} "
              f"[{dci[0]:+.4f}, {dci[1]:+.4f}] {p:6.3f} {r10:7.4f} {r50:7.4f}")

    # ── chart ──────────────────────────────────────────────────────────────
    bg, fg, grid = "#f4f5f7", "#1f2937", "#d1d5db"
    bar = ["#5b6b7f", "#8a97a6", "#8a97a6", "#3f5f7f", "#2f4a63"]
    labels = [lab for _, lab in SYSTEMS]
    means = [r[1] for r in rows]
    lo = [r[1] - r[2][0] for r in rows]
    hi = [r[2][1] - r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    x = range(len(rows))
    ax.bar(x, means, color=bar, width=0.62, yerr=[lo, hi], capsize=4,
           error_kw={"ecolor": fg, "elinewidth": 1.2})
    ax.axhline(means[0], color=fg, lw=0.9, ls="--", alpha=0.6)
    for i, m in enumerate(means):
        ax.text(i, m + hi[i] + 0.006, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, color=fg)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=fg)
    ax.set_ylabel("nDCG@10 (mean over 323 queries)", color=fg, fontsize=9.5)
    ax.set_ylim(0, 0.5)
    ax.tick_params(colors=fg, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.yaxis.grid(True, color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("NFCorpus, human qrels. Bars: mean nDCG@10; whiskers: 95% bootstrap CI of the mean; dashed: dense baseline",
                 fontsize=9.5, color=fg, loc="left")
    fig.text(0.01, 0.01,
             "Source: results/scores/*.json via experiments/make_readme_chart.py. "
             "Per-system CIs; the verdicts in LESSONS.md use the paired bootstrap.",
             fontsize=7, color=fg, alpha=0.75)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=bg)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
