# BUILD — rag-retrieval-lab sprint plan

This repo's slice of `~/Desktop/applications/TRACK.md` (**Q1: finish rag-retrieval-lab**).
The hub carries status + cross-cutting principles; this file carries the build detail.

**Definition of done (Q1):** all six notebooks run, every prediction resolved to an
outcome, a predictions→results table committed, `LESSONS.md` written. No new repos until
this and Q2 are done (TRACK.md ground rule).

**The discipline:** every notebook states its prediction + mechanism *before* running.
A prediction written afterwards is a story, not a prediction.

---

## Current state (2026-08-17) — stop-and-ship reached

| component | status |
|---|---|
| `ragexp/metrics.py` | ✅ done, unit-tested |
| `ragexp/data.py` | ✅ done, tested (3633 docs / 323 queries / 12334 judgments) |
| `ragexp/embed.py` | ✅ done, tested — content-addressed cache under `results/emb/` |
| `ragexp/retrieve.py` | ✅ done, tested — `rrf()` grew a `depth` param; nb 03 resolved the open question empirically (**convention: depth=1000**, lossless vs full-ranking fusion at scored ks) |
| `ragexp/runs.py` | ✅ done, tested — per-query score persistence to `results/scores/` |
| `ragexp/rerank.py` | ✅ done, tested — LLM judge runs on **headless Claude Code** (`claude -p`, subscription auth, no API key) + `zerank-1-small` (needs `torch.inference_mode` on MPS); sweeps checkpointed to `results/rerank/` and **committed**, so notebooks reproduce with no LLM/GPU |
| test suite | ✅ 46 tests (`pytest -q`; integration marks need local HF caches) |
| nb 01 `the-ruler` | ✅ run — zero-poisoning understates 40%; paired p≈0 where independent CIs overlap |
| nb 02 `dense-baseline` | ✅ run — nDCG@10 **0.3167**, recall@10 0.155, recall@50 0.251 |
| nb 03 `lexical-and-fusion` | ✅ prediction resolved — BM25 **loses** to dense everywhere; **RRF nulls vs dense** (source-system null transferred); fusion only beats the weaker parent |
| nb 04 `reranking` | ✅ prediction held — nDCG@10 0.317→**0.405** (LLM judge, p≈0) / 0.388 (zerank); **recall@50 frozen bit-for-bit** (self-check passed); judge-grading-own-homework demo: nDCG=1.0000 vs own labels, 0.405 vs human |
| nb 05 `pooling-vs-summary` | ⬜ todo |
| nb 06 `learning-to-rank` | ⬜ todo — ⚠️ verify MSLR-WEB10K license before publishing derived numbers |
| `LESSONS.md` + predictions→results table | ✅ done 2026-08-19 — `LESSONS.md` written from the notebooks' closing cells, all means recomputed from `results/scores/*.json` (n=323); predictions→outcomes table lives in both README.md and LESSONS.md |

Env: `pyenv` virtualenv `rag-retrieval-lab` (3.12.9). Repo private on GitHub; flip public + secret-scan only after the deliverable lands.

---

## Build order (each step unlocks a notebook)

1. **`embed.py`** — sentence-transformers wrapper, cached embeddings to `results/`.
   Unlocks nb 02.
2. **`retrieve.py`** — `dense()`, `bm25()`, `rrf()`, `pool_chunks()`. Unlocks nb 02, 03, 05.
3. **nb 01 `the-ruler`** — metrics + the paired-vs-independent bootstrap demo (smoke test
   already produced the punchline: paired p≈0, independent CIs overlap on the same data).
4. **nb 02 `dense-baseline`** — the bar everything beats.
5. **nb 03 `lexical-and-fusion`** — BM25 + RRF. **Prediction: unsure** (BM25 strong on
   bio-medical terms; RRF often helps on BEIR — genuinely open).
6. **`rerank.py`** — LLM judge + `zerank-1-small` (2B, off-the-shelf). Unlocks nb 04.
   Needs an API key; commit outputs to `results/` so downstream reproduces free.
7. **nb 04 `reranking`** — THE SPINE. **Prediction: nDCG rises, recall@k does not move**
   (reranking reorders a fixed pool → cannot change top-k membership → recall@k is
   structurally immune to reranker self-agreement). If recall@k moves, the harness is
   broken — this doubles as a self-check.
8. **nb 05 `pooling-vs-summary`** — title-embed vs chunk-pool; trained adapter.
   **Prediction: pooling helps** (body carries signal the title doesn't); **adapter nulls**
   (same encoder → query and doc already co-embedded → nothing to align; corpus-independent
   argument, should transfer from the source system).
9. **nb 06 `learning-to-rank`** — XGBoost/LambdaMART on **MSLR-WEB10K** (via TFDS
   `mslr_web`; 10k queries, 136 features, 0–4 graded). **Prediction: LambdaMART wins**
   (MSLR is LTR's home turf). Test whether the [Elsevier null](https://github.com/elsevierlabs-os/build-ltr-models-using-llm)
   was about LTR or about their LLM-generated labels.
10. **`LESSONS.md` + predictions→results table** — the deliverable.

**Stop-and-ship point:** notebooks 01–04 done = a complete, publishable artifact. 05/06
are depth.

---

## Port table (from `Michiel-DK/mast`)

| Source | Use here | Note |
|---|---|---|
| `scripts/analysis/_retrieval_metrics.py` | `metrics.py` | ✅ ported — math only (~60 of 240 lines; rest was MAST fixture schema) |
| `mast/storage/_retrieval_fusion.py` (151) | `retrieve.py::rrf()` | RRF, pure module |
| `mast/matching/reranker.py` (362) | `rerank.py` | listwise 0–3 grader; swap LLM client |
| `scripts/analysis/track_a1_embed_pool/embed_pool_experiment.py` | nb 05 | pooling template |
| `scripts/analysis/track_a1_embed_pool/embed_adapter_train.py` | nb 05 | InfoNCE adapter (the null) |
| `docs/technical/RETRIEVAL_LEDGER.md` | `LESSONS.md` format | what / number / verdict / lesson |

**Nothing proprietary ports** — no firm data, no client pitches, no labels. NFCorpus +
MSLR-WEB10K are public.

---

## Datasets (verified 2026-07-16)

- **NFCorpus** (`BeIR/nfcorpus` + `-qrels`): 3633 docs, 323 judged queries, 12334
  judgments (~38/query), grades {1: 11758, 2: 576} + implicit 0. Docs = `title` + `text`
  (~1.7k chars). Graded + densely judged + tiny = the whole retrieval story.
- **MSLR-WEB10K** (TFDS `mslr_web`): 10k queries, 136 pre-computed features, 0–4 graded.
  For nb 06 only. ⚠️ confirm license on the MSR page before publishing derived numbers.
- ❌ ZeroEntropy graded MTEB labels — described in their blog but **not published** (HF org
  has only `zeroentropy/polysemy`). Cite the finding, can't reproduce it. Their
  `zerank-1-small` model IS usable (nb 04).

---

## Out of scope for Q1 (parked, do not start — moratorium)

- **MAST-side judge validation** (human-calibrated gold set, behavior-vs-judge audit) —
  that's **Q3**, gated behind Q1+Q2. Scoped in scratchpad briefs `S-JUDGE-VALIDATION-brief-v2.md`;
  move to MAST `improvements.md` when Q3 opens. ⚠️ scratchpad is ephemeral — see TRACK.md note.
- LangGraph adaptive-retrieval loop — separate, later, only if measured against this harness.
- Corpus-fidelity / synthetic planted-truth corpus — **cut** (unverifiable appeal to a
  private reference; use the public benchmark instead).
