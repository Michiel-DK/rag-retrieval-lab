# query-rewrite

origin: {"kind": "lane", "url": "https://github.com/Michiel-DK/rag-retrieval-lab/pull/54", "label": "query-prf follow-up: the model-rewrite arm"}

Retrieval lane, written 2026-11-17 before any generation or run. Reader question: **C5** (does the trick everyone talks about hold on office work?). decider: operator, whether the rag card says "model query rewriting helps / does not help on NFCorpus" as measured; flips when either arm's Δ nDCG@10 CI lower bound is above 0 (helps) or upper bound below 0 (hurts). Queued after `query-prf` (#54) held its null. PRF drifts because the top-3 hits are often off-topic. A model rewrite does not read the hits, so the PRF null does not predict this one. Literature read in full first: HyDE (2212.10496) and Query2doc (2303.07678), rows 12 and 13 of `docs/literature-2026-11-16.md`.

## Generation (fixed before any call)
One generation per NFCorpus test query, 323 calls (`results/query_order.json`), **shared by both arms**. Both arms then differ only in how the same passage is used, which makes hyde vs q2d a paired test. The queue brief also names "646 generations" in its cost line; its body says one generation per query with the same model and prompt, and this file follows the body.

Generator: Haiku through `claude -p --model haiku`, the nb 07 route (`ragexp/contextualize.py`): subscription login, no API key, no tools, no settings, empty working directory, `MAX_THINKING_TOKENS=0`. A route check on 2026-11-17 (prompt "Reply with the single word OK.", not an NFCorpus query) resolved the alias to `claude-haiku-5-5` with 0 thinking tokens. No temperature control on this route (API default). No output cap on this route either, so the length is capped by instruction only and the passage is never truncated afterwards. Per call the file records the model id, input / cache-creation / cache-read / output / thinking tokens, and `total_cost_usd`. That cost is the CLI's list-price figure (`costBasis: list`) under subscription auth, not an invoice. Ceiling $2: the script stops before a call that could cross it, and the lane HOLDs.

System prompt (Query2doc App. A's anti-clarification line):

```
You are asked to write a passage that answers the given query. Do not ask the user for further clarification.
```

User prompt (HyDE A.1.4, the TREC-COVID "scientific paper passage" instruction, closest to PubMed abstracts; plus a length cap near Query2doc's 128 tokens):

```
Please write a scientific paper passage to answer the question
Question: {query}
Passage:

Reply with the passage only: one paragraph of at most 100 words.
```

A call that fails or returns an empty reply after 2 retries is stored as `null`. That query then uses the plain query vector in both arms (= `dense`), and the post counts such queries. Generations go to `results/query-rewrite/generations.json`. They are committed once, never regenerated, and pinned by sha256 in each run's params. The scoring reads only that file, so `ragexp check` re-derives every number with no model call. Predicted: at most 5 of 323 passages are a clarification request, a refusal or a non-passage. The post counts them by a fixed rule: the reply contains "?" in its first sentence, or starts with "I " / "I'm" / "Could you" / "Please".

## Arms
- `query-rewrite-hyde` (method `dense-hyde`): HyDE Eq. 8 at N = 1. v = normalise((f(q) + f(p)) / 2), with f the lab's MiniLM-L6 encoder (unit vectors, the `dense` cache), then the same exact inner-product search as `dense`.
- `query-rewrite-q2d` (method `dense-q2d`): Query2doc Eq. 2, the dense form: f("{query} [SEP] {passage}"), with the query once. Repetition (Eq. 1) is the paper's sparse form only. The literal `[SEP]` tokenises to MiniLM's special token (id 102, checked). MiniLM cuts input at 256 word pieces; a 100-word passage plus the query fits.

## Prediction
Metric: nDCG@10 (primary), recall@50 and recall@300, paired bootstrap Δ (seed 42, 10,000 resamples) on the 323 queries, human qrels. Baselines: `dense` (nDCG@10 0.3167, recall@50 0.2508), `pool-depth-dense` for recall@300 (0.4242), `query-prf` (`dense-prf`: 0.3120 / 0.2652 / 0.4542), `rerank_llm` (0.4052).

**Unsure, leaning null on nDCG@10 for both arms, with a real chance of a loss for q2d.**
- `dense-hyde` vs `dense`: Δ nDCG@10 point −0.005, range −0.030 to +0.020; Δ recall@50 point +0.005, range −0.020 to +0.030; Δ recall@300 point +0.010, range −0.020 to +0.040.
- `dense-q2d` vs `dense`: Δ nDCG@10 point −0.010, range −0.040 to +0.020; Δ recall@50 point 0.000, range −0.030 to +0.030; Δ recall@300 point +0.005, range −0.030 to +0.040.
- `dense-q2d` vs `dense-hyde`: q2d at or below hyde on nDCG@10 (point −0.005).
- Both arms vs `query-prf`: Δ nDCG@10 within ±0.025, CI including 0.
- Both arms vs `rerank_llm`: below by more than 0.06 nDCG@10, CI wholly below 0.

Kill clause: an arm whose Δ nDCG@10 vs `dense` has a CI upper bound below 0 **hurts**, and the post says so with the splits below. An arm whose CI lower bound is above 0 **helps**: the prediction is refused in that direction, and the post says that too.

Pre-registered splits, the same as `query-prf` (post-hoc rows are labelled as such): the 63 queries with dense recall@50 = 0 against the other 260; and 1-word (n = 103) against 5+-word (n = 96) queries. Predicted: the gain, if any, sits in the 1-word queries (the passage adds the context a one-word query lacks), not the 5+-word ones. On the 63, nDCG@10 and recall@50 start at 0 and can only rise, so their rise is a floor effect and is not read as a win.

## Mechanism
Query2doc's own NFCorpus row is SimLM −0.6 and E5 +0.2 nDCG@10. Both retrievers were fine-tuned on expanded queries, and the paper blames the NFCorpus and SciFact drops on train/eval mismatch. Ours is a frozen encoder that never saw an expanded query, with a zero-shot prompt. HyDE's large gains are over an unsupervised encoder. Over a fine-tuned one (its Table 4) the gains run from −1.9 to +5.3 points and depend on the LM. NFCorpus queries are NutritionFacts titles and topic words (median 2 words); the relevant documents are PubMed abstracts, 38.2 per query. A generated passage can bridge the lay-to-medical vocabulary, which is what would help. But it also commits to one reading of a vague query ("deafness") while the qrels reward many, and a short model passage cannot cover 38 relevant abstracts. Averaging (hyde) keeps the query at half weight. Concatenation (q2d) mean-pools ~3 query tokens against ~130 passage tokens, so the passage dominates and drifts further. In nb 07, Haiku lines added to chunks bought nothing pooled, and PRF nulled on nDCG@10 here. Neither is evidence for this arm, but both say the encoder, not the query text, is the limit on this corpus.
