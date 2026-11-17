# Second corpus: which one, and is it closer to what automations clients search? (2026-11-17)

> STATUS: scope, no run. Written by scheduler lane `lab-rag-second-corpus-scope` (operator 17 Nov 2026: "shouldn't we start
> testing on other corpus? maybe more applicable for automations industries?"). Reader question **C5** (does the trick everyone
> talks about hold on office work?). Decider: operator, which public corpus the rag lab adds second (and whether an in-house office
> corpus follows); flips when the comparison shows a candidate fails the lab's own corpus criteria (graded or dense judgments, size
> a laptop re-runs, licence) or fits the automations industries no better than NFCorpus.

## Verdict on the decider: it flips

**Every public candidate fails "graded" and "densely judged"**, the two README criteria that made NFCorpus the pick (README "The
dataset"). The counts below come from the published qrels files, downloaded and counted in this lane (sha256 at the end). They do
not come from papers or memory:

| | judgments per test query (mean / median) | grades in qrels |
|---|---|---|
| NFCorpus (anchor, BEIR Table 1 [F]) | 38.2 | 1, 2 (+ implicit 0) |
| FiQA-2018 test | 2.63 / 2 | all 1 (1,706 of 1,706) |
| CQADupStack, 12 subforums pooled | 1.80 / 1 | all 1 (23,703 of 23,703) |
| WixQA-ExpertWritten | 1.29 / 1 | article list, no grade |
| ConTEB Insurance | 1 / 1 | one gold chunk per query |

So the decider's flip condition fires on its first clause for every candidate. What the operator decides is whether to **accept
binary, sparse judgments** for the domain fit and the transfer question. If yes, the pick below is WixQA. If no, the only route
that meets every criterion and fits the domain is the in-house corpus (last section). It has no published comparator and would
take about a week of work.

What binary and sparse cost us, concretely:
- nDCG@10 still computes with binary gains. With about 1.3 relevant documents per query it behaves close to a reciprocal-rank
  measure. Graded LTR (nb 06's question) is not possible.
- "Not retrieved" and "not judged" cannot be told apart. A relevant article the annotator did not list counts as a miss. NFCorpus's
  38 judgments per query made that rare. Here it is the normal case. A run should report judged@10 (the share of its top 10 that is
  judged) beside nDCG@10.
- recall@50 has few hits to count (1 to 3 per query), so per-query recall is mostly 0 or 1.

## Candidate table

Rank of each fact: **[F]** full text read in this lane (or row-ranked [F] in `docs/literature-2026-11-16.md`), **[A]** abstract
or dataset card only, **[S]** secondary source or search snippet, **[M]** measured in this lane from the downloaded file,
**est.** an estimate scaled from the NFCorpus anchor. Embed time scales the README's "embeds in <1 min" anchor (3,633 docs,
MiniLM-L6, `max_seq_length=256`) by document count, with documents clipped at 256 tokens. These are estimates, not timings.

| corpus | docs | test queries | judgments / query | graded? | doc structure | embeds on the laptop | licence (commercial use; can we copy into a public repo?) | domain fit to the automations industries | published numbers for our arms |
|---|---|---|---|---|---|---|---|---|---|
| **NFCorpus** (anchor) | 3,633 [F] BEIR T1 | 323 [F] | 38.2 [F] | yes, 3-level [F] | title + abstract [F] | <1 min (README) | **not reported** by its authors (BEIR App. E [F]); the HF card of the BEIR packaging (`BeIR/nfcorpus`) says cc-by-sa-4.0 [A] | none: medical abstracts | BEIR T2 [F], Bruch T2 [F] (same MiniLM, RRF) |
| **FiQA-2018** (BEIR) | 57,638 [F] BEIR T1 (57,640 in the source paper [F]) | 648 [F] [M] | 2.63 [M] (2.6 [F]); **implicit**: a question's accepted answer posts, not assessor judgments (Maia et al. §2.2 [F]) | no, all 1 [M] | answer post, no title (BEIR T1 title column [F]) | ~8–15 min est. | data licence **not reported** (BEIR App. E [F]); the CC BY 4.0 on the WWW'18 paper covers the paper, not the data [F]; the posts are StackExchange content, CC BY-SA [S] | partial. A client asking a finance question resembles `accounting-client-questions` (a client mails a question, answered from the dossier). But the source is opinion posts on personal investing, not a firm's documents | **yes, for every first-stage arm [F]**: BEIR T2 nDCG@10 BM25 0.236, TAS-B 0.300, BM25+CE 0.347; T9 recall@100 BM25 0.539. **Bruch T2** (same all-MiniLM-L6-v2, RRF η = 60, stemmed BM25 k1 0.9 / b 0.4, cutoff **1000** for FiQA): Recall@1000 Lex 0.720, Sem 0.908, TM2C2 0.907, RRF 0.904; NDCG@1000 Lex 0.315, Sem 0.467, TM2C2 0.496, RRF 0.464. Sem is not marked significantly different from RRF (no ∗), so on FiQA the field's RRF-vs-dense result is itself a null |
| **CQADupStack** (BEIR) | 457,199 over 12 subforums [F]; one subforum 17,405–40,221 (webmasters, english) [S] HF size API | 13,145 [F]; per subforum 506–2,906 [M] | 1.80 pooled [M] vs **1.4 in BEIR T1** [F] (BEIR averages over subforums; the two do not match and this lane did not resolve why); median 1 [M] | no, all 1 [M] | title + body [F] | all 12: ~1.5–2 h est., fails "a laptop re-runs"; one subforum ~3–6 min est. | Apache-2.0 (BEIR App. E [F]); the content is StackExchange CC BY-SA [S] | weak. The task is "find the duplicate of this forum question". Closest is de-duplicating support tickets, which no current flow does. The queries are forum titles, not office requests | BEIR T2 nDCG@10 **mean of 12 only** [F]: BM25 0.299, TAS-B 0.314, BM25+CE 0.370; no per-subforum number at [F] |
| **ConTEB** (row 6 of `literature-2026-11-16.md`) | Insurance: 1 document in 60 chunks (T1 [F], 60 rows [M]); ESG Reports 30 docs | Insurance 120 [F] [M]; ESG Reports 36 [F] | 1 gold chunk per query [M] | no [M] | chunks of one long document [F] | seconds | HF dataset card names no licence [A] | insurance statistics per EU country, the closest *topic* to policy documents. But Insurance is a **controlled** set: questions were written by hand to be ambiguous without document structure (§3.2 "Sabotage", App. A.3 [F]). It tests chunk context, not office search | T2 nDCG@10 per model [F] (ModernBERT-large, Anthropic Contextual, BM25). Not our encoder |
| **WixQA** (Cohen et al., arXiv 2505.08643, 2025) | 6,221 help-centre articles [F] [M]: 66% articles, 32% feature requests, 1% known issues [F]; mean 2,270 chars, p90 5,452 [M] | ExpertWritten 200 [F] [M]; Simulated 200 [F] [M]; Synthetic 6,221 (GPT-4o-generated [F], not usable as a test set) | ExpertWritten 1.29 [M]; 52 of 200 queries need >1 article [M] (paper 27% [F]). Simulated 1.15 [M], 27 of 200 multi [M] (paper 14% [F]) | no: the list of article URLs the expert used for the answer [F] App. A | title + contents (+ html) [M] | ~1–2 min est.; most articles are clipped at 256 tokens (mean 2,270 chars), so nb 05's pooling question returns | **MIT**, KB and datasets [F] abstract + HF card [A]. Commercial use and copying allowed with the notice | **closest of the public sets.** Real customer queries to a company's own support knowledge base, answered by its support experts [F] §3.2. That is the `accounting-client-questions` shape (question in, answer from the firm's own documents, a human approves) and the support side of the real-estate and recruitment flows. English and SaaS, not Dutch office admin | **none for our metrics.** Tables 3–5 report GPT-4o-judged "context recall" at top-5 for BM25 and e5-large-v2 (ExpertWritten 0.73 / 0.81) [F]. That is a model-judge metric on answer coverage, not nDCG or recall over the qrels. No comparator |

Considered and dropped (rank [A]/[S], not read in full):
- **PJB** (person–job retrieval, arXiv 2603.17386, Mar 2026) [S]. Chinese, about 200k resumes, and the annotation process is unclear per a third-party page.
- **DevRev-Search** (support tickets, arXiv 2601.04646, Jan 2026) [S]. Labels come mainly from an LLM judge, so it fails "human judgments".
- **TechQA** (IBM technotes; WixQA §2 and Table 1 [F]: 600 test questions, IBM Technotes as the KB). Corpus size not checked in this lane.
- **Doc2Dial / MultiDoc2Dial** (government-service documents; WixQA §2 [F]). The queries are dialogue turns, not standalone searches.
- **Graded BEIR sets** (TREC-COVID, DBPedia, Robust04, TREC-NEWS, Touché; BEIR T1 [F]). They meet "graded", but each is 171k to 4.6M docs, and they fit the domain no better than NFCorpus.

## Power: which NFCorpus findings can each corpus resolve?

Per-query Δ nDCG@10 standard deviations come from the committed NFCorpus score files (`results/scores/*.json`, 323 queries,
computed in this lane, read-only): rerank_llm − dense 0.157, bm25 − dense 0.201, rrf − dense 0.149, query-prf − dense 0.104. With
the same spread, the paired 95% CI half-width is 1.96·sd/√n:

| finding on NFCorpus (Δ nDCG@10 vs dense) | half-width at n = 323 | at 200 (WixQA-EW) | at 648 (FiQA) | resolves on WixQA? | on FiQA? |
|---|---|---|---|---|---|
| LLM rerank +0.089 | 0.017 | 0.022 | 0.012 | yes, if the effect is ≥ about 0.03 | yes |
| BM25 −0.049 (untuned) | 0.022 | 0.028 | 0.016 | yes, if ≥ about 0.03 | yes |
| RRF −0.007 (null) | 0.016 | 0.021 | 0.012 | only as "no effect over ±0.02" | ±0.012 |
| query-PRF −0.005 (null) | 0.011 | 0.014 | 0.008 | ±0.014 | ±0.008 |

These are **lower bounds**. With about one relevant document per query, per-query nDCG@10 is close to 0 or 1/log2(rank+1), so the
spread is likely larger than NFCorpus's. A null that "transfers" at WixQA's width says less than the same null at FiQA's.
ExpertWritten and Simulated together give 400 queries (half-width about 0.016 for rerank). But the two sets differ (Simulated is
LLM-distilled from chatbot dialogues, §3.3 [F]), so a pooled run is a second analysis, not the primary one.

## Pick

**First (conditional on the operator accepting binary, sparse judgments): WixQA-ExpertWritten**, with Simulated as a pre-registered
secondary set. It is the only public set whose queries are real customers asking a company's own document collection, which is the
automations shape. It has the cleanest licence of the five (MIT), it re-runs in minutes, and it has title + body. The costs, stated
openly: n = 200; about 1.3 binary judgments per query; and **no published number for any of our arms**, so every result stands on
our own instrument. BEIR's [F] numbers cannot check our BM25 preprocessing here. The large effects (rerank, BM25) can transfer or
fail at n = 200. The two nulls can only be bounded to about ±0.02.

**Runner-up, and the better calibration corpus: FiQA-2018.** n = 648 gives CIs about half as wide. It is the one candidate with
a published [F] number for **every** first-stage arm we run, on the same encoder: Bruch Table 2 has all-MiniLM-L6-v2, stemmed BM25
and RRF η = 60 on the same 648 queries (at cutoff 1000), and BEIR Table 2 adds BM25 nDCG@10 0.236 and a cross-encoder rerank at
0.347. That is the check the NFCorpus literature read says our untuned BM25 needs, and on FiQA the field's own RRF-vs-dense result
is a null (0.464 vs 0.467 NDCG@1000), so our NFCorpus RRF null can be set against a published null. Against it: its judgments are
implicit answer-post links, its data licence is not reported, and personal-investing opinion posts are further from client office
work than a support knowledge base is.

Why not the order reversed: the reader question is C5, office work, and the operator asked for domain fit ("more applicable for
automations industries"). FiQA answers a different question better: does our harness agree with the field on a second corpus
(power, comparators for dense, BM25, RRF and a reranker). WixQA answers the operator's question, but with no comparator and at
n = 200. The trade is close. If the operator wants the calibration first, reverse the order: FiQA first, WixQA after. The
baselines brief is written so the corpus block is the only thing that changes.

**Where it runs.** This repo's own scope rule (README "Scope of the repo", 2 Nov 2026; `CLAUDE.md`: "a new dataset or modality is a
new repo that copies this skeleton") keeps a second corpus out of this tree. The baselines lane therefore bootstraps a private
sibling, `rag-wixqa-lab`, from this skeleton (the `lab-template-create` precedent, 6 Nov 2026). Running it here instead would need
a dated amendment of that rule by the operator.

## The in-house option: a fake-data office corpus

Shape: documents written from the automations demo data (fake client mails, dossier notes, listing descriptions, vacancy texts,
policy and deadline sheets, in the demo projects' Dutch/Flemish office register). Queries and **graded** judgments are written
blind by a separate session, as `as-crm-security-build` did for its held-out cases.

- **Size for a paired bootstrap to say something.** With NFCorpus as the anchor (rerank Δ sd 0.157, half-width 0.017 at n = 323),
  about 250 queries give ±0.019 and 325 give ±0.017. Judgments: pool the top 20 of dense, BM25 and RRF per query (a union of about
  40 documents) and grade every pooled document 0/1/2. At 300 queries that is about 12,000 graded pairs, NFCorpus-like density
  (38/query). Corpus: 3,000 to 5,000 short documents, so it embeds in a minute or two like NFCorpus.
- **Effort.** The automations repo holds 130 files under `projects/*/fixtures/` (sinks, replay and execute logs). They seed the
  vocabulary but are not a document collection. Generating the corpus is one session. Writing 300 queries blind is a second. Grading
  12,000 pooled pairs is the bulk: model time if a model grades, days if a human does. Estimate: 3 to 5 lanes and about a week of
  calendar time, plus a held-out split frozen before any arm runs.
- **The trap.** If a Claude session writes the graded judgments and a Claude judge reranks, that is nb 04's self-agreement problem:
  the judge agrees with its own labels and the score goes up for no real reason. It also breaks lab-template step 3 ("no model
  judge as the number"). Two fixes: (a) a human-graded subset (for example 50 queries × pooled docs) on which every headline number
  is reported, with the model-graded rest as secondary; (b) a non-Claude rerank judge for that corpus. Without one of them, the
  in-house corpus can carry first-stage results (dense, BM25, RRF) but not a reranker claim.
- **No published comparator.** Every number stands alone. It answers "does it hold on *our* office work", not "does it match the
  field".

It is not queued. It is one line in `docs/inbox/2026-11-17-in-house-office-corpus.md` (same PR) for the operator to decide.

## Sources and files

Papers read in full this lane: BEIR, arXiv 2104.08663 (Tables 1, 2, 9; App. D, E); WixQA, arXiv 2505.08643 (all sections and
appendices); Maia et al., "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering", DOI 10.1145/3184558.3192301
(both pages, Maynooth MURAL copy); ConTEB, arXiv 2505.24782 v2 (Table 1, §3.2, App. A.3, A.5; row 6 of the 16 Nov literature
read). Bruch et al. 2210.11934 (Tables 1, 2, 3 and §3.2 setup): FiQA row present (above); no CQADupStack row. PDFs are not committed.

Downloaded qrels and query files (sha256, from `huggingface.co/datasets`, fetched 2026-11-17):
- `BeIR/fiqa-qrels` `test.tsv` 6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44
- `Wix/WixQA` `wixqa_expertwritten/test.jsonl` a1bb6888a8394f980bc88f0d09b873fcd283ff4c9782600bc203f8103c2f1bad;
  `wixqa_simulated/test.jsonl` 9a2b9f42c081467b3de5bd92511e770bca3cc7977f169526fd78dd3c985ac4a8;
  `wix_kb_corpus/wix_kb_corpus.jsonl` 46b852c60b85fb1828e7c6118bb56ba796eba99120a6b69c6b984c06e1c5e1fe
- `illuin-conteb/insurance` queries parquet db1c3a6a3190ea316ea6db28ca4749a9dbac01f286c87848d9f7e214147b258d; documents parquet
  8eb2d62824b4d62c4b3fd6b937e36a9cf2143fffa19c0daedea5059081e74f15
- `BeIR/cqadupstack-qrels` `<subforum>/test.tsv`:
  android 8965c2259e76144f857a36b197f2dedf073e838c5196e6720990ba10251e322b
  english d9377c26abd502f470a5ff331e36209662fc5d091fe824201e663620b5434b6a
  gaming a39736b2be1cf8922ed3a7c56017c02a5492131cc75156d77c6c3825124ba4dc
  gis 45a27e08f2371d72dcffff27a7d5d66e282fa2f022e6f46e2b7dc3f1dbf66b2e
  mathematica 328d123124f9cffde8508a39c05954ec8dce81fd1eae8658b729022c374cd342
  physics 3550c476a608c75550dfdda2a9d9551416f514660e6268b8153131d80df393df
  programmers ed69afa873c64d34298792d694c82c4293e4b7e3c6ea3edc7a73662d4b0cba76
  stats 819b62de1d83923cf83def7f161726e7bff88f69c0840cb391b50c2f7f0f89e8
  tex ec6952e95a549782a8d3b304ef0effaf12f24c567a7cbc0ae9f6d6328a43ff04
  unix a7bd24ce081b0bb892602e9b0067017dc66fd03e7593563ad095cb3cd05e92ed
  webmasters 6b56b9ff5ee60346934d12453dab985648c01ad2b2c67abec1d225e08ecc28de
  wordpress bfd319c103eac106e054399160f874466a018b54548b6a474645ed80a2281b78
