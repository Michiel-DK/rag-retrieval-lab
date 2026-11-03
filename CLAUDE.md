# rag-retrieval-lab

Retrieval methods explained by running them on NFCorpus, predictions written down first. Read
`SHOWCASE.md` for the write-up of notebooks 01 to 04 and `LESSONS.md` before changing any method. One dataset per lab repo: a new dataset or modality is a new repo that copies this skeleton (README "Scope of the repo"), never a folder here.

## Bearings

```sh
python -m pytest -q tests --ignore=tests/test_integration.py   # the CI gate (unit only; integration downloads NFCorpus)
ragexp diff <a> <b>        # paired bootstrap on committed scores, no corpus needed
ragexp check --all         # recompute registered + manifested runs vs results/ (exit 1 on any drift)
ragexp run <method> [--name N --param k=v]   # new run; needs committed predictions/<name>.md; never overwrites
RAGEXP_RESULTS=<scratch> ragexp run ...      # exploratory: ungated, not evidence
# internal dev harness (private): checked out locally at harness/, gitignored, not part of this tree
```

## Lane args

Pass these explicitly to `single-lane-build`. A declaration here does nothing on its own; the workflow never reads
`CLAUDE.md`.

```yaml
sensitivePaths:
  full:
    - 'ragexp/metrics\.py$'
repoInvariants:
  - 'recall@50 is frozen under any reranker: a permutation of a 50-doc pool cannot change what is in the pool — a
     reranker that moves it is a broken harness (notebook 04 self-check), never a result'
  - 'results/*.json are committed evidence; a lane reads them and never regenerates them (re-embedding is a local,
     operator-run step)'
  - 'CI runs unit tests only; a change that needs the integration suite says so in its PR and the operator runs it'
repoReviewers: []
proceedAfterDiagnosis: '<name a test that is RED right now>'
```

## Arm protocol — control-arm pilot (from 2026-10-29, harness stub S-171)

Every change starts with a coin flip, recorded BEFORE any work:

1. `harness/claude/skills/backlog-scout/arm_flip.sh rag-retrieval-lab <slug>` → prints the branch. Create it:
   `git checkout -b <printed>`.
2. `arm-lane/<slug>`: run the change through `single-lane-build` with the lane arg `branch: "arm-lane/<slug>"`
   (harness S-172, validated inside the router) — the builder must create exactly that branch; naming it in
   the stub text alone is not enough, one builder invented its own on 2026-10-30.
   `arm-plain/<slug>`: do the change in this session. Same model, same CLAUDE.md, no lane, no gate.
3. Same accept criterion in both arms: the CI gate green plus the change's own falsifier, stated in the PR body.
   One PR per change, merged on that criterion.
4. Never rename the branch. The arm is read from its prefix; a renamed branch becomes an unlabeled row.
5. Every change flips — fix/, docs/, one-line README edits too. 6 of pierre's first 10 PRs skipped the flip and
   are unlabeled rows forever. The pre-push hook refuses a push from any branch that is not `arm-lane/*`,
   `arm-plain/*` or master; a deliberate skip is `ARM_SKIP="<reason>" git push`, which records `arm:none`.

One-time setup per clone (NOT committed — `.git/hooks` is local; redo it on a fresh clone):
`ln -sf "$PWD/harness/claude/skills/backlog-scout/arm_guard.sh" .git/hooks/pre-push`
Verify it bites before trusting it:
`printf 'refs/heads/fix/x 1111111111111111111111111111111111111111 refs/heads/fix/x 0000000000000000000000000000000000000000\n' | .git/hooks/pre-push origin x` → must exit 1.

Readout, any time: `node harness/claude/skills/backlog-scout/arm_readout.js --repos pierre,rag-retrieval-lab`.
Its reverted / fix-forward / tokens columns stay `n/a` until the harness's Monday backfills have run over the
window; `added_code` ignores notebooks, `added` does not — this repo tracks 7, so read `added_code`.
This is a pilot of the mechanics at tiny n. It produces no evidence about the harness's effect and no number for
the public scoreboard.

### Starting the next lab repo as a pilot arm (what this repo got wrong, in order)
1. Install the pre-push hook BEFORE the first commit, then flip every change — the unlabeled rows here are the
   changes made before the protocol existed plus the ones that skipped it.
2. Do not track notebooks; keep exploration in a gitignored scratch dir. Tracked notebooks are why the plain
   arm's median added lines read ~7x the lane's here (390 vs 57) and the size column is uninterpretable.
3. Pre-register the outcome before the first flip. "Merged" cannot be it (the operator merges both arms).
   Candidates that can move in an experiments repo: CI red at merge, refuter kills before merge, cost per merged
   change, and a committed `results/*.json` later regenerated with a different number.
4. Target ≥50 flipped changes per arm (this repo did ~27 merges in its first 5 days, so that is weeks, not months).
5. Copy the skeleton, not a shared library: `<pkg>/` (data, metrics, methods), `experiments/run_<method>.py`,
   committed `results/`, a chart script, unit tests as the CI gate, lane args + this section in `CLAUDE.md`.
   Shared tooling (results schema, predictions-first README, lane args) is promoted into the harness as a
   template only once this and the new repo both carry it — two consumers, not one.
