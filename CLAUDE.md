# rag-retrieval-lab

Retrieval methods explained by running them on NFCorpus, predictions written down first. Read
`SHOWCASE.md` for the frozen write-up and `LESSONS.md` before changing any method.

## Bearings

```sh
python -m pytest -q tests --ignore=tests/test_integration.py   # the CI gate (unit only; integration downloads NFCorpus)
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
2. `arm-lane/<slug>`: run the change through `single-lane-build`, and name that branch in the stub text so the
   builder extends it instead of inventing its own.
   `arm-plain/<slug>`: do the change in this session. Same model, same CLAUDE.md, no lane, no gate.
3. Same accept criterion in both arms: the CI gate green plus the change's own falsifier, stated in the PR body.
   One PR per change, merged on that criterion.
4. Never rename the branch. The arm is read from its prefix; a renamed branch becomes an unlabeled row.

Readout, any time: `node harness/claude/skills/backlog-scout/arm_readout.js --repos grounded-match,rag-retrieval-lab`.
This is a pilot of the mechanics at tiny n. It produces no evidence about the harness's effect and no number for
the public scoreboard.
