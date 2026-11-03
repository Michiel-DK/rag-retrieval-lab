"""Predict first, enforced: a run that writes evidence needs a committed prediction.

``predictions/<run name>.md`` must exist, contain a non-empty ``## Prediction``
and ``## Mechanism`` section, be tracked by git, and have no uncommitted edits.
A file committed before the run is by construction written before the result.
The commit that holds it is returned so the run can record it.

The gate is for runs that write into the checkout's committed ``results/``.
A run redirected elsewhere (``RAGEXP_RESULTS``) is exploratory: not gated,
and not evidence.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .paths import PREDICTIONS_DIR

REQUIRED_SECTIONS = ("Prediction", "Mechanism")


class PredictionMissing(ValueError):
    """No usable committed prediction for this run. Maps to exit 2 in the CLI."""


@dataclass(frozen=True)
class PredictionRef:
    path: Path
    commit: str  # last commit touching the file


def sections(text: str) -> Dict[str, str]:
    """``## Heading`` -> body text (stripped), for every level-2 heading."""
    out: Dict[str, str] = {}
    parts = re.split(r"^##[ \t]+(.+?)[ \t]*$", text, flags=re.MULTILINE)
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1].strip()
    return out


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def require_prediction(name: str, predictions_dir: Path = PREDICTIONS_DIR) -> PredictionRef:
    path = predictions_dir / f"{name}.md"
    hint = f"write {path} with '## Prediction' and '## Mechanism', commit it, then run"
    if not path.is_file():
        raise PredictionMissing(f"no prediction for {name!r}: {hint}")
    body = sections(path.read_text())
    empty = [s for s in REQUIRED_SECTIONS if not body.get(s)]
    if empty:
        raise PredictionMissing(f"{path}: empty or missing section(s) {', '.join(empty)}: {hint}")
    try:
        tracked = _git(path.parent, "ls-files", "--error-unmatch", path.name)
    except FileNotFoundError:
        raise PredictionMissing("git not found: cannot verify the prediction was committed first") from None
    if tracked.returncode != 0:
        raise PredictionMissing(f"{path} is not committed: {hint}")
    if _git(path.parent, "diff", "--quiet", "HEAD", "--", path.name).returncode != 0:
        raise PredictionMissing(f"{path} has uncommitted edits; commit the prediction as it stands before running")
    commit = _git(path.parent, "log", "-1", "--format=%H", "--", path.name).stdout.strip()
    if not commit:
        raise PredictionMissing(f"{path}: no commit found for the prediction")
    return PredictionRef(path, commit)
