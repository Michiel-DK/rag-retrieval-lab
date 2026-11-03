"""Where ragexp reads and writes ``results/``. One place, resolved once at import.

Order:
1. ``RAGEXP_RESULTS`` if set and non-empty.
2. The source checkout's ``results/`` when ragexp runs from it (editable install,
   or the repo on ``sys.path``). Detected by ``pyproject.toml`` next to the package.
3. ``./results`` under the current directory. A non-editable install has no
   committed results, so it gets a fresh folder where you stand, never one inside
   site-packages.

The value binds at import: set ``RAGEXP_RESULTS`` before importing ragexp.

``PREDICTIONS_DIR`` (``RAGEXP_PREDICTIONS`` > checkout ``predictions/`` > ./predictions)
holds the written-first predictions that ``ragexp run`` requires before it writes
into the checkout's committed ``results/`` (``EVIDENCE_RESULTS_DIR``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

ENV_VAR = "RAGEXP_RESULTS"
PREDICTIONS_ENV_VAR = "RAGEXP_PREDICTIONS"


def checkout_root(pkg_file: str | Path) -> Path | None:
    """The source checkout ragexp runs from, or None for an installed copy."""
    root = Path(pkg_file).resolve().parent.parent
    return root if (root / "pyproject.toml").is_file() else None


def resolve_results_dir(env: Mapping[str, str], pkg_file: str | Path, cwd: Path) -> Path:
    override = env.get(ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    checkout = checkout_root(pkg_file)
    if checkout is not None:
        return checkout / "results"
    return Path(cwd).resolve() / "results"


def resolve_predictions_dir(env: Mapping[str, str], pkg_file: str | Path, cwd: Path) -> Path:
    override = env.get(PREDICTIONS_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    checkout = checkout_root(pkg_file)
    return (checkout if checkout is not None else Path(cwd).resolve()) / "predictions"


RESULTS_DIR = resolve_results_dir(os.environ, __file__, Path.cwd())
PREDICTIONS_DIR = resolve_predictions_dir(os.environ, __file__, Path.cwd())
_CHECKOUT = checkout_root(__file__)
# Where committed evidence lives. None for an installed copy: there, every run is exploratory.
EVIDENCE_RESULTS_DIR = _CHECKOUT / "results" if _CHECKOUT is not None else None
