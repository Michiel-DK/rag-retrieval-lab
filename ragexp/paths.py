"""Where ragexp reads and writes ``results/``. One place, resolved once at import.

Order:
1. ``RAGEXP_RESULTS`` if set and non-empty.
2. The source checkout's ``results/`` when ragexp runs from it (editable install,
   or the repo on ``sys.path``). Detected by ``pyproject.toml`` next to the package.
3. ``./results`` under the current directory. A non-editable install has no
   committed results, so it gets a fresh folder where you stand, never one inside
   site-packages.

The value binds at import: set ``RAGEXP_RESULTS`` before importing ragexp.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

ENV_VAR = "RAGEXP_RESULTS"


def resolve_results_dir(env: Mapping[str, str], pkg_file: str | Path, cwd: Path) -> Path:
    override = env.get(ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(pkg_file).resolve().parent.parent
    if (checkout / "pyproject.toml").is_file():
        return checkout / "results"
    return Path(cwd).resolve() / "results"


RESULTS_DIR = resolve_results_dir(os.environ, __file__, Path.cwd())
