"""Run manifest: what produced a results/scores file, written next to it by ``ragexp run``.

``results/manifests/<name>.json`` records the registry key and constructor
params (enough for ``ragexp check`` to rebuild the run), the repo commit and
whether the tree was dirty, the prediction commit (null = exploratory),
package versions, and the sha256 of the scores file it describes.

Only NEW runs get one. The score files from notebooks 01 to 07 are committed
evidence and are not rewritten; ``check`` falls back to registry defaults for them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from . import runs
from .methods import Method, describe
from .predictions import PredictionRef

SCHEMA = 1
DEFAULT_RECALL_KS = (10, 50)  # what every run scored before "eval" existed
DEFAULT_NDCG_KS = (10,)
PACKAGES = ("ragexp", "numpy", "rank-bm25", "sentence-transformers", "torch", "datasets")


def manifest_dir() -> Path:
    return runs._SCORES.parent / "manifests"  # follows _SCORES, so RAGEXP_RESULTS and test patches carry over


def manifest_path(name: str) -> Path:
    return manifest_dir() / f"{name}.json"


def init_params(method: Method) -> Dict[str, object]:
    """Constructor arguments that rebuild ``method`` (dataclass fields minus ``name``)."""
    if not dataclasses.is_dataclass(method):
        raise ValueError(f"{type(method).__name__} is not a dataclass; its run cannot be recorded for rebuild")
    return {f.name: getattr(method, f.name) for f in dataclasses.fields(method) if f.name != "name"}


def _versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _git_state() -> Dict[str, object]:
    cwd = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        return {"commit": None, "dirty": None}
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(method_key: str, method: Method, scores_path: Path, prediction: Optional[PredictionRef],
                   n_queries: int, means: Dict[str, Optional[float]],
                   recall_ks: Sequence[int] = DEFAULT_RECALL_KS, ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS) -> Dict[str, object]:
    return {
        "schema": SCHEMA,
        "name": method.name,
        "method": method_key,
        "init_params": init_params(method),
        "eval": {"recall_ks": list(recall_ks), "ndcg_ks": list(ndcg_ks)},  # the scoring call, not constructor args
        "describe": describe(method),
        "scores_file": scores_path.name,
        "scores_sha256": sha256(scores_path),
        "n_queries": n_queries,
        "means": means,
        "exploratory": prediction is None,
        "prediction": None if prediction is None else {"file": prediction.path.name, "commit": prediction.commit},
        "repo": _git_state(),
        "versions": _versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(manifest: Dict[str, object]) -> Path:
    path = manifest_path(str(manifest["name"]))
    if path.exists():
        raise FileExistsError(f"{path} exists; manifests are never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


def eval_ks(m: Optional[Dict[str, object]]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """(recall_ks, ndcg_ks) a run was scored with; manifests without "eval" and notebook-era files get the defaults."""
    e = (m or {}).get("eval")
    if not e:
        return DEFAULT_RECALL_KS, DEFAULT_NDCG_KS
    return tuple(e["recall_ks"]), tuple(e["ndcg_ks"])


def load_manifest(name: str) -> Optional[Dict[str, object]]:
    path = manifest_path(name)
    return json.loads(path.read_text()) if path.is_file() else None
