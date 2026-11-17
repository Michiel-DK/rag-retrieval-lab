"""Lane query-rewrite: one model-written passage per query (HyDE / Query2doc shape).

One ``claude -p`` call per NFCorpus test query writes a short "scientific paper
passage" that answers it (HyDE 2212.10496, A.1.4 instruction; Query2doc 2303.07678,
App. A anti-clarification system line). The route is nb 07's (``contextualize.py``):
subscription login, no API key, no tools, no settings, an empty working directory,
thinking off. The prompt below is the one committed in ``predictions/query-rewrite.md``.

Passages are checkpointed to ``results/query-rewrite/generations.json`` with the
per-call model id, token counts and the CLI's list-price cost. The file is committed
once and never regenerated; the ``dense-hyde`` / ``dense-q2d`` methods read only it,
pinned by sha256, so ``ragexp check`` re-derives every number with no model call.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

from .paths import RESULTS_DIR

GENERATIONS_FILE = RESULTS_DIR / "query-rewrite" / "generations.json"

SYSTEM = ("You are asked to write a passage that answers the given query. "
          "Do not ask the user for further clarification.")

PROMPT = """Please write a scientific paper passage to answer the question
Question: {query}
Passage:

Reply with the passage only: one paragraph of at most 100 words."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_generations(path: Path = GENERATIONS_FILE, sha256: Optional[str] = None) -> Dict[str, Optional[str]]:
    """query_id -> passage (``None`` for a failed call). Raises if ``sha256`` is given and differs."""
    path = Path(path)
    if sha256 is not None:
        got = sha256_file(path)
        if got != sha256:
            raise ValueError(f"{path} sha256 {got} != pinned {sha256}: the generations file changed")
    data = json.loads(path.read_text())
    return {q: (g["passage"] if g else None) for q, g in data["generations"].items()}


class ClaudeCLIRewriter:
    """One ``claude -p`` call per query -> (passage, record of model, tokens and cost)."""

    def __init__(self, model: str = "haiku", timeout: int = 300):
        self.model = model
        self.timeout = timeout
        self._cwd = tempfile.mkdtemp(prefix="ragexp-rewrite-")  # no CLAUDE.md here

    def passage(self, query: str) -> tuple[str, dict]:
        proc = subprocess.run(
            ["claude", "-p", PROMPT.format(query=query), "--model", self.model, "--output-format", "json",
             "--system-prompt", SYSTEM, "--tools", "", "--strict-mcp-config",
             "--setting-sources", "", "--no-session-persistence"],
            capture_output=True, text=True, timeout=self.timeout, cwd=self._cwd,
            env={**os.environ, "MAX_THINKING_TOKENS": "0"},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {proc.stderr[:500]}")
        out = json.loads(proc.stdout)
        if out.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(out.get('result'))[:500]}")
        text = " ".join(str(out.get("result") or "").split())
        u = out.get("usage") or {}
        record = {
            "model": ",".join(sorted(out.get("modelUsage") or {})) or "unknown",
            "input_tokens": u.get("input_tokens", 0),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
            "cost_usd": float(out.get("total_cost_usd") or 0.0),
        }
        return text, record  # an empty text is the builder's to retry, so its cost still counts


class BudgetExceeded(RuntimeError):
    """The next call could cross the cost ceiling; the lane HOLDs."""


def build_generations(
    queries: Sequence[tuple[str, str]],
    rewriter,
    path: Path = GENERATIONS_FILE,
    ceiling_usd: float = 2.0,
    retries: int = 2,
    verbose: bool = True,
) -> dict:
    """Write one passage per (query_id, text) not yet in ``path``. Sequential, checkpointed per call.

    Every attempt that returns counts toward ``ceiling_usd``, empty replies included (a non-zero
    exit reports no cost and is not counted). Before each
    attempt the guard requires spent + 3 x (the costliest call so far) to stay under the
    ceiling, else it saves and raises ``BudgetExceeded``. A query whose attempts all fail
    is stored as ``None``. Stored queries, ``None`` included, are never re-run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {
        "meta": {"system": SYSTEM, "prompt": PROMPT, "model_alias": getattr(rewriter, "model", None),
                 "route": "claude -p, subscription auth, MAX_THINKING_TOKENS=0",
                 "cost_basis": "total_cost_usd from claude -p (costBasis: list)"},
        "spent_usd": 0.0, "attempts": 0, "generations": {},
    }
    worst = max([0.0] + [g["cost_usd"] for g in data["generations"].values() if g])

    def save():
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")

    for i, (qid, text) in enumerate(queries, start=1):
        if qid in data["generations"]:
            continue
        result = None
        for _ in range(retries + 1):
            if data["spent_usd"] + 3 * max(worst, 0.001) > ceiling_usd:
                save()
                raise BudgetExceeded(f"spent ${data['spent_usd']:.4f}; next call could cross ${ceiling_usd}")
            data["attempts"] += 1
            try:
                passage, rec = rewriter.passage(text)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError):
                continue
            data["spent_usd"] += rec["cost_usd"]
            worst = max(worst, rec["cost_usd"])
            if passage:
                result = {"passage": passage, **rec}
                break
        data["generations"][qid] = result
        save()
        if verbose and (i % 25 == 0 or i == len(queries)):
            print(f"[{len(data['generations'])}/{len(queries)}] spent ${data['spent_usd']:.4f}", flush=True)
    save()
    return data
