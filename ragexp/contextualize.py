"""Contextual Retrieval context lines (nb 07), written by headless Claude Code.

Anthropic's Contextual Retrieval prepends a short LLM-written line to each
chunk, situating it in its document, before the chunk is embedded. Here one
``claude -p`` call per document sees the whole abstract and its numbered
chunks and returns one line per chunk. It runs on the local subscription
login, no API key, with a minimal system prompt, no tools and no settings, so
nothing from this repo leaks into the call.

Lines are checkpointed per document to ``results/context/<name>.json``: the
build resumes where it stopped, and nb 07 reruns from the committed file with
no LLM at all.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .data import Doc
from .retrieve import _chunks

_CONTEXT_DIR = Path(__file__).resolve().parent.parent / "results" / "context"

_SYSTEM = "You write short retrieval context for document chunks. Reply with JSON only."

# The per-chunk wording is the post's; asking for every chunk of a document
# in one reply is this lab's change (one call per document, not per chunk).
_PROMPT = """<document>
{document}
</document>

The document above is split into {n} numbered chunks:

{chunks}

For each chunk, give a short succinct context to situate this chunk within the overall \
document for the purposes of improving search retrieval of the chunk. Answer only with a JSON \
object mapping each chunk number to its context, like {{"1": "...", "2": "..."}}, and nothing else."""


class ClaudeCLIContextualizer:
    """One ``claude -p`` call per document -> one context line per chunk."""

    def __init__(self, model: str = "haiku", timeout: int = 300):
        self.model = model
        self.timeout = timeout
        self._cwd = tempfile.mkdtemp(prefix="ragexp-context-")  # no CLAUDE.md here

    def contexts(self, doc: Doc) -> tuple[List[str], str]:
        """Return (one line per chunk of ``doc.text``, the model id that wrote them)."""
        chunks = _chunks(doc.text)
        listing = "\n\n".join(f"<chunk {i}>\n{c}\n</chunk {i}>" for i, c in enumerate(chunks, start=1))
        prompt = _PROMPT.format(document=doc.full, n=len(chunks), chunks=listing)
        raw, model_id = self._call(prompt)
        return parse_contexts(raw, n=len(chunks)), model_id

    def _call(self, prompt: str) -> tuple[str, str]:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model, "--output-format", "json",
             "--system-prompt", _SYSTEM, "--tools", "", "--strict-mcp-config",
             "--setting-sources", "", "--no-session-persistence"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self._cwd,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {proc.stderr[:500]}")
        out = json.loads(proc.stdout)
        if out.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(out.get('result'))[:500]}")
        return out["result"], ",".join(sorted(out.get("modelUsage") or {})) or "unknown"


def parse_contexts(raw: str, n: int) -> List[str]:
    """Parse a ``{"1": "...", ..., "n": "..."}`` reply into n one-line strings."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"no JSON object in reply: {raw[:200]}")
    parsed = {int(k): v for k, v in json.loads(raw[start : end + 1]).items()}
    missing = set(range(1, n + 1)) - set(parsed)
    if missing:
        raise ValueError(f"reply missing chunks {sorted(missing)[:5]}")
    lines = [" ".join(str(parsed[i]).split()) for i in range(1, n + 1)]
    if not all(lines):
        raise ValueError("empty context line in reply")
    return lines


def load_contexts(name: str) -> Dict[str, Optional[dict]]:
    """Read a committed context file: {doc_id: {"contexts": [...], "model": id} or None}."""
    path = _CONTEXT_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_contexts(
    name: str,
    docs: Sequence[Doc],
    contextualizer,
    workers: int = 8,
    retries: int = 2,
    verbose: bool = True,
) -> Dict[str, Optional[dict]]:
    """Write context lines for every doc not yet in ``results/context/<name>.json``.

    A doc whose call or parse still fails after ``retries`` retries is stored as
    ``None``: it keeps its plain chunks in nb 07, and the notebook counts them.
    Already-stored docs, ``None`` included, are never re-run.
    """
    _CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    path = _CONTEXT_DIR / f"{name}.json"
    done = load_contexts(name)
    todo = [d for d in docs if d.doc_id not in done]
    lock = threading.Lock()

    def one(doc: Doc) -> Optional[dict]:
        for _ in range(retries + 1):
            try:
                lines, model_id = contextualizer.contexts(doc)
                return {"contexts": lines, "model": model_id}
            except (RuntimeError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError):
                continue
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            with lock:
                done[futures[fut].doc_id] = fut.result()
                if i % 50 == 0 or i == len(todo):
                    path.write_text(json.dumps(done, indent=0, sort_keys=True))
            if verbose and (i % 100 == 0 or i == len(todo)):
                print(f"[{len(done)}/{len(docs)}] docs with context", flush=True)
    path.write_text(json.dumps(done, indent=0, sort_keys=True))
    return done
