"""Embedding — turn text into vectors, once, and cache it.

One small model (`all-MiniLM-L6-v2`, 384-dim) so the whole corpus embeds in
under a minute on a laptop and the repo has no GPU dependency. The point of
the lab is the *methods*, not the leaderboard — a bigger embedder would raise
every number by a constant and change none of the lessons.

Embeddings are cached to ``results/emb/`` keyed by (model, text-hash) so
notebooks re-run in seconds after the first pass. The cache is content-
addressed: change the text or the model and you get a fresh vector, never a
stale one.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE = Path(__file__).resolve().parent.parent / "results" / "emb"


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy — importing sentence_transformers is slow
        self._cache_dir = _CACHE / model_name.replace("/", "__")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _key(self, texts: Sequence[str]) -> Path:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(b"\x00")
        for t in texts:
            h.update(t.encode())
            h.update(b"\x00")
        return self._cache_dir / f"{h.hexdigest()[:16]}.npy"

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Embed ``texts`` -> (n, dim) float32 array. Cached by content.

        ``normalize=True`` gives unit vectors, so a dot product IS cosine
        similarity — that's what ``retrieve.dense`` relies on.
        """
        texts = list(texts)
        cache_path = self._key(texts)
        if cache_path.exists():
            return np.load(cache_path)

        vecs = self._load().encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(cache_path, vecs)
        return vecs
