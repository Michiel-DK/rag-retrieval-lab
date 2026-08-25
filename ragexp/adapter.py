"""Linear adapter — the trained alternative to a fixed representation (nb 05).

Ported from the source system's ``embed_adapter_train.py``. A single dense
matrix ``W`` (dim x dim) sits on the query side: a query scores a document as
``(W q) . d`` instead of ``q . d``. Initialised to the identity, so ``W = I``
reproduces the un-adapted ranking exactly, and any gain has to come from
training rather than from a different pipeline.

Training is exact InfoNCE: the softmax denominator runs over the *whole*
corpus, not a sampled batch of negatives, which is affordable here because the
corpus is 3,633 documents. Weight decay pulls ``W`` back toward ``I``, so the
optimiser has to earn every step away from the free baseline.

The gradient is written out by hand. There is no autograd and no GPU, so the
notebook reproduces on any machine.

Notation, with ``N`` training pairs, ``n`` documents and ``dim`` dimensions:

    logits[i, j] = (W q_i) . d_j      shape (N, n)
    L            = cross-entropy(logits, positive doc of pair i)
                   + 0.5 * wd * ||W - I||^2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TrainLog:
    """One epoch of training telemetry. Train-side only, by design.

    Hyperparameters get fixed from these numbers. The eval split is scored
    once, at the end, so it cannot leak back into the choice of ``lr`` or
    ``epochs``.
    """

    epoch: int
    loss: float
    acc_at_1: float
    drift: float  # ||W - I||, Frobenius


def train_adapter(
    queries: np.ndarray,
    doc_matrix: np.ndarray,
    targets: Sequence[int],
    epochs: int = 60,
    lr: float = 0.5,
    wd: float = 3e-2,
    dtype=np.float32,
) -> Tuple[np.ndarray, List[TrainLog]]:
    """Fit the query-side adapter ``W`` by full-batch InfoNCE.

    ``queries`` is (N, dim) unit query vectors, one row per (query, positive
    document) pair. ``doc_matrix`` is (n, dim) unit document vectors.
    ``targets[i]`` is the row of ``doc_matrix`` that pair ``i`` should rank
    first.

    Returns ``(W, history)``. ``W`` has the same shape as ``I`` and is applied
    as ``queries @ W.T``.
    """
    q = np.ascontiguousarray(queries, dtype=dtype)
    d = np.ascontiguousarray(doc_matrix, dtype=dtype)
    t = np.asarray(targets, dtype=np.int64)
    if q.ndim != 2 or d.ndim != 2:
        raise ValueError("queries and doc_matrix must both be 2-D")
    if q.shape[1] != d.shape[1]:
        raise ValueError(f"dim mismatch: queries {q.shape[1]} vs docs {d.shape[1]}")
    if t.shape[0] != q.shape[0]:
        raise ValueError(f"targets has {t.shape[0]} rows, queries has {q.shape[0]}")
    if t.size and (t.min() < 0 or t.max() >= d.shape[0]):
        raise ValueError("targets index outside doc_matrix")

    dim = q.shape[1]
    n_pairs = q.shape[0]
    eye = np.eye(dim, dtype=dtype)
    w = eye.copy()
    rows = np.arange(n_pairs)
    history: List[TrainLog] = []

    for epoch in range(1, epochs + 1):
        logits = (q @ w.T) @ d.T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)

        penalty = 0.5 * wd * float(((w - eye) ** 2).sum())
        loss = float(-np.log(probs[rows, t] + 1e-12).mean()) + penalty
        acc = float((probs.argmax(axis=1) == t).mean())
        history.append(
            TrainLog(epoch, loss, acc, float(np.linalg.norm(w - eye)))
        )

        grad_logits = probs
        grad_logits[rows, t] -= 1.0
        grad_logits /= n_pairs
        grad_w = (q.T @ grad_logits @ d).T + wd * (w - eye)
        w -= lr * grad_w.astype(dtype)

    return w, history


def apply_adapter(w: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Map query vectors through ``W`` and re-normalise to unit length.

    Re-normalising keeps the dot product a cosine, which is what
    ``ragexp.retrieve.dense`` assumes. It also makes ``W = I`` a no-op on
    already-unit queries, so the identity check is exact.
    """
    out = np.atleast_2d(np.asarray(queries, dtype=np.float32)) @ np.asarray(
        w, dtype=np.float32
    ).T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-8, None)
