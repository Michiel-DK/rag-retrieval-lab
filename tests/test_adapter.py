"""Unit tests for ragexp.adapter — synthetic inputs, no dataset, no model."""
import numpy as np
import pytest

from ragexp.adapter import apply_adapter, train_adapter


def unit(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def test_identity_init_is_a_no_op():
    """W = I must reproduce the un-adapted query exactly. nb 05's self-check
    rests on this: any adapter gain has to come from training."""
    q = unit(np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]]))
    out = apply_adapter(np.eye(3, dtype=np.float32), q)
    assert np.allclose(out, q, atol=1e-6)


def test_apply_adapter_returns_unit_vectors():
    q = unit(np.array([[1.0, 0.0, 0.0]]))
    w = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = apply_adapter(w, q)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-6)


def test_zero_epochs_returns_identity():
    q = unit(np.array([[1.0, 0.0]]))
    d = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    w, history = train_adapter(q, d, [0], epochs=0)
    assert history == []
    assert np.allclose(w, np.eye(2), atol=1e-7)


def test_training_drives_loss_down_on_a_learnable_map():
    """Queries live in a rotated copy of the doc space. The identity scores
    the wrong doc first; a linear map can undo the rotation."""
    d = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    theta = np.pi / 2
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    q = unit(d @ rot.T)  # query i is doc i, rotated 90 degrees
    w, history = train_adapter(q, d, [0, 1], epochs=80, lr=1.0, wd=0.0)
    assert history[-1].loss < history[0].loss
    assert history[-1].acc_at_1 == 1.0
    assert history[-1].drift > history[0].drift


def test_weight_decay_holds_w_near_identity():
    d = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    q = unit(np.array([[0.0, 1.0], [1.0, 0.0]]))
    _, loose = train_adapter(q, d, [0, 1], epochs=30, lr=0.5, wd=0.0)
    _, tight = train_adapter(q, d, [0, 1], epochs=30, lr=0.5, wd=1.0)
    assert tight[-1].drift < loose[-1].drift


def test_rejects_mismatched_dimensions():
    q = np.zeros((2, 3), dtype=np.float32)
    d = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="dim mismatch"):
        train_adapter(q, d, [0, 1])


def test_rejects_target_outside_corpus():
    q = unit(np.array([[1.0, 0.0]]))
    d = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="outside doc_matrix"):
        train_adapter(q, d, [7])


def test_rejects_target_count_mismatch():
    q = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    d = unit(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="targets has"):
        train_adapter(q, d, [0])
