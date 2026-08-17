"""Unit tests for ragexp.embed — the cache contract, without loading a model."""
import numpy as np
import pytest

from ragexp.embed import Embedder


def test_cache_key_is_content_addressed():
    e = Embedder()
    k1 = e._key(["hello", "world"])
    k2 = e._key(["hello", "world"])
    k3 = e._key(["hello", "worlds"])
    assert k1 == k2
    assert k1 != k3


def test_cache_key_separator_prevents_boundary_collisions():
    e = Embedder()
    # ["ab", "c"] and ["a", "bc"] concatenate identically — the \x00
    # separator must keep their keys distinct.
    assert e._key(["ab", "c"]) != e._key(["a", "bc"])


def test_cache_key_depends_on_model_name():
    assert Embedder()._key(["x"]) != Embedder("other/model")._key(["x"])


def test_encode_reads_from_cache_without_model(tmp_path, monkeypatch):
    e = Embedder()
    monkeypatch.setattr(e, "_cache_dir", tmp_path)

    def boom():
        raise AssertionError("model should not load on a cache hit")

    monkeypatch.setattr(e, "_load", boom)

    texts = ["cached text"]
    vec = np.ones((1, 384), dtype=np.float32)
    np.save(e._key(texts), vec)

    out = e.encode(texts)
    assert np.array_equal(out, vec)


@pytest.mark.integration
def test_encode_roundtrip_with_real_model(tmp_path, monkeypatch):
    """Encodes two short strings with the real MiniLM (cached locally) and
    checks normalization + cache write."""
    e = Embedder()
    monkeypatch.setattr(e, "_cache_dir", tmp_path)

    out = e.encode(["a short sentence", "another one"])
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert e._key(["a short sentence", "another one"]).exists()
