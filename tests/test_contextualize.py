"""Unit tests for ragexp.contextualize — fake contextualizer, no subprocess."""
import json

import pytest

import ragexp.contextualize as cx
from ragexp.contextualize import build_contexts, parse_contexts
from ragexp.data import Doc


def test_parse_contexts_orders_by_chunk_number_and_flattens_whitespace():
    raw = 'Here you go:\n{"2": "second\\n line", "1": "first"}'
    assert parse_contexts(raw, n=2) == ["first", "second line"]


def test_parse_contexts_rejects_missing_or_empty_chunks():
    with pytest.raises(ValueError):
        parse_contexts('{"1": "only one"}', n=2)
    with pytest.raises(ValueError):
        parse_contexts('{"1": "  "}', n=1)
    with pytest.raises(ValueError):
        parse_contexts("no json here", n=1)


class FakeContextualizer:
    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.calls = []

    def contexts(self, doc):
        self.calls.append(doc.doc_id)
        if doc.doc_id in self.fail_ids:
            raise ValueError("unparseable")
        return [f"ctx {doc.doc_id}"], "fake-model"


def test_build_contexts_checkpoints_failures_as_none_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(cx, "_CONTEXT_DIR", tmp_path)
    docs = [Doc("a", "A", "x"), Doc("b", "B", "y")]
    fake = FakeContextualizer(fail_ids={"b"})
    out = build_contexts("t", docs, fake, workers=1, retries=2, verbose=False)
    assert out == {"a": {"contexts": ["ctx a"], "model": "fake-model"}, "b": None}
    assert fake.calls.count("b") == 3  # one try plus two retries
    assert json.loads((tmp_path / "t.json").read_text()) == out

    again = FakeContextualizer()
    build_contexts("t", docs, again, workers=1, verbose=False)
    assert again.calls == []  # stored docs, None included, are never re-run
