"""ragexp.rewrite: the generation builder with a fake rewriter, no subprocess."""
import json

import pytest

from ragexp.rewrite import PROMPT, SYSTEM, BudgetExceeded, build_generations, load_generations, sha256_file


class FakeRewriter:
    model = "fake"

    def __init__(self, replies, cost=0.01):
        self.replies = dict(replies)  # query text -> list of replies (str, "" = empty, Exception)
        self.cost = cost
        self.calls = []

    def passage(self, text):
        self.calls.append(text)
        r = self.replies[text].pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"model": "fake-1", "input_tokens": 1, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0, "output_tokens": 2, "thinking_tokens": 0, "cost_usd": self.cost}


def test_builder_retries_empty_and_errors_stores_none_and_resumes(tmp_path):
    path = tmp_path / "g.json"
    fake = FakeRewriter({"a": ["", "passage a"], "b": [RuntimeError("x")] * 3})
    data = build_generations([("A", "a"), ("B", "b")], fake, path=path, verbose=False)
    assert data["generations"]["A"]["passage"] == "passage a"
    assert data["generations"]["B"] is None
    assert data["attempts"] == 5 and data["spent_usd"] == pytest.approx(0.02)  # the empty reply's cost counts
    assert data["meta"]["prompt"] == PROMPT and data["meta"]["system"] == SYSTEM
    again = FakeRewriter({})
    build_generations([("A", "a"), ("B", "b")], again, path=path, verbose=False)
    assert again.calls == []  # stored queries, None included, are never re-run
    assert load_generations(path) == {"A": "passage a", "B": None}
    assert load_generations(path, sha256=sha256_file(path))["A"] == "passage a"
    with pytest.raises(ValueError, match="changed"):
        load_generations(path, sha256="0" * 64)


def test_builder_stops_before_crossing_the_ceiling(tmp_path):
    path = tmp_path / "g.json"
    fake = FakeRewriter({t: ["p"] for t in "abcdef"}, cost=0.1)
    with pytest.raises(BudgetExceeded):
        build_generations([(t.upper(), t) for t in "abcdef"], fake, path=path, ceiling_usd=0.55, verbose=False)
    saved = json.loads(path.read_text())
    assert len(saved["generations"]) == 3  # before call 3: 0.2 + 3 x 0.1 = 0.5 <= 0.55; before call 4: 0.6 > 0.55
    assert saved["spent_usd"] == pytest.approx(0.3)
