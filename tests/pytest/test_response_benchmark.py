import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from local_rag.cli.benchmark_response import measure, summarize


def event(name, data):
    return f"event: {name}\ndata: {json.dumps({'success': True, 'data': data})}\n\n".encode()


def test_first_answer_excludes_stage_events_and_reset_drafts():
    wire = (event("stage", {"stage": "retrieving"}) + event("delta", {"text": "discarded"})
            + event("reset", {}) + event("delta", {"text": "final"})
            + event("result", {"status": "answered", "answer": "final", "interaction": "knowledge"}))
    api = SimpleNamespace(url="http://127.0.0.1:8000", opener=SimpleNamespace(open=Mock(return_value=io.BytesIO(wire))))
    with patch("local_rag.cli.benchmark_response.perf_counter", side_effect=[0] + [x / 100 for x in range(1, 100)]):
        result = measure(api, "question")
    assert result["resets"] == 1
    assert 100 < result["first_answer_ms"] < result["complete_ms"]
    assert result["eligible"]


def test_refusals_greetings_errors_and_target_misses_do_not_pass():
    valid = {"eligible": True, "first_answer_ms": 900, "complete_ms": 1600}
    assert summarize([valid], 1000, "first_answer_ms")["passed"]
    assert not summarize([valid], 1000, "complete_ms")["passed"]
    assert not summarize([valid, {**valid, "eligible": False}], 1000, "first_answer_ms")["passed"]
    assert not summarize([], 1000, "first_answer_ms")["passed"]
    for status, interaction in [("insufficient_evidence", "knowledge"), ("answered", "greeting")]:
        wire = event("result", {"status": status, "answer": "text", "interaction": interaction})
        api = SimpleNamespace(url="http://127.0.0.1:8000", opener=SimpleNamespace(open=Mock(return_value=io.BytesIO(wire))))
        assert not measure(api, "question")["eligible"]
    api.opener.open.return_value = io.BytesIO(event("stage", {}))
    with pytest.raises(RuntimeError, match="without a final"):
        measure(api, "question")
