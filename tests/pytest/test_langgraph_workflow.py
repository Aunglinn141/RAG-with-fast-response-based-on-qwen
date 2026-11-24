"""Exercise the real compiled graph with deterministic local model/index doubles."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from threading import Barrier, Event

import pytest

from local_rag.core.cancellation import RequestCancelled
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.core.timing import record_model_call, timed_operation
from local_rag.rag.service import RagService
from local_rag.rag.workflow import RagRunContext
from tests.unit.test_rag import FakeChat, FakeIndex


@pytest.mark.parametrize("question,empty,expected", [
    ("谁在桃园结义？", False, ["prepare", "retrieve", "build_context", "generate", "finalize"]),
    ("谁在桃园结义？", True, ["prepare", "retrieve", "build_context"]),
    ("你好", True, ["prepare", "build_context", "generate", "finalize"]),
])
def test_graph_routes_knowledge_empty_context_and_greetings(question, empty, expected):
    index, chat = FakeIndex([] if empty else None), FakeChat()
    rag = RagService(index, RagSettings(), chat)
    updates = list(rag.graph.stream({"question": question}, stream_mode="updates"))
    assert [name for update in updates for name in update] == expected
    result = updates[-1][expected[-1]]["result"]
    if expected[-1] == "build_context":
        assert result["reason"] == "no_sources"
        assert not chat.calls and not chat.scope_calls and not chat.digest_calls
    else:
        assert result["status"] == "answered"
    if question == "你好":
        assert not index.calls and not chat.scope_calls
        assert result["interaction"] == "greeting" and result["sources"] == []
    if not empty:
        # Later nodes must not mutate an earlier streamed state snapshot.
        assert updates[2]["build_context"]["result"]["status"] == "insufficient_evidence"


@pytest.mark.parametrize("stop_after", [None, "retrieve", "build_context"])
def test_cancellation_at_graph_boundaries_prevents_model_calls(stop_after):
    chat, stopped = FakeChat(), Event()
    rag = RagService(FakeIndex(), RagSettings(), chat)
    if stop_after is None:
        stopped.set()
    with pytest.raises(RequestCancelled):
        with closing(rag.graph.stream({"question": "谁结义？"},
                                     context=RagRunContext(cancel=stopped), stream_mode="updates")) as updates:
            for update in updates:
                if stop_after in update:
                    stopped.set()
    assert not chat.calls and not chat.scope_calls and not chat.digest_calls


def test_graph_is_reused_without_carrying_prior_answers_sources_or_errors():
    index, chat = FakeIndex(), FakeChat()
    rag = RagService(index, RagSettings(), chat)
    graph = rag.graph
    first = rag.ask("谁结义？", model="gemma3:1b")
    chat.reply = "invalid json"
    with pytest.raises(DocumentError, match="回答格式"):
        rag.ask("谁在桃园结义？")
    index.matches = []
    last = rag.ask("文档中提到了谁？")
    assert rag.graph is graph
    assert first["status"] == "answered" and first["sources"]
    assert last["reason"] == "no_sources" and not last["sources"]
    assert last["model"] == {"name": RagSettings().model_name, "version": None}
    assert "raw_answer" not in last and "cancel" not in last and "on_event" not in last


def test_parallel_graph_runs_isolate_callbacks_cancellation_and_results():
    barrier = Barrier(2)

    class ConcurrentChat(FakeChat):
        def chat_stream(self, model, prompt, **options):
            question = json.loads(prompt)["question"]
            barrier.wait(timeout=10)
            yield '{"answerable":true,"answer":"'
            yield question
            yield '"}'

    rag = RagService(FakeIndex(), RagSettings(), ConcurrentChat())
    stopped = Event()
    cancelled_events, completed_events = [], []

    def cancel_first(name, data):
        cancelled_events.append((name, data))
        if name == "delta":
            stopped.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        cancelled = executor.submit(rag.ask, "谁结义？", model="gemma3:1b",
                                    on_event=cancel_first, cancel=stopped)
        completed = executor.submit(rag.ask, "谁在桃园结义？",
                                    on_event=lambda *e: completed_events.append(e))
        with pytest.raises(RequestCancelled):
            cancelled.result(timeout=15)
        result = completed.result(timeout=15)
    assert result["question"] == result["answer"] == "谁在桃园结义？"
    assert result["model"]["name"] == RagSettings().model_name
    assert "".join(data["text"] for name, data in completed_events if name == "delta") == result["answer"]
    assert "".join(data["text"] for name, data in cancelled_events if name == "delta") == "谁结义？"
    assert completed_events[-1] == ("stage", {"stage": "verifying"})


def test_timing_context_survives_graph_nodes_and_resets_for_each_request():
    class TimedIndex(FakeIndex):
        @timed_operation("retrieval")
        def search(self, *args, **kwargs):
            record_model_call("embedding", "bge-m3", {}, .001)
            return super().search(*args, **kwargs)

    class TimedChat(FakeChat):
        def classify(self, model, *args, **kwargs):
            record_model_call("chat", model, {}, .002)
            return super().classify(model, *args, **kwargs)

        def chat(self, model, *args, **kwargs):
            record_model_call("chat", model, {}, .003)
            return super().chat(model, *args, **kwargs)

    rag = RagService(TimedIndex(), RagSettings(), TimedChat())
    for _ in range(2):
        timings = rag.ask("谁结义？")["timings"]
        assert [call["phase"] for call in timings["model_calls"]] == ["embedding", "scope_check", "answer"]
        assert timings["retrieval_ms"] >= 0 and timings["chat_ms"] >= timings["retrieval_ms"]
        assert timings["llm_ms"] == 5 and timings["generation_ms"] == 3


def test_graph_schema_and_visualization_are_available_on_python_310():
    graph = RagService(FakeIndex(), RagSettings(), FakeChat()).graph
    assert "question" in graph.get_input_jsonschema()["properties"]
    assert "on_event" not in graph.get_input_jsonschema()["properties"]
    assert "prepare" in graph.get_graph().draw_mermaid()
