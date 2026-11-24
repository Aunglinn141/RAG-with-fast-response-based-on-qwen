"""Streaming contract checks, with real loopback transport and no model weights."""

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from local_rag.api.errors import register_error_handlers
from local_rag.api.limits import HeavyRequestMiddleware, RequestLimiter, is_heavy_request
from local_rag.api.routes.ask import create_router
from local_rag.clients.ollama import OllamaClient, OllamaError, OllamaOutputLimitError
from local_rag.core.cancellation import RequestCancelled
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.rag.service import RagService
from local_rag.rag.streaming import AnswerPreview
from tests.unit.test_rag import FakeChat, FakeIndex, QUOTE


@pytest.mark.parametrize("answer", [
    "中文回答。", '带引号"内容"和\\路径', "第一行\n第二行\t内容", "图书📚已归还。",
])
@pytest.mark.parametrize("ascii_escaped", [False, True])
def test_answer_preview_decodes_arbitrary_single_character_fragments(answer, ascii_escaped):
    raw = json.dumps({"answerable": True, "answer": answer, "evidence": [{"quote": "不得泄露原文"}]},
                     ensure_ascii=ascii_escaped)
    preview = AnswerPreview()
    output = "".join(preview.feed(char) for char in raw)
    assert output == answer
    assert preview.raw == raw
    assert "不得泄露原文" not in output


def test_nested_answer_and_key_text_are_not_streamed():
    raw = json.dumps({"evidence": [{"answer": "private"}], "extra": 'text "answer": "fake"',
                      "answer": "正确答案"}, ensure_ascii=False)
    preview = AnswerPreview()
    assert "".join(preview.feed(c) for c in raw) == "正确答案"


@pytest.mark.parametrize("raw", ['{"answer":123}', '{"answer":"\\x"}', '<think>secret</think>',
                                 '{"answer":"\\udc00"}', '{"answer":"' + 'x' * 3001])
def test_invalid_preview_cannot_leak_raw_protocol(raw):
    with pytest.raises(DocumentError) as error:
        AnswerPreview().feed(raw)
    assert error.value.code == "invalid_answer"


class StreamChat(FakeChat):
    def __init__(self, replies=None):
        super().__init__()
        self.replies = replies or [json.dumps(self.reply, ensure_ascii=False)]
        self.attempts = 0
        self.closed = 0

    def chat_stream(self, model, prompt, *, cancel=None, **options):
        self.calls.append({"model": model, "prompt": prompt, **options})
        reply = self.replies[self.attempts]
        self.attempts += 1
        try:
            if isinstance(reply, Exception):
                yield '{"answer":"未完成'
                raise reply
            for char in reply:
                if cancel is not None and cancel.is_set():
                    raise RequestCancelled()
                yield char
        finally:
            self.closed += 1


def test_real_deltas_arrive_before_finalization_and_final_result_matches_sync():
    chat = StreamChat()
    events = []
    result = RagService(FakeIndex(), RagSettings(), chat).ask("谁结义？", on_event=lambda *e: events.append(e))
    assert result["status"] == "answered"
    assert "".join(data["text"] for name, data in events if name == "delta") == result["answer"]
    assert [data["stage"] for name, data in events if name == "stage"] == ["retrieving", "generating", "verifying"]
    assert events[-1] == ("stage", {"stage": "verifying"})
    assert chat.closed == 1


def test_model_abstention_replaces_incomplete_draft():
    reply = json.dumps({"answerable": False, "answer": "生成中的草稿", "evidence": [{"source_id": "S1", "quote": "伪造引用"}]})
    events = []
    result = RagService(FakeIndex(), RagSettings(), StreamChat([reply])).ask("问题", on_event=lambda *e: events.append(e))
    assert any(name == "delta" for name, _ in events)
    assert result["reason"] == "model_abstained"
    assert result["answer"] != "生成中的草稿"


def test_truncation_resets_preview_and_retries_only_once():
    chat = StreamChat([OllamaOutputLimitError(2048), json.dumps(FakeChat().reply)])
    events = []
    result = RagService(FakeIndex(), RagSettings(), chat).ask("问题", on_event=lambda *e: events.append(e))
    assert result["status"] == "answered"
    assert chat.attempts == 2
    assert sum(name == "reset" for name, _ in events) == 1
    reset = next(i for i, (name, _) in enumerate(events) if name == "reset")
    assert "".join(data["text"] for name, data in events[reset + 1:] if name == "delta") == result["answer"]
    assert chat.calls[-1]["response_format"]["properties"]["answer"]["maxLength"] == 180
    assert "evidence" not in chat.calls[-1]["response_format"]["properties"]


def test_repeated_stream_truncation_and_cancel_are_not_final_answers():
    chat = StreamChat([OllamaOutputLimitError(2048)] * 2)
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), chat).ask("问题", on_event=lambda *e: None)
    assert error.value.code == "generation_truncated" and chat.attempts == 2
    stopped = Event()
    chat = StreamChat()
    def emit(name, data):
        if name == "delta": stopped.set()
    with pytest.raises(RequestCancelled):
        RagService(FakeIndex(), RagSettings(), chat).ask("问题", on_event=emit, cancel=stopped)
    assert chat.closed == 1


def test_no_sources_does_not_load_streaming_model():
    chat = StreamChat()
    result = RagService(FakeIndex([]), RagSettings(), chat).ask("问题", on_event=lambda *e: None)
    assert result["reason"] == "no_sources" and chat.attempts == 0


@contextmanager
def ollama_server(packets, release=None, remote=False):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson" if self.path == "/api/chat" else "application/json")
            self.end_headers()
            if self.path == "/api/show":
                self.wfile.write(json.dumps({"capabilities": ["completion"], **({"remote_host": "remote"} if remote else {})}).encode())
                return
            try:
                for i, packet in enumerate(packets):
                    self.wfile.write((json.dumps(packet, ensure_ascii=False) + "\n").encode())
                    self.wfile.flush()
                    if i == 0 and release is not None:
                        assert release.wait(5)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=1), requests
    finally:
        if release is not None: release.set()
        server.shutdown()
        server.server_close()
        worker.join()


def test_ollama_yields_before_completion_ignores_thinking_and_sends_stream_true():
    release = Event()
    packets = [{"message": {"content": "你", "thinking": "secret"}, "done": False},
               {"message": {"content": "好"}, "done": True}]
    with ollama_server(packets, release) as (client, calls):
        stream = client.chat_stream("local", "hello")
        assert next(stream) == "你"
        assert not release.is_set()
        release.set()
        assert list(stream) == ["好"]
        assert calls[-1][1]["stream"] is True and calls[-1][1]["keep_alive"] == -1


@pytest.mark.parametrize("packets", [
    [{"message": {"content": "partial"}, "done": False}],
    [{"error": "model failure"}],
    [{"message": {"content": ""}, "done": True}],
    [{"message": {"content": 1}, "done": True}],
])
def test_stream_transport_rejects_incomplete_and_invalid_results(packets):
    with ollama_server(packets) as (client, _):
        with pytest.raises(OllamaError):
            list(client.chat_stream("local", "hello"))


def test_stream_transport_length_and_remote_restrictions():
    with ollama_server([{"message": {"content": "partial"}, "done": True, "done_reason": "length"}]) as (client, _):
        with pytest.raises(OllamaOutputLimitError):
            list(client.chat_stream("local", "hello"))
    with ollama_server([], remote=True) as (client, calls):
        with pytest.raises(OllamaError):
            list(client.chat_stream("local", "hello"))
        assert [path for path, _ in calls] == ["/api/show"]


def test_api_stream_final_envelope_and_validation():
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(RagService(FakeIndex(), RagSettings(), StreamChat())))
    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "谁结义？"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        events = [block.splitlines() for block in response.text.strip().split("\n\n")]
        assert events[-1][0] == "event: result"
        final = json.loads(events[-1][1][6:])
        assert final["success"] and final["data"]["status"] == "answered"
        assert any(block[0] == "event: delta" for block in events)
        assert client.post("/chat/stream", json={"question": " "}).status_code == 422
        schema = client.get("/openapi.json").json()
        assert "text/event-stream" in schema["paths"]["/chat/stream"]["post"]["responses"]["200"]["content"]
    assert is_heavy_request({"type": "http", "method": "POST", "path": "/chat/stream/"})


def test_api_stream_error_is_a_failure_event():
    chat = StreamChat([OllamaError("unavailable")])
    app = FastAPI()
    app.include_router(create_router(RagService(FakeIndex(), RagSettings(), chat)))
    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "问题"})
    assert "event: error" in response.text and "event: result" not in response.text
    last = json.loads(response.text.strip().split("\n\n")[-1].splitlines()[1][6:])
    assert last["data"] is None and last["error"]["code"] == "generation_unavailable"


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
def test_disconnect_retains_slot_until_model_worker_has_exited(spec):
    entered, cancelled, release = Event(), Event(), Event()
    class BlockingRag:
        def ask(self, *args, on_event, cancel, **kwargs):
            on_event("delta", {"text": "草稿"})
            entered.set()
            assert cancel.wait(5)
            cancelled.set()
            assert release.wait(5)
            raise RequestCancelled()
    async def scenario():
        app = FastAPI()
        limiter = RequestLimiter(1)
        app.add_middleware(HeavyRequestMiddleware, limiter=limiter)
        app.include_router(create_router(BlockingRag()))
        body = json.dumps({"question": "问题"}).encode()
        disconnected, sent = asyncio.Event(), asyncio.Event()
        delivered = False
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if message["type"] == "http.response.body" and b"event: delta" in message.get("body", b""):
                sent.set()
                if spec == "2.4": raise OSError("client disconnected")
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec}, "method": "POST",
                 "path": "/chat/stream", "raw_path": b"/chat/stream", "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")], "scheme": "http",
                 "http_version": "1.1", "server": ("test", 80), "client": ("test", 123)}
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(sent.wait(), 3)
            disconnected.set()
            assert await asyncio.to_thread(cancelled.wait, 3)
            assert not limiter.try_acquire()
        finally:
            release.set()
            try: await asyncio.wait_for(task, 5)
            except Exception:
                if spec != "2.4": raise
        assert limiter.try_acquire()
        limiter.release()
        await limiter.wait_for_idle()
    asyncio.run(scenario())
