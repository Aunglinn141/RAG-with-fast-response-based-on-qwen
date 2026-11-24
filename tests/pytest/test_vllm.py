"""vLLM HTTP/SSE contracts and RAG integration, without a GPU or model download."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from local_rag.api.app import create_app
from local_rag.api.runtime import RuntimeWarmup
from local_rag.clients.chat import create_chat_client
from local_rag.clients.errors import ModelError, ModelOutputLimitError, ModelUnavailableError
from local_rag.clients.ollama import OllamaClient
from local_rag.clients.vllm import VLLMClient
from local_rag.core.cancellation import RequestCancelled
from local_rag.core.config import RagSettings, VectorSettings, DocumentSettings, RuntimeSettings
from local_rag.core.timing import timed_operation
from local_rag.rag.service import RagService
from tests.unit.test_rag import FakeChat, FakeIndex

MODEL = "Qwen/Qwen3-0.6B"
CATALOG = {"data": [{"id": MODEL, "root": MODEL, "max_model_len": 8192}]}


def completion(content="回答", reason="stop"):
    return {"choices": [{"index": 0, "message": {"content": content}, "finish_reason": reason}],
            "usage": {"completion_tokens": 3}}


def chunk(content=None, reason=None, **delta):
    return {"choices": [{"index": 0, "delta": {"content": content, **delta}, "finish_reason": reason}]}


def sse(*packets):
    return b"".join(("data: " + (p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)) + "\n\n").encode()
                    for p in packets)


@pytest.fixture
def server():
    state = SimpleNamespace(requests=[], response=completion(), status=200)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append((self.path, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(CATALOG if self.path == "/v1/models" else {"version": "test"}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, payload))
            self.send_response(state.status)
            if state.status == 302:
                self.send_header("Location", "http://127.0.0.1:1/never")
            self.end_headers()
            self.wfile.write(state.response if isinstance(state.response, bytes) else json.dumps(state.response).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=http.serve_forever, daemon=True)
    worker.start()
    state.client = VLLMClient(f"http://localhost:{http.server_port}/v1", timeout=2)
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        worker.join()


def test_real_http_schema_metrics_and_catalog(server, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    assert server.client.list_chat_models()["models"][0]["name"] == MODEL
    assert server.client.version() == "test"
    @timed_operation("chat")
    def run():
        return {"answer": server.client.chat(MODEL, "问题", response_format=schema)}
    result = run()
    assert result["answer"] == "回答"
    assert result["timings"]["model_calls"][0]["eval_tokens"] == 3
    path, body = server.requests[-1]
    assert path == "/v1/chat/completions"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "options" not in body and "keep_alive" not in body


def test_real_sse_hides_reasoning_and_consumes_usage(server):
    server.response = sse(chunk(reasoning_content="secret reasoning"), chunk("你"), chunk("好"),
                          chunk(reason="stop"), {"choices": [], "usage": {"completion_tokens": 2}}, "[DONE]")
    assert list(server.client.chat_stream(MODEL, "问题")) == ["你", "好"]
    assert server.requests[-1][1]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("status", [302, 400, 404, 500])
def test_http_errors_and_redirects_fail_without_fallback(server, status):
    server.status = status
    with pytest.raises(ModelError):
        server.client.chat(MODEL, "问题")
    assert len(server.requests) == 1


@pytest.mark.parametrize("url", ["https://127.0.0.1:8001", "http://example.org", "http://127.0.0.1:0",
                                 "http://user@localhost:8001", "http://localhost/v2", "http://localhost/v1?x=1"])
def test_only_loopback_service_allowed(url):
    with pytest.raises(ModelError):
        VLLMClient(url)


@pytest.mark.parametrize("wire", [sse(chunk("partial")), sse("[DONE]"), sse(chunk("x"), "[DONE]"),
                                  sse(chunk(reason="stop"), "[DONE]"), sse({"error": "failure"}),
                                  sse("not-json"), sse({"choices": [{}]}), sse(chunk(123)),
                                  sse(chunk("x", "tool_calls")), b"data: " + b"x" * 65537],
                         ids=["eof", "empty", "missing-stop", "no-content", "error", "json", "shape", "type", "tools", "oversize"])
def test_broken_streams_are_never_treated_as_success(wire):
    client = VLLMClient()
    response = io.BytesIO(wire)
    with patch.object(client._opener, "open", return_value=response), pytest.raises(ModelError):
        list(client.chat_stream(MODEL, "问题"))
    assert response.closed


def test_length_errors_and_cancellation_close_connection():
    client = VLLMClient()
    with patch.object(client, "_request", return_value=completion("partial", "length")):
        with pytest.raises(ModelOutputLimitError):
            client.chat(MODEL, "问题")
    response = io.BytesIO(sse(chunk("partial", "length")))
    with patch.object(client._opener, "open", return_value=response), pytest.raises(ModelOutputLimitError):
        list(client.chat_stream(MODEL, "问题"))
    assert response.closed
    cancel = Event()
    response = io.BytesIO(sse(chunk("first"), chunk("second"), chunk(reason="stop"), "[DONE]"))
    with patch.object(client._opener, "open", return_value=response):
        stream = client.chat_stream(MODEL, "问题", cancel=cancel)
        assert next(stream) == "first"
        cancel.set()
        with pytest.raises(RequestCancelled):
            next(stream)
    assert response.closed


def test_concurrent_generation_is_not_serialized():
    client, barrier = VLLMClient(), Barrier(2)
    def respond(*args, **kwargs):
        barrier.wait(timeout=3)
        return completion()
    with patch.object(client, "_request", side_effect=respond), ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: client.chat(MODEL, "问题"), range(2))) == ["回答", "回答"]


def test_config_factory_and_model_names_do_not_use_ollama_aliases(monkeypatch):
    monkeypatch.setenv("RAG_CHAT_BACKEND", "vllm")
    monkeypatch.setenv("VLLM_CHAT_MODEL", MODEL)
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "unrelated:tag")
    settings = RagSettings.from_env()
    assert settings.model_name == MODEL
    assert isinstance(create_chat_client(settings), VLLMClient)
    assert isinstance(create_chat_client(RagSettings()), OllamaClient)
    with pytest.raises(ValueError):
        RagSettings(chat_backend="typo")
    with patch.object(VLLMClient, "_request", return_value=CATALOG):
        assert create_chat_client(settings).model_digest(MODEL).startswith("vllm-metadata:")
        with pytest.raises(ModelUnavailableError):
            create_chat_client(settings).require_local_model(MODEL + ":latest", "completion")


def test_vllm_warmup_checks_embedding_server_separately(server):
    index = SimpleNamespace(settings=VectorSettings(),
                            retrieve_context=lambda *args: {"model": {"name": "bge-m3"}},
                            list_documents=lambda: [],
                            embedder=SimpleNamespace(client=SimpleNamespace(loaded_models=lambda: [{"name": "bge-m3:latest"}])))
    rag = SimpleNamespace(client=server.client, settings=RagSettings(chat_backend="vllm", model_name=MODEL))
    result = RuntimeWarmup(index, rag).run()
    assert result["status"] == "ready" and result["both_models_loaded"]
    generations = [body for path, body in server.requests if body]
    assert len(generations) == 2 and all(body["max_tokens"] == 1 for body in generations)
    assert generations[0]["response_format"]["json_schema"]["schema"]["required"] == ["answerable", "answer"]
    assert generations[1]["response_format"]["json_schema"]["schema"]["required"] == ["basis", "within_scope"]


def test_context_limit_checked_at_warmup(server):
    with pytest.raises(ModelError, match="max_model_len"):
        server.client.warm_chat(MODEL, num_ctx=16384)


def test_fast_mode_retains_scope_gate_and_complete_answer_budget():
    client = FakeChat()
    rag = RagService(FakeIndex(), RagSettings(fast_mode=True), client)
    assert rag.ask("谁在桃园结义？")["status"] == "answered"
    scope = client.scope_calls[0]
    assert scope["response_format"]["required"] == ["within_scope"]
    assert scope["num_predict"] == 32
    assert client.calls[0]["response_format"]["properties"]["answer"]["maxLength"] == 300
    assert client.calls[0]["num_predict"] == 2048
    client.scope_allowed = False
    assert rag.ask("未记载的电话号码是什么？")["status"] == "insufficient_evidence"
    assert len(client.calls) == 1


def test_rag_generation_uses_vllm_json_and_stream(server):
    rag = RagService(FakeIndex(), RagSettings(chat_backend="vllm", model_name=MODEL), server.client)
    reply = json.dumps({"answerable": True, "answer": "你好！"}, ensure_ascii=False)
    server.response = completion(reply)
    assert rag.ask("你好")["answer"] == "你好！"
    server.response = sse(chunk(reply[:20]), chunk(reply[20:]), chunk(reason="stop"), "[DONE]")
    events = []
    result = rag.ask("你好", on_event=lambda name, data: events.append((name, data)))
    assert result["answer"] == "你好！"
    assert "".join(data["text"] for name, data in events if name == "delta") == "你好！"


def test_api_exposes_selected_backend_and_served_model(server, isolated_root):
    from tests.fixtures.embeddings import FakeEmbeddings
    app = create_app(DocumentSettings(storage_dir=isolated_root / "documents"),
                     VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3"),
                     embedder=FakeEmbeddings(), chat_client=server.client,
                     rag_settings=RagSettings(chat_backend="vllm", model_name=MODEL, fast_mode=True),
                     runtime_settings=RuntimeSettings(warmup_on_startup=False))
    with TestClient(app) as api:
        assert api.get("/health").json()["data"]["chat_backend"] == "vllm"
        models = api.get("/models").json()["data"]
        assert models["backend"] == "vllm" and models["models"][0]["name"] == MODEL
