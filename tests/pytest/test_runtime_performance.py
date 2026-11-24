"""Resident-model and existing-index contracts without real models."""
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from local_rag.api.runtime import RuntimeWarmup
from local_rag.api.limits import is_heavy_request
from local_rag.clients.ollama import OllamaClient, OllamaError
from local_rag.core.config import RagSettings, VectorSettings, RuntimeSettings
from local_rag.core.exceptions import DocumentError
from local_rag.core.timing import timed_operation, model_phase, stage
from local_rag.rag.scenarios import plan_retrieval
from tests.pytest.test_book_retrieval import book_index


@pytest.mark.parametrize("value", ["-1", "30m", "0", "3600", "1.5s"])
def test_keep_alive_setting_is_shared_by_embedding_and_chat(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", value)
    assert RagSettings.from_env().keep_alive == VectorSettings.from_env().keep_alive == value


@pytest.mark.parametrize("value", ["forever", "", "None", "1 day", "inf"])
def test_invalid_keep_alive_is_rejected(value):
    with pytest.raises(ValueError):
        RagSettings(keep_alive=value)


def test_repeat_chat_retains_model_and_switch_releases_only_previous_chat():
    calls = []
    client = OllamaClient()
    def request(path, payload=None):
        calls.append((path, payload))
        if path == "/api/show":
            return {"capabilities": ["completion", "embedding"]}
        if path == "/api/embed":
            return {"embeddings": [[1, 0]]}
        return {"done": True, "message": {"content": "ok"}}
    with patch.object(client, "_request", side_effect=request):
        client.chat("small:1b", "first")
        client.embed("bge-m3", ["question"])
        client.chat("small:1b", "second")
        assert not [p for path, p in calls if path == "/api/generate"]
        client.chat("other:1b", "third")
    assert [p for path, p in calls if path == "/api/generate"] == [{"model": "small:1b", "keep_alive": 0}]
    assert all(p["keep_alive"] == -1 for path, p in calls if path in {"/api/chat", "/api/embed"})


def test_warm_chat_uses_empty_request_and_same_context_as_generation():
    client = OllamaClient()
    with patch.object(client, "require_local_model"), patch.object(client, "_request", return_value={"done": True}) as req:
        client.warm_chat("small:1b", num_ctx=8192)
        first = req.call_args.args[1]
        assert first["messages"] == [] and first["options"]["num_ctx"] == 8192
        assert first["keep_alive"] == -1
        client.warm_chat("small:1b", num_ctx=8192)
        assert req.call_count == 2


def test_real_client_records_http_and_stream_metrics_without_double_counting():
    client = OllamaClient()
    def open_request(req, **kwargs):
        if req.full_url.endswith("/api/show"):
            return io.BytesIO(json.dumps({"capabilities": ["completion", "embedding"]}).encode())
        body = json.loads(req.data)
        if req.full_url.endswith("/api/embed"):
            return io.BytesIO(json.dumps({"embeddings": [[1, 0]], "load_duration": 2_000_000}).encode())
        packet = {"message": {"content": "ok"}, "done": True, "load_duration": 3_000_000,
                  "eval_duration": 4_000_000, "eval_count": 2}
        return io.BytesIO((json.dumps(packet) + ("\n" if body.get("stream") else "")).encode())
    @model_phase("query_translation")
    def translate():
        return client.chat("small:1b", "translate")
    @timed_operation("retrieval")
    def nested():
        with stage("embedding"):
            client.embed("bge-m3", ["question"])
        return {}
    @timed_operation("chat")
    def run():
        translate()
        nested()
        list(client.chat_stream("small:1b", "answer"))
        return {}
    with patch.object(client._opener, "open", side_effect=open_request):
        first, second = run()["timings"], run()["timings"]
    for result in (first, second):
        assert len(result["model_calls"]) == 3
        assert [c["phase"] for c in result["model_calls"]] == ["query_translation", "embedding", "answer"]
        assert result["llm_load_ms"] == 6 and result["llm_eval_ms"] == 8
        assert result["generation_ms"] <= result["llm_ms"]
        assert result["embedding_chroma_ms"] == result["embedding_ms"]


def test_warmup_and_search_reuse_persisted_vectors_and_cached_books(book_index):
    index, record = book_index
    settings = RagSettings()
    client = SimpleNamespace(active_chat_model=None, warm_chat=lambda *a, **k: {"done": True},
        loaded_models=lambda: [{"name":"bge-m3:latest"}, {"name":settings.model_name}])
    warmup = RuntimeWarmup(index, SimpleNamespace(client=client, settings=settings))
    before = index.registry.get(record["document_id"])
    with patch.object(index.embedder, "embed_documents", side_effect=AssertionError("documents embedded")), \
         patch.object(index, "_index", side_effect=AssertionError("index rebuilt")), \
         patch.object(index.vectors, "upsert", side_effect=AssertionError("vectors written")):
        assert warmup.run()["status"] == "ready"
        with patch.object(index.documents, "hierarchy", side_effect=AssertionError("book read again")), \
             patch.object(index.knowledge, "sync", side_effect=AssertionError("derived index rebuilt")):
            result = index.retrieve_context("关羽是谁", 4, None, plan_retrieval("关羽是谁", settings), settings)
            assert result["matches"] and result["timings"]["embedding_chroma_ms"] >= 0
    assert index.registry.get(record["document_id"]) == before


def test_cache_invalidates_for_file_change_reindex_and_delete(book_index):
    index, record = book_index
    settings = RagSettings()
    def search():
        return index.hybrid_search("关羽", settings=settings)
    search()
    directory = index.documents.directory(record["document_id"])
    path = directory / "document.json"
    path.write_text(path.read_text(encoding="utf-8")+"\n", encoding="utf-8")
    with patch.object(index.documents, "hierarchy", wraps=index.documents.hierarchy) as read:
        search()
        assert read.call_count == 1
    old_generation = index.registry.get(record["document_id"])["generation"]
    index.reindex(record["document_id"])
    assert index.registry.get(record["document_id"])["generation"] != old_generation
    assert record["document_id"] not in index._book_cache
    assert search()["matches"]
    index.delete(record["document_id"])
    assert record["document_id"] not in index._book_cache and not search()["matches"]


def test_warmup_admission_and_unavailable_model():
    assert is_heavy_request({"type":"http", "method":"POST", "path":"/warmup"})
    client = SimpleNamespace(active_chat_model=None)
    index = SimpleNamespace(settings=VectorSettings(),
                            retrieve_context=lambda *a: (_ for _ in ()).throw(OllamaError("offline")))
    warmup = RuntimeWarmup(index, SimpleNamespace(client=client, settings=RagSettings()))
    with pytest.raises(DocumentError) as exc:
        warmup.run()
    assert exc.value.code == "warmup_unavailable" and warmup.last["status"] == "failed"
