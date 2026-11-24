"""Local model discovery, request isolation, and model-specific thinking controls."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

from local_rag.clients.ollama import OllamaClient, OllamaError, OllamaOutputLimitError
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.rag.service import RagService
from tests.unit.test_rag import FakeChat, FakeIndex


@pytest.mark.parametrize("name", ["qwen3:1.7b", "gemma3:1b", "deepseek-r1:14b", "custom/llama:local"])
def test_local_model_families_can_be_configured(name):
    assert RagSettings(model_name=name).model_name == name


def test_catalog_checks_capability_and_cloud_metadata_without_loading():
    client = OllamaClient()
    entries = [{"name": name, "digest": "digest", "size": 123, "details": {"parameter_size": "1B"}}
               for name in ["gemma3:1b", "bge-m3:latest", "remote-alias:latest"]]
    responses = [
        {"models": entries},
        {"capabilities": ["completion"]},
        {"capabilities": ["embedding"]},
        {"capabilities": ["completion"], "remote_host": "https://example.com"},
    ]
    with patch.object(client, "_request", side_effect=responses) as request:
        result = client.list_chat_models()
    assert [item["name"] for item in result["models"]] == ["gemma3:1b"]
    assert {item["name"] for item in result["unavailable_models"]} == {"bge-m3:latest", "remote-alias:latest"}
    assert [call.args[0] for call in request.call_args_list] == ["/api/tags"] + ["/api/show"] * 3


@pytest.mark.parametrize("models", [None, {}, ["invalid"], [{"name": ""}]])
def test_invalid_catalog_fails_explicitly(models):
    client = OllamaClient()
    with patch.object(client, "_request", return_value={"models": models}):
        with pytest.raises(OllamaError):
            client.list_chat_models()


def test_catalog_empty_is_success_and_duplicate_tags_are_not_repeated():
    client = OllamaClient()
    with patch.object(client, "_request", return_value={"models": []}):
        assert client.list_chat_models() == {"models": [], "unavailable_models": []}
    with patch.object(client, "_request", side_effect=[
        {"models": [{"name": "gemma3:1b"}, {"name": "gemma3:1b"}]},
        {"capabilities": ["completion"]},
    ]) as request:
        assert len(client.list_chat_models()["models"]) == 1
        assert request.call_count == 2


def test_per_question_model_does_not_change_default_or_embedding_configuration():
    service = RagService(FakeIndex(), RagSettings(), FakeChat())
    chosen = service.ask("谁在桃园结义？", model="gemma3:1b")
    default = service.ask("谁在桃园结义？")
    assert chosen["model"]["name"] == "gemma3:1b"
    assert default["model"]["name"] == "qwen3:1.7b"
    assert service.settings.model_name == "qwen3:1.7b"
    assert chosen["retrieval"]["model"]["name"] == "bge-m3:latest"


def test_unavailable_selected_model_does_not_fall_back_even_with_no_documents():
    service = RagService(FakeIndex([]), RagSettings(), FakeChat())
    with pytest.raises(DocumentError) as error:
        service.ask("问题", model="missing")
    assert error.value.code == "invalid_chat_model"
    assert not service.index.calls
    assert not service.client.calls


def test_generation_retry_keeps_selected_model():
    client = FakeChat()
    service = RagService(FakeIndex(), RagSettings(), client)
    with patch.object(client, "chat", side_effect=[
        OllamaOutputLimitError(2048), json.dumps(client.reply, ensure_ascii=False),
    ]) as chat:
        result = service.ask("问题", model="gemma3:1b")
    assert result["status"] == "answered"
    assert [call.args[0] for call in chat.call_args_list] == ["gemma3:1b", "gemma3:1b"]
    assert set(client.digest_calls) == {"gemma3:1b"}


def test_concurrent_service_calls_do_not_leak_model_selection():
    client = FakeChat()
    service = RagService(FakeIndex(), RagSettings(), client)
    barrier = Barrier(2)
    def answer(model, prompt, **kwargs):
        barrier.wait(timeout=5)
        reply = {**client.reply, "answer": model}
        return json.dumps(reply, ensure_ascii=False)
    with patch.object(client, "chat", side_effect=answer):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda name: service.ask("问题", model=name),
                                    ["qwen3:1.7b", "gemma3:1b"]))
    assert [(r["answer"], r["model"]["name"]) for r in results] == [
        ("qwen3:1.7b", "qwen3:1.7b"), ("gemma3:1b", "gemma3:1b")]
    assert service.settings.model_name == "qwen3:1.7b"


@pytest.mark.parametrize("metadata,expected", [
    ({"values": [False, True]}, False),
    ({"values": ["low", "medium", "high"]}, "low"),
    ({"values": [True], "default": True}, None),
    (None, None),
])
def test_thinking_controls_follow_selected_model_metadata(metadata, expected):
    client = OllamaClient()
    info = {"capabilities": ["completion", "thinking"]}
    if metadata is not None:
        info["thinking"] = metadata
    with patch.object(client, "_request", side_effect=[
        info, {"message": {"content": "answer"}, "done": True, "done_reason": "stop"},
    ]) as request:
        client.chat("local-model", "question")
    payload = request.call_args_list[-1].args[1]
    if expected is None:
        assert "think" not in payload
    else:
        assert payload["think"] == expected
