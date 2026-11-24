"""Bounded generation recovery; no real model or formal storage required."""

import json
from unittest.mock import patch

import pytest

from local_rag.clients.ollama import OllamaClient, OllamaError, OllamaOutputLimitError
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.rag.prompts import COMPACT_REPLY_INSTRUCTION
from local_rag.rag.service import RagService
from tests.unit.test_rag import FakeChat, FakeIndex, QUOTE, match


@pytest.fixture
def rag():
    return RagService(FakeIndex(), RagSettings(), client=FakeChat())


def test_limit_retries_once_with_same_sources_and_shorter_schema(rag):
    reply = json.dumps(rag.client.reply, ensure_ascii=False)
    with patch.object(rag.client, "chat", side_effect=[OllamaOutputLimitError(2048), reply]) as chat:
        result = rag.ask("关公做了些什么？")
    assert result["status"] == "answered"
    assert result["answer"] == rag.client.reply["answer"]
    assert len(rag.index.calls) == 1
    first, retry = chat.call_args_list
    assert first.args == retry.args
    assert first.kwargs["num_predict"] == retry.kwargs["num_predict"]
    assert COMPACT_REPLY_INSTRUCTION in retry.kwargs["system_prompt"]
    schema = retry.kwargs["response_format"]["properties"]
    initial = first.kwargs["response_format"]["properties"]
    assert set(initial) == set(schema) == {"answerable", "answer"}
    assert initial["answer"]["maxLength"] == 300
    assert schema["answer"]["maxLength"] == 180
    assert result["evidence"] == []


def test_repeated_limit_is_502_and_does_not_loop(rag):
    with patch.object(rag.client, "chat", side_effect=OllamaOutputLimitError(2048)) as chat:
        with pytest.raises(DocumentError) as error:
            rag.ask("关公做了些什么？")
    assert chat.call_count == 2
    assert (error.value.status, error.value.code) == (502, "generation_truncated")
    assert "2048" in str(error.value)


def test_retry_still_respects_model_declared_missing_knowledge(rag):
    reply = json.dumps({"answerable": False, "answer": "编造内容",
                        "evidence": [{"source_id": "S1", "quote": "文档不存在的事实"}]})
    with patch.object(rag.client, "chat", side_effect=[OllamaOutputLimitError(2048), reply]):
        result = rag.ask("关公做了些什么？")
    assert result["status"] == "insufficient_evidence"
    assert result["answer"] != "编造内容"


def test_version_change_after_truncation_prevents_retry(rag):
    with patch.object(rag.client, "chat", side_effect=OllamaOutputLimitError(2048)) as chat:
        with patch.object(rag.client, "model_digest", side_effect=["before", "changed"]):
            with pytest.raises(DocumentError) as error:
                rag.ask("关公做了些什么？")
    assert chat.call_count == 1
    assert error.value.code == "chat_model_version_mismatch"


def test_connection_failure_is_not_retried_as_truncation(rag):
    with patch.object(rag.client, "chat", side_effect=OllamaError("offline")) as chat:
        with pytest.raises(DocumentError) as error:
            rag.ask("关公做了些什么？")
    assert chat.call_count == 1
    assert (error.value.status, error.value.code) == (503, "generation_unavailable")


def test_retry_instructions_fit_reserved_context_budget(rag):
    rag.index.matches = [match(QUOTE + "原文内容" * 300, i + 1) for i in range(4)]
    reply = json.dumps(rag.client.reply, ensure_ascii=False)
    with patch.object(rag.client, "chat", side_effect=[OllamaOutputLimitError(2048), reply]) as chat:
        rag.ask("关公做了些什么？")
    for call in chat.call_args_list:
        size = len(call.args[1].encode("utf-8")) + len(call.kwargs["system_prompt"].encode("utf-8"))
        assert size + rag.settings.num_predict + 1024 <= rag.settings.num_ctx
    assert chat.call_args_list[0].args[1] == chat.call_args_list[1].args[1]


@pytest.mark.parametrize("content", ['{"answer":"not complete"}', "", "partial JSON"])
def test_limit_is_typed_even_for_empty_or_parseable_content(content):
    client = OllamaClient(timeout=1)
    responses = [{"capabilities": ["completion"]},
                 {"message": {"content": content}, "done": True, "done_reason": "length"}]
    with patch.object(client, "_request", side_effect=responses):
        with pytest.raises(OllamaOutputLimitError) as error:
            client.chat("qwen3:1.7b", "question", num_predict=2048)
    assert error.value.num_predict == 2048


def test_output_budget_environment(monkeypatch):
    monkeypatch.setenv("RAG_NUM_PREDICT", "1536")
    monkeypatch.setenv("RAG_NUM_CTX", "8192")
    assert RagSettings.from_env().num_predict == 1536


@pytest.mark.parametrize("value", ["0", "-1", "2049", "1.5", "invalid"])
def test_invalid_output_budget_environment(monkeypatch, value):
    monkeypatch.setenv("RAG_NUM_PREDICT", value)
    with pytest.raises(ValueError):
        RagSettings.from_env()


@pytest.mark.parametrize("value", [True, 512.5])
def test_output_budget_requires_integer(value):
    with pytest.raises(ValueError):
        RagSettings(num_predict=value)
