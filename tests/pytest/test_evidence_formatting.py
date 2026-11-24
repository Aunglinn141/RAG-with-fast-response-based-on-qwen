"""Legacy quote matching remains available; current answers do not require citations."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_rag.api.errors import register_error_handlers
from local_rag.api.routes.ask import create_router
from local_rag.core.config import RagSettings
from local_rag.rag.evidence import match_evidence_quote
from local_rag.rag.service import RagService
from tests.unit.test_rag import FakeChat, FakeIndex, QUOTE, match


@pytest.mark.parametrize("quote", [
    "关公曰：二位夫人在车上，可先献茶。",
    "关公曰：‘二位夫人在车上，可先献茶。’",
    '关公曰："二位夫人在车上，可先献茶。"',
    "关公曰：二位夫人\n在车上，可先献茶。",
])
def test_restore_deepseek_missing_quotes_and_qwen_typographic_quotes(quote):
    text = "普净请关公方丈待茶。关公曰：“二位夫人在车上，可先献茶。”普净教取茶先奉夫人。"
    restored = match_evidence_quote(text, quote)
    assert restored in text
    assert "二位夫人在车上，可先献茶。" in restored
    assert "“" in restored


def test_exact_quote_is_unchanged():
    assert match_evidence_quote(QUOTE, QUOTE) == QUOTE


def test_line_wrap_maps_back_to_the_actual_original():
    original = "开放时间为周二\n至周日。"
    assert match_evidence_quote(original, "开放时间为周二至周日。") == original


@pytest.mark.parametrize("source,quote", [
    ("关公没有杀死此人。", "关公杀死此人。"),
    ("二百余人埋伏于寺中。", "三百余人埋伏于寺中。"),
    ("关公先离开。此处还有其他事件。后来返回。", "关公先离开。后来返回。"),
    ("开放时间为周二至周日。", "开放时间为周一至周日。"),
    ("关公曰：“不可。”众人退去。", "关公曰：可。众人退去。"),
    ("the rapist fled", "therapist fled"),
    ("budget 12 34", "budget 1234"),
    ("can't leave", "cant leave"),
    ("关公曰：“不可”。", "关公曰：不可！"),
])
def test_changes_to_facts_or_meaning_are_rejected(source, quote):
    assert match_evidence_quote(source, quote) is None


@pytest.mark.parametrize("quote", ['“”‘’', "   ", "关公", '"关"'])
def test_short_or_format_only_quotes_are_rejected(quote):
    assert match_evidence_quote("关公曰：“你好。”", quote) is None


def test_ambiguous_format_mapping_is_not_guessed():
    assert match_evidence_quote("关公曰：“不可。” 关公曰：‘不可。’", "关公曰：不可。") is None


def test_restoration_cannot_expand_past_length_limit():
    assert match_evidence_quote("这是" + "\n" * 800 + "原文。", "这是原文。") is None


def test_latin_line_wrap_retains_word_boundary():
    assert match_evidence_quote("the\nbook is red", "the book is red") == "the\nbook is red"


def service_for(text, reply):
    return RagService(FakeIndex([match(text)]), RagSettings(), FakeChat(reply))


def test_http_preserves_natural_answer_without_requiring_quotes():
    text = "关公曰：“二位夫人在车上，可先献茶。”"
    answer = "关公要求先照顾车上的两位夫人，让人先给她们奉茶。"
    service = service_for(text, {"answerable": True, "answer": answer})
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(service))
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "关公做了些什么"})
    assert response.status_code == 200
    result = response.json()["data"]
    assert result["status"] == "answered" and result["reason"] is None
    assert result["answer"] == answer and answer not in text
    assert result["answer_policy"] == "knowledge_base_only"
    assert result["evidence"] == []
    assert result["sources"][0]["cited"] is False
    assert result["sources"][0]["metadata"] == service.index.matches[0]["metadata"]


@pytest.mark.parametrize("extra", [
    {"evidence": [{"source_id": "S99", "quote": "模型输出的旧字段"}]},
    {"evidence": []}, {"evidence": None}, {},
])
def test_legacy_evidence_cannot_block_or_become_trusted_metadata(extra):
    answer = "三人在桃园结成兄弟。"
    service = service_for(QUOTE, {"answerable": True, "answer": answer, **extra})
    result = service.ask("概括三人的关系")
    assert result["status"] == "answered" and result["answer"] == answer
    assert result["reason"] is None
    assert result["evidence"] == [] and not result["sources"][0]["cited"]


def test_missing_knowledge_and_missing_retrieval_are_distinct():
    service = service_for(QUOTE, {"answerable": False, "answer": "不应展示的猜测"})
    result = service.ask("结义花费了多少钱")
    assert result["reason"] == "model_abstained"
    assert result["answer"] != "不应展示的猜测"
    service.index.matches = []
    assert service.ask("问题")["reason"] == "no_sources"
