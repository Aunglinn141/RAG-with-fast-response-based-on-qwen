"""English answers are translations of completed KB answers, not independent QA."""
import json
import re
from threading import Event
from unittest.mock import patch

import pytest

from local_rag.clients.ollama import OllamaError, OllamaOutputLimitError
from local_rag.core.cancellation import RequestCancelled
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.rag.service import RagService
from tests.unit.test_rag import FakeIndex, match
from tests.pytest.test_language_routing import LanguageChat


DRAFT = "孙悟空是唐僧的徒弟，五百年前曾大闹天宫。资料没有提供他的电话号码。"
ENGLISH = "Sun Wukong is a disciple of Tang Seng and caused havoc in Heaven five hundred years earlier. His telephone number is not provided."


class TranslationChat(LanguageChat):
    def __init__(self, *, stream_replies=None, translation=ENGLISH, answerable=True):
        super().__init__(answer=DRAFT)
        self.reply["answerable"] = answerable
        self.translation = "孙悟空是谁？"
        self.translated = translation
        self.replies = stream_replies or [translation if isinstance(translation, Exception) else json.dumps({"answer": translation})]

    def chat(self, model, prompt, **options):
        if "text" in json.loads(prompt):
            self.calls.append({"model": model, "prompt": prompt, **options})
            if isinstance(self.translated, Exception):
                raise self.translated
            return self.translated if isinstance(self.translated, str) and self.translated.startswith("{") else json.dumps({"answer": self.translated})
        return super().chat(model, prompt, **options)


@pytest.mark.parametrize("stream", [False, True])
def test_translation_only_receives_grounded_draft_and_stream_only_exposes_english(stream):
    chat, events = TranslationChat(), []
    source = match("孙悟空是唐僧的徒弟，五百年前曾大闹天宫。")
    result = RagService(FakeIndex([source]), RagSettings(), chat).ask("who is Sun Wu Kong",
        on_event=(lambda *e: events.append(e)) if stream else None)
    assert result["answer"] == ENGLISH
    assert result["sources"][0]["metadata"] == source["metadata"]
    assert result["sources"][0]["text"] == source["text"]
    assert result["retrieval"]["strategy"]["answer_pipeline"] == "grounded_answer_then_translation"
    original, translation = chat.calls
    assert json.loads(original["prompt"])["reply_language"] == "zh"
    assert json.loads(translation["prompt"]) == {"text": DRAFT, "glossary": {"孙悟空": "Sun Wukong", "唐僧": "Tang Seng"}}
    assert set(translation["response_format"]["properties"]) == {"answer"}
    pattern = translation["response_format"]["properties"]["answer"]["pattern"]
    assert re.fullmatch(pattern, ENGLISH)
    assert not re.fullmatch(pattern, "Sun Wukong protects Tang Seng from妖魔.")
    assert "pattern" not in original["response_format"]["properties"]["answer"]
    assert "Do not add or remove claims" in translation["system_prompt"]
    assert len(chat.scope_calls) == 1
    assert "draft" not in result and "translation" not in result
    if stream:
        text = "".join(data["text"] for name, data in events if name == "delta")
        assert text == ENGLISH and DRAFT not in text


@pytest.mark.parametrize("stream", [False, True])
def test_unanswerable_chinese_draft_does_not_trigger_translation(stream):
    chat, events = TranslationChat(answerable=False), []
    result = RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong",
        on_event=(lambda *e: events.append(e)) if stream else None)
    assert result["status"] == "insufficient_evidence"
    assert "knowledge-base" in result["answer"]
    assert len(chat.calls) == 1 and chat.attempts == 0
    assert not any(name == "delta" for name, _ in events)


@pytest.mark.parametrize("translation", [None, "", 7, "只返回中文。", '{"answer": "unfinished'])
def test_bad_translation_is_an_error_and_never_returns_chinese_draft_as_english(translation):
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), TranslationChat(translation=translation)).ask("who is Sun Wu Kong")
    assert error.value.code == "answer_translation_unavailable"


def test_translation_retry_does_not_regenerate_or_stream_the_chinese_draft():
    chat = TranslationChat(stream_replies=[OllamaOutputLimitError(2048), json.dumps({"answer":ENGLISH})])
    events = []
    result = RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong", on_event=lambda *e: events.append(e))
    assert result["answer"] == ENGLISH
    assert len(chat.calls) == 3 and len(chat.scope_calls) == 1
    assert json.loads(chat.calls[1]["prompt"]) == json.loads(chat.calls[2]["prompt"])
    assert json.loads(chat.calls[1]["prompt"])["text"] == DRAFT
    assert chat.calls[2]["response_format"]["properties"]["answer"]["maxLength"] == 900
    reset = next(i for i, event in enumerate(events) if event == ("reset", {"reason": "translation_retry"}))
    assert "".join(d["text"] for n, d in events[reset+1:] if n=="delta") == ENGLISH


def test_translation_truncation_is_bounded_to_two_attempts():
    chat = TranslationChat(stream_replies=[OllamaOutputLimitError(2048)]*2)
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong", on_event=lambda *e: None)
    assert error.value.code == "generation_truncated" and chat.attempts == 2


def test_translation_connection_error_is_reported_without_chinese_answer():
    chat = TranslationChat(translation=OllamaError("offline"))
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong")
    assert error.value.code == "generation_unavailable"


def test_translation_cancel_and_version_change_are_preserved():
    stopped = Event()
    chat = TranslationChat()
    def emit(name, data):
        if name == "delta": stopped.set()
    with pytest.raises(RequestCancelled):
        RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong", on_event=emit, cancel=stopped)
    assert chat.closed == 1
    chat = TranslationChat()
    with patch.object(chat, "model_digest", side_effect=["before","before","changed"]):
        with pytest.raises(DocumentError) as error:
            RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong")
    assert error.value.code == "chat_model_version_mismatch"


@pytest.mark.parametrize("question", ["hello", "孙悟空是谁？", "Who is Sun Wukong? Answer in Chinese."])
def test_greeting_and_chinese_output_do_not_enter_answer_translation(question):
    chat = LanguageChat(answer="你好。")
    result = RagService(FakeIndex(), RagSettings(), chat).ask(question)
    assert len(chat.calls) == 1 and result["retrieval"]["strategy"]["answer_pipeline"] == "direct"


def test_translation_glossary_never_introduces_people_or_titles_absent_from_answer():
    from local_rag.rag.name_aliases import translation_glossary
    assert translation_glossary("资料不足，无法确定。") == {}
    assert translation_glossary("孙悟空是齐天大圣。") == {
        "孙悟空": "Sun Wukong", "齐天大圣": "Great Sage Equal to Heaven"}


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_language_translation_retries_the_same_draft_and_returns_only_final_english(stream):
    mixed = "Sun Wukong protects Tang Seng from妖魔and other dangers."
    replies = [json.dumps({"answer": mixed}), json.dumps({"answer": ENGLISH})]
    chat, events = TranslationChat(stream_replies=replies), []
    original_chat = chat.chat
    pending = iter(replies)
    def translate(model, prompt, **options):
        if "text" not in json.loads(prompt):
            return original_chat(model, prompt, **options)
        chat.calls.append({"model": model, "prompt": prompt, **options})
        return next(pending)
    with patch.object(chat, "chat", side_effect=translate):
        result = RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong",
            on_event=(lambda *e: events.append(e)) if stream else None)
    assert result["answer"] == ENGLISH
    assert len(chat.calls) == 3 and len(chat.scope_calls) == 1
    assert json.loads(chat.calls[1]["prompt"]) == json.loads(chat.calls[2]["prompt"])
    assert json.loads(chat.calls[2]["prompt"])["text"] == DRAFT
    if stream:
        reset = next(i for i, event in enumerate(events) if event == ("reset", {"reason": "translation_retry"}))
        assert "".join(d["text"] for n, d in events[reset+1:] if n == "delta") == ENGLISH


@pytest.mark.parametrize("stream", [False, True])
def test_repeated_mixed_language_is_an_error_after_one_retry(stream):
    mixed = "Sun Wukong gained powers through his修炼and protected Tang Seng."
    chat = TranslationChat(translation=mixed, stream_replies=[json.dumps({"answer": mixed})]*2)
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong",
            on_event=(lambda *e: None) if stream else None)
    assert error.value.code == "answer_translation_unavailable"
    assert len(chat.calls) == 3


def test_length_and_language_repair_share_one_translation_retry_budget():
    chat = TranslationChat(stream_replies=[OllamaOutputLimitError(2048),
        json.dumps({"answer": "Sun Wukong protects Tang Seng from妖魔."})])
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), chat).ask("who is Sun Wu Kong", on_event=lambda *e: None)
    assert error.value.code == "answer_translation_unavailable" and chat.attempts == 2
