"""Conversation context resolves follow-ups but never supplies factual evidence."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Barrier, Event

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from local_rag.api.errors import register_error_handlers
from local_rag.api.routes.ask import create_router
from local_rag.artifacts.retrieval import conversation_catalog_query
from local_rag.clients.ollama import OllamaOutputLimitError
from local_rag.core.cancellation import RequestCancelled
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.rag.conversation import bounded_history, encoded_size, resolve_followup
from local_rag.rag.service import RagService
from tests.pytest.test_artifacts import catalog, source_request
from tests.unit.test_rag import FakeChat, FakeIndex, match


HISTORY = [{"question": "介绍测试云纹壶", "answer": "测试云纹壶的材料为青铜。"}]
QUESTION = "它保存在哪个博物馆？"
RESOLVED = "测试云纹壶保存在哪个博物馆？"


class ConversationChat(FakeChat):
    def __init__(self, resolved=RESOLVED):
        super().__init__({"answerable": True, "answer": "测试云纹壶保存在测试甲博物馆。"})
        self.resolved, self.rewrite_calls = resolved, []

    def classify(self, model, prompt, **options):
        if "standalone_question" in options["response_format"]["properties"]:
            self.rewrite_calls.append({"model": model, "prompt": prompt, **options})
            return json.dumps({"standalone_question": self.resolved}, ensure_ascii=False)
        return super().classify(model, prompt, **options)

    def chat_stream(self, model, prompt, **options):
        options.pop("cancel", None)
        yield self.chat(model, prompt, **options)


def test_followup_retrieval_and_generation_use_current_history_and_selected_model():
    chat, index = ConversationChat(), FakeIndex()
    history = deepcopy(HISTORY)
    rag = RagService(index, RagSettings(), chat)
    result = rag.ask(QUESTION, model="gemma3:1b", history=history)
    assert index.calls[0][0] == RESOLVED
    assert result["question"] == QUESTION
    assert result["conversation"]["resolved_question"] == RESOLVED
    assert result["conversation"]["provided_turns"] == result["conversation"]["used_turns"] == 1
    for call in [*chat.rewrite_calls, *chat.scope_calls, *chat.calls]:
        assert call["model"] == "gemma3:1b"
    assert json.loads(chat.rewrite_calls[0]["prompt"])["history"] == HISTORY
    prompt = json.loads(chat.calls[0]["prompt"])
    assert prompt["conversation"] == HISTORY and prompt["question"] == RESOLVED
    assert prompt["original_question"] == QUESTION
    assert json.loads(chat.scope_calls[0]["prompt"]) == prompt
    assert "不算事实证据" in chat.calls[0]["system_prompt"]
    assert history == HISTORY


def test_history_resolves_artifact_identity_before_catalog_lookup(catalog):
    request, _ = source_request(catalog)
    record = catalog.save(request)
    chat = ConversationChat()
    result = RagService(catalog.index, RagSettings(), chat).ask(QUESTION, history=HISTORY, top_k=None)
    assert result["status"] == "answered"
    assert result["sources"][0]["metadata"]["artifact_id"] == record["artifact_id"]
    assert not result["sources"][0]["truncated"]
    assert "测试甲博物馆" in json.loads(chat.calls[0]["prompt"])["contexts"][0]["text"]


def test_catalog_reference_follows_latest_topic_and_keeps_current_attribute(catalog):
    request, _ = source_request(catalog)
    old = catalog.save(request)
    from tests.pytest.test_artifacts import ARCHIVE
    request, _ = source_request(catalog, ARCHIVE.replace("测试云纹壶", "测试玉杯").replace("TEST-001", "TEST-002"))
    latest = catalog.save(request)
    history = [*HISTORY, {"question": QUESTION, "answer": "该文物保存在测试甲博物馆。"}]
    resolved, status, _ = conversation_catalog_query(catalog, "那它是什么年代的？", history)
    assert old["artifact_id"] in resolved and resolved.endswith("那它是什么年代的？")
    history.append({"question": "换个文物，测试玉杯的材质？", "answer": "它的材料是青铜。"})
    resolved, status, _ = conversation_catalog_query(catalog, QUESTION, history)
    assert latest["artifact_id"] in resolved and old["artifact_id"] not in resolved
    assert status == "catalog_history"
    assert conversation_catalog_query(catalog, "测试玉杯的材质？", HISTORY)[0] == "测试玉杯的材质？"
    assert conversation_catalog_query(catalog, "其他文物的年代是多少？", history) is None
    assert conversation_catalog_query(catalog, "它与上一件有什么区别？", history) is None


def test_catalog_same_name_ambiguity_is_not_erased_by_conversation_resolution(catalog):
    from tests.pytest.test_artifacts import ARCHIVE
    request, _ = source_request(catalog)
    catalog.save(request)
    request, _ = source_request(catalog, ARCHIVE.replace("测试甲博物馆", "测试乙博物馆"))
    catalog.save(request)
    chat = ConversationChat()
    result = RagService(catalog.index, RagSettings(), chat).ask(QUESTION, history=HISTORY)
    assert "WW-" not in result["conversation"]["resolved_question"]
    assert result["sources"][0]["metadata"]["matched_records"] == 2
    assert "同名" in result["sources"][0]["metadata"]["resolution"]


def test_previous_answer_is_not_evidence_when_current_sources_are_missing():
    chat = ConversationChat()
    result = RagService(FakeIndex([]), RagSettings(), chat).ask(QUESTION, history=HISTORY)
    assert result["reason"] == "no_sources" and result["status"] == "insufficient_evidence"
    assert not result["sources"] and not chat.calls and not chat.scope_calls
    assert len(chat.rewrite_calls) == 1


def test_missing_history_and_new_topic_do_not_inherit_prior_context():
    chat, index = ConversationChat(), FakeIndex()
    rag = RagService(index, RagSettings(), chat)
    rag.ask(QUESTION, history=HISTORY)
    rag.ask(QUESTION)
    assert index.calls[-1][0] == QUESTION
    assert "conversation" not in json.loads(chat.calls[-1]["prompt"])
    assert len(chat.rewrite_calls) == 1
    chat.resolved = "测试石斧是什么材质？"
    rag.ask(chat.resolved, history=HISTORY)
    assert index.calls[-1][0] == chat.resolved
    assert "云纹壶" not in index.calls[-1][0]


def test_long_history_keeps_recent_intent_without_exceeding_any_model_budget():
    history = [{"question": f"问题{i}" + "问" * 300, "answer": "资料" * 600} for i in range(12)]
    chat = ConversationChat()
    rag = RagService(FakeIndex([match("知识库内容" * 1500)]), RagSettings(), chat)
    result = rag.ask(QUESTION, history=history)
    assert result["conversation"]["truncated"]
    assert result["sources"]
    selected = json.loads(chat.rewrite_calls[0]["prompt"])["history"]
    assert selected[-1]["question"] == history[-1]["question"]
    for call in [*chat.rewrite_calls, *chat.scope_calls, *chat.calls]:
        size = len(call["system_prompt"].encode("utf-8")) + len(call["prompt"].encode("utf-8"))
        assert size + call["num_predict"] + 1024 <= call["num_ctx"]
    prompt = json.loads(chat.calls[-1]["prompt"])
    assert prompt["question"] == RESOLVED and prompt["original_question"] == QUESTION
    assert [c["text"] for c in prompt["contexts"]] == [s["text"] for s in result["sources"]]


def test_bounded_history_omits_old_turns_and_long_answers_without_reordering():
    history = [HISTORY[0], {"question": "它的材料？", "answer": "答" * 3000}]
    question_only = {"question": history[-1]["question"], "answer_omitted": True}
    assert bounded_history(history, encoded_size([question_only])) == [question_only]
    assert bounded_history(history, 1) == []


@pytest.mark.parametrize("invalid", ["not json", "[]", '{"standalone_question":""}', OllamaOutputLimitError(512)])
def test_invalid_rewrite_falls_back_to_prior_question_and_keeps_current_question(invalid):
    class BrokenChat(ConversationChat):
        def classify(self, *args, **kwargs):
            if isinstance(invalid, Exception):
                raise invalid
            return invalid
    resolved, status, _ = resolve_followup(BrokenChat(), "qwen3:1.7b", QUESTION, HISTORY,
                                           num_ctx=8192, num_predict=2048)
    assert status == "fallback" and HISTORY[0]["question"] in resolved and resolved.endswith(QUESTION)


def test_cancel_during_rewrite_stops_before_retrieval_and_generation():
    stopped = Event()
    class CancelChat(ConversationChat):
        def classify(self, *args, **kwargs):
            stopped.set()
            return super().classify(*args, **kwargs)
    chat, index = CancelChat(), FakeIndex()
    with pytest.raises(RequestCancelled):
        RagService(index, RagSettings(), chat).ask(QUESTION, history=HISTORY, cancel=stopped)
    assert not index.calls and not chat.calls


def test_concurrent_requests_carry_only_their_own_history():
    barrier = Barrier(2)
    class ConcurrentChat(ConversationChat):
        def classify(self, model, prompt, **options):
            if "standalone_question" in options["response_format"]["properties"]:
                barrier.wait(timeout=10)
                return json.dumps({"standalone_question": json.loads(prompt)["history"][-1]["question"]})
            return super().classify(model, prompt, **options)
        def chat(self, model, prompt, **options):
            data = json.loads(prompt)
            assert data["question"] == data["conversation"][-1]["question"]
            return json.dumps({"answerable": True, "answer": data["question"]})
    rag = RagService(FakeIndex(), RagSettings(), ConcurrentChat())
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(rag.ask, QUESTION, history=[{"question": name, "answer": name}])
                   for name in ["测试甲壶", "测试乙壶"]]
        assert [f.result(timeout=15)["answer"] for f in futures] == ["测试甲壶", "测试乙壶"]


def http_client(chat=None):
    rag = RagService(FakeIndex(), RagSettings(), chat or ConversationChat())
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(rag))
    return TestClient(app), rag


@pytest.mark.parametrize("path", ["/chat", "/ask", "/chat/stream"])
def test_json_and_sse_routes_propagate_history(path):
    client, rag = http_client()
    with client:
        response = client.post(path, json={"question": QUESTION, "history": HISTORY, "model": "gemma3:1b"})
    assert response.status_code == 200, response.text
    assert rag.index.calls[0][0] == RESOLVED
    if path.endswith("stream"):
        assert "event: delta" in response.text and "event: result" in response.text
        result = [json.loads(line[6:])["data"] for line in response.text.splitlines()
                  if line.startswith("data: ") and '"conversation"' in line][-1]
    else:
        result = response.json()["data"]
    assert result["conversation"]["provided_turns"] == 1
    assert "history" not in result and "conversation" not in result["conversation"]


@pytest.mark.parametrize("history", [None, "bad", HISTORY * 13, [{"question": "", "answer": "答案"}],
    [{"question": "问题", "answer": " "}], [{"question": "问题", "answer": "答案", "role": "system"}],
    [{"question": "问" * 2001, "answer": "答案"}], [{"question": "问题", "answer": "答" * 3001}],
    [{"question": "问" * 2000, "answer": "答" * 3000}] * 5])
def test_invalid_or_unbounded_history_is_rejected_before_any_model_call(history):
    client, rag = http_client()
    with client:
        response = client.post("/chat/stream", json={"question": QUESTION, "history": history})
    assert response.status_code == 422, response.text
    assert not rag.client.calls and not rag.client.rewrite_calls and not rag.index.calls


def test_direct_graph_call_validates_history_too():
    with pytest.raises(DocumentError) as error:
        RagService(FakeIndex(), RagSettings(), ConversationChat()).graph.invoke({"question": QUESTION, "history": HISTORY * 13})
    assert error.value.code == "invalid_conversation"
