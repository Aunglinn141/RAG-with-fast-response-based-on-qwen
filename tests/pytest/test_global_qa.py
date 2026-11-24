"""Global automatic retrieval and multiple-document context selection contracts."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_rag.api.errors import register_error_handlers
from local_rag.api.routes.ask import create_router
from local_rag.core.config import DocumentSettings, VectorSettings, RagSettings
from local_rag.rag.retrieval import select_context_candidates
from local_rag.rag.service import RagService
from tests.fixtures.embeddings import FakeEmbeddings
from tests.unit.test_rag import FakeChat, FakeIndex, match


def item(document_id, number, score, text=None):
    result = match(text or f"{document_id}中的事项{number}。", number)
    result["chunk_id"] = f"{document_id}:{number}"
    result["metadata"]["document_id"] = document_id
    result["score"] = score
    result["distance"] = 1 - score
    return result


def test_similarly_relevant_other_document_can_contribute_early():
    candidates = [item("a", 1, .9), item("a", 2, .89), item("b", 1, .88)]
    original = deepcopy(candidates)
    selected = select_context_candidates(candidates, 2)
    assert [row["metadata"]["document_id"] for row in selected] == ["a", "b"]
    assert candidates == original


def test_weak_document_is_not_forced_ahead_of_relevant_results():
    candidates = [item("a", 1, .9), item("a", 2, .89), item("b", 1, .1)]
    assert [row["metadata"]["document_id"] for row in select_context_candidates(candidates, 2)] == ["a", "a"]


def test_exact_duplicate_text_and_repeated_chunk_ids_do_not_consume_context():
    candidates = [item("a", 1, .9, "馆舍周二开放。"), item("b", 1, .89, "馆舍周二开放。"),
                  item("a", 1, .88, "重复编号"), item("c", 1, .87, "收费为十元。"),
                  item("d", 1, .86, " \n ")]
    selected = select_context_candidates(candidates, 8)
    assert [row["chunk_id"] for row in selected] == ["a:1", "c:1"]


def test_different_numbers_and_negation_are_not_deduplicated():
    candidates = [item("a", 1, .9, "票价为十元。"), item("b", 1, .89, "票价为百元。"),
                  item("c", 1, .88, "周一不开放。"), item("d", 1, .87, "周一开放。")]
    assert len(select_context_candidates(candidates, 8)) == 4


def test_stable_order_and_bound_for_one_document():
    candidates = [item("a", number, .9) for number in range(20)]
    assert select_context_candidates(candidates, 8) == candidates[:8]


def test_question_api_without_tuning_searches_whole_library_and_uses_server_policy():
    index = FakeIndex([item(f"doc{number}", 1, .9 - number / 100) for number in range(30)])
    rag = RagService(index, RagSettings(), FakeChat())
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(rag))
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "有哪些事项？", "model": "qwen3:1.7b"})
    assert response.status_code == 200
    result = response.json()["data"]
    assert index.calls == [("有哪些事项？", 24, None)]
    assert result["retrieval"]["scope"] == "all"
    assert result["retrieval"]["selection"] == "diverse"
    assert result["retrieval"]["candidate_limit"] == 24
    assert result["retrieval"]["returned_chunks"] == 24
    assert result["retrieval"]["used_chunks"] <= 8
    assert all(source["text"] in next(row["text"] for row in index.matches if row["chunk_id"] == source["chunk_id"])
               for source in result["sources"])


def test_explicit_debug_scope_and_top_k_remain_explicit():
    index = FakeIndex()
    service = RagService(index, RagSettings(), FakeChat())
    result = service.ask("问题", top_k=3, document_id="debug-document")
    assert index.calls == [("问题", 3, "debug-document")]
    assert result["retrieval"]["selection"] == "similarity"
    assert result["retrieval"]["scope"] == "document"
    assert result["retrieval"]["candidate_limit"] == 3


def test_omitted_top_k_with_explicit_debug_document_keeps_previous_default():
    index = FakeIndex()
    result = RagService(index, RagSettings(), FakeChat()).ask("问题", top_k=None, document_id="debug-document")
    assert index.calls == [("问题", 4, "debug-document")]
    assert result["retrieval"]["top_k"] == 4


def test_automatic_policy_honors_configuration():
    index = FakeIndex()
    settings = RagSettings(retrieval_candidates=12, context_chunks=6)
    result = RagService(index, settings, FakeChat()).ask("问题", top_k=None)
    assert index.calls == [("问题", 12, None)]
    assert result["retrieval"]["top_k"] == 6


@pytest.mark.parametrize("candidates,chunks", [(0, 1), (51, 8), (4, 8), (24, 11), (24, 0), (True, 1), (24, True), (24.5, 8)])
def test_invalid_auto_retrieval_configuration(candidates, chunks):
    with pytest.raises(ValueError):
        RagSettings(retrieval_candidates=candidates, context_chunks=chunks)


def test_auto_retrieval_environment(monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATES", "16")
    monkeypatch.setenv("RAG_CONTEXT_CHUNKS", "6")
    settings = RagSettings.from_env()
    assert (settings.retrieval_candidates, settings.context_chunks) == (16, 6)


def test_empty_library_does_not_generate_in_auto_mode():
    chat = FakeChat()
    result = RagService(FakeIndex([]), RagSettings(), chat).ask("问题", top_k=None)
    assert result["status"] == "insufficient_evidence"
    assert result["reason"] == "no_sources"
    assert not chat.calls


def test_real_persistent_store_supplies_multiple_documents_and_excludes_deleted(isolated_root):
    from local_rag.api.app import create_app
    chat = FakeChat()
    app = create_app(
        DocumentSettings(storage_dir=isolated_root / "documents"),
        VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3"),
        embedder=FakeEmbeddings(), rag_settings=RagSettings(), chat_client=chat,
    )
    def reply(model, prompt, **kwargs):
        data = json.loads(prompt)
        return json.dumps({"answerable": True, "answer": "".join(source["text"] for source in data["contexts"]),
                           "evidence": [{"source_id": source["source_id"], "quote": source["text"]}
                                        for source in data["contexts"]]}, ensure_ascii=False)
    with TestClient(app) as client:
        ids = []
        for name, text in [("north.txt", "北馆周一开放。"), ("south.txt", "南馆周二开放。")]:
            response = client.post("/documents", files={"file": (name, text.encode("utf-8"), "text/plain")})
            assert response.status_code == 201
            ids.append(response.json()["data"]["document_id"])
        with patch.object(chat, "chat", side_effect=reply):
            result = client.post("/chat", json={"question": "两个场馆分别哪天开放？"}).json()["data"]
        assert result["status"] == "answered"
        assert result["retrieval"]["scope"] == "all"
        assert {source["metadata"]["document_id"] for source in result["sources"]} == set(ids)
        assert result["evidence"] == []
        assert "北馆周一开放" in result["answer"] and "南馆周二开放" in result["answer"]
        assert client.delete(f"/documents/{ids[1]}").status_code == 200
        with patch.object(chat, "chat", side_effect=reply):
            result = client.post("/chat", json={"question": "场馆哪天开放？"}).json()["data"]
        assert {source["metadata"]["document_id"] for source in result["sources"]} == {ids[0]}
