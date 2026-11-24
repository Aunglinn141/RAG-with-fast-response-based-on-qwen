"""Scenario routing, source ranges and persistent hierarchy lifecycle contracts."""

from copy import deepcopy
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from local_rag.core.config import DocumentSettings, RagSettings, VectorSettings
from local_rag.documents.service import DocumentStore
from local_rag.documents.hierarchy import build_hierarchy, extract_summary
from local_rag.indexing.service import IndexService
from local_rag.rag.context import expand_context
from local_rag.rag.scenarios import plan_retrieval, recognize_scene
from local_rag.rag.service import RagService
from tests.fixtures.embeddings import FakeEmbeddings
from tests.unit.test_rag import FakeChat, FakeIndex


@pytest.mark.parametrize("question,scene", [
    ("关公做了些什么？", "overview"), ("总结主要经历", "overview"),
    ("概括这一章", "overview"), ("北馆和南馆分别哪天开放？", "comparison"),
    ("比较两种办理流程", "comparison"), ("如何办理申请？", "procedure"),
    ("操作步骤是什么？", "procedure"), ("申请费是多少元？", "fact"),
    ("你好", "fact"), ("How to submit the form?", "procedure"),
    ("Summarize the story", "overview"),
])
def test_scenario_rules(question, scene):
    assert recognize_scene(question)[0] == scene


def document(store, name="story.txt"):
    text = ("第1章 筹备\n" + "小林在北馆准备展览，先登记借用设备，再核对清单。临时展台只在周二开放，周一不开放。\n" * 8
            + "第2章 实施\n" + "小林在南馆组织讲解，安排同伴接待访客。南馆每周四开放，报名费为十元。\n" * 8)
    record = store.ingest(io.BytesIO(text.encode("utf-8")), name, "text/plain")
    return record


@pytest.fixture
def stored(isolated_root):
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "docs", chunk_size=100, chunk_overlap=20))
    record = document(store)
    return store, record


def hit(chunk, score=.9):
    return {**deepcopy(chunk), "score": score, "distance": 1 - score, "model_version": "test"}


def assert_origins(sources, text, record):
    originals = {c["chunk_id"]: c for c in record["chunks"]}
    for source in sources:
        meta = source["metadata"]
        assert text[meta["start_index"]:meta["end_index"]] == source["text"]
        assert meta["document_id"] == record["document_id"]
        for key in meta["source_chunk_ids"]:
            assert key in originals
            chunk = originals[key]["metadata"]
            assert chunk["start_index"] < meta["end_index"] and chunk["end_index"] > meta["start_index"]


def test_children_map_to_persistent_parents_and_real_chapter_summaries(stored):
    store, record = stored
    _, text, structure = store.hierarchy(record["document_id"])
    assert len(structure["sections"]) == 2
    assert set(structure["children"]) == {c["chunk_id"] for c in record["chunks"]}
    parents = {p["parent_id"]: p for p in structure["parents"]}
    for child in record["chunks"]:
        assert structure["children"][child["chunk_id"]]
        for parent_id in structure["children"][child["chunk_id"]]:
            assert child["chunk_id"] in parents[parent_id]["child_ids"]
    for section in structure["sections"]:
        assert section["summary_method"] == "extractive-v2"
        assert section["summary"] == "\n".join(text[s["start_index"]:s["end_index"]] for s in section["summary_spans"])
        assert all(section["start_index"] <= s["start_index"] < s["end_index"] <= section["end_index"] for s in section["summary_spans"])
    assert (store.directory(record["document_id"]) / "retrieval.json").is_file()


def test_markdown_and_docx_units_are_boundaries(isolated_root):
    from local_rag.documents.models import Extraction, TextUnit
    from local_rag.documents.splitters import split_document
    for extension in (".md", ".docx", ".pdf"):
        extraction = Extraction([TextUnit("甲章。" * 300, page_number=1 if extension == ".pdf" else None,
                                         section_title="甲章"),
                                 TextUnit("乙章。" * 300, page_number=2 if extension == ".pdf" else None,
                                          section_title="乙章")])
        text, chunks = split_document(extraction, "unit-doc", "book" + extension, extension, 100, 20)
        record = {"document_id": "unit-doc", "chunks": chunks}
        hierarchy = build_hierarchy(record, text)
        assert len(hierarchy["sections"]) == 2
        boundary = hierarchy["sections"][1]["start_index"]
        assert not any(p["start_index"] < boundary < p["end_index"] for p in hierarchy["parents"])


def test_cached_hierarchy_reused_and_bad_cache_rebuilt(stored):
    store, record = stored
    first = store.hierarchy(record["document_id"])
    path = store.directory(record["document_id"]) / "retrieval.json"
    stamp = path.stat().st_mtime_ns
    assert store.hierarchy(record["document_id"]) == first
    assert path.stat().st_mtime_ns == stamp
    path.write_text("{invalid", encoding="utf-8")
    assert store.hierarchy(record["document_id"]) == first
    changed = store.hierarchy(record["document_id"], 1000)
    assert changed[2]["fingerprint"] != first[2]["fingerprint"]


def test_invalid_source_offsets_are_not_guessed(stored):
    store, record = stored
    text = (store.directory(record["document_id"]) / "text.txt").read_text(encoding="utf-8")
    record["chunks"][0]["metadata"]["start_index"] += 1
    with pytest.raises(ValueError, match="positions"):
        build_hierarchy(record, text)


def test_parent_lookup_returns_more_than_child_without_crossing_chapter(stored):
    store, record = stored
    data = store.hierarchy(record["document_id"])
    anchor = hit(record["chunks"][1])
    plan = plan_retrieval("报名费是多少？", RagSettings())
    contexts = expand_context("报名费是多少？", [anchor], {record["document_id"]: data}, plan, 8)
    assert len(contexts) == 1
    assert len(contexts[0]["text"]) > len(anchor["text"])
    assert "第2章" not in contexts[0]["text"]
    assert contexts[0]["metadata"]["context_type"] == "parent"
    assert_origins(contexts, data[1], record)


def test_neighbors_extend_child_and_zero_radius_is_an_ablation(stored):
    store, record = stored
    data = store.hierarchy(record["document_id"])
    anchor = hit(record["chunks"][2])
    def selected(radius):
        plan = plan_retrieval("如何准备？", RagSettings(parent_child=False, neighbor_radius=radius))
        return expand_context("如何准备？", [anchor], {record["document_id"]: data}, plan, 8)[0]
    plain, extended = selected(0), selected(1)
    assert plain["text"] == anchor["text"]
    assert len(extended["text"]) > len(plain["text"])
    assert extended["metadata"]["neighbor_chunk_ids"]
    assert extended["metadata"]["context_type"] == "neighbor"
    assert_origins([extended], data[1], record)


def test_neighbor_at_first_and_last_chunk_stays_in_document(stored):
    store, record = stored
    data = store.hierarchy(record["document_id"])
    plan = plan_retrieval("如何准备？", RagSettings(parent_child=False, neighbor_radius=2))
    contexts = expand_context("如何准备？", [hit(record["chunks"][0]), hit(record["chunks"][-1])],
                              {record["document_id"]: data}, plan, 8)
    assert len(contexts) == 2
    assert "第2章" not in contexts[0]["text"]
    assert "第1章" not in contexts[1]["text"]
    assert_origins(contexts, data[1], record)


def test_overview_uses_summaries_from_multiple_chapters_and_fact_does_not(stored):
    store, record = stored
    data = store.hierarchy(record["document_id"])
    matches = [hit(record["chunks"][0]), hit(record["chunks"][-1], .89)]
    overview = expand_context("概括小林做了什么", matches, {record["document_id"]: data},
                              plan_retrieval("概括小林做了什么", RagSettings()), 8)
    summaries = [s for s in overview if s["metadata"]["context_type"] == "summary"]
    assert len({s["metadata"]["section_id"] for s in summaries}) == 2
    assert_origins(overview, data[1], record)
    fact = expand_context("报名费多少？", matches, {record["document_id"]: data},
                          plan_retrieval("报名费多少？", RagSettings()), 8)
    assert not any(s["metadata"]["context_type"] == "summary" for s in fact)


def test_extracts_keep_negation_and_numbers_and_have_no_fabricated_text():
    text = "北馆周一不开放。南馆周四开放，费用为十元。"
    summary = extract_summary(text, 0, len(text), "北馆哪天不开放？")
    output = [text[s["start_index"]:s["end_index"]] for s in summary]
    assert "北馆周一不开放。" in output
    assert "南馆周四开放，费用为十元。" in output


def test_overlapping_child_hits_share_one_parent(stored):
    store, record = stored
    data = store.hierarchy(record["document_id"])
    matches = [hit(record["chunks"][i]) for i in (0, 1, 2)]
    contexts = expand_context("报名费？", matches, {record["document_id"]: data},
                              plan_retrieval("报名费？", RagSettings()), 8)
    assert len(contexts) == 1


def test_balanced_budget_crops_around_hit_and_keeps_exact_offsets(stored):
    store, record = stored
    data = store.hierarchy(record["document_id"])
    matches = [hit(record["chunks"][2]), hit(record["chunks"][-2])]
    contexts = expand_context("南馆报名费是多少？", matches, {record["document_id"]: data},
                              plan_retrieval("南馆报名费是多少？", RagSettings()), 8)
    service = RagService(FakeIndex(), RagSettings(max_context_characters=180), FakeChat())
    sources = service._sources("南馆报名费是多少？", contexts, balanced=True)
    assert len(sources) == 2
    assert sum(len(s["text"]) for s in sources) <= 180
    assert_origins(sources, data[1], record)
    assert all(s["truncated"] for s in sources)


def test_cached_positions_refresh_when_source_is_replaced(stored):
    store, record = stored
    before = store.hierarchy(record["document_id"])[2]["fingerprint"]
    path = store.directory(record["document_id"])
    text = (path / "text.txt").read_text(encoding="utf-8").replace("十元", "百元")
    for child in record["chunks"]:
        child["text"] = child["text"].replace("十元", "百元")
    (path / "text.txt").write_text(text, encoding="utf-8")
    (path / "document.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    after = store.hierarchy(record["document_id"])
    assert after[2]["fingerprint"] != before
    assert "百元" in after[2]["sections"][-1]["summary"]


@pytest.mark.parametrize("kwargs", [
    {"parent_characters": 999}, {"parent_characters": 6001}, {"parent_characters": True},
    {"neighbor_radius": -1}, {"neighbor_radius": 3}, {"neighbor_radius": True},
    {"enhanced_retrieval": "true"}, {"parent_child": 1}, {"section_summaries": None},
])
def test_invalid_retrieval_configuration(kwargs):
    with pytest.raises(ValueError):
        RagSettings(**kwargs)


def test_environment_and_summary_ablation(monkeypatch):
    monkeypatch.setenv("RAG_ENHANCED_RETRIEVAL", "false")
    monkeypatch.setenv("RAG_PARENT_CHILD", "false")
    monkeypatch.setenv("RAG_PARENT_CHARACTERS", "1800")
    monkeypatch.setenv("RAG_NEIGHBOR_RADIUS", "0")
    monkeypatch.setenv("RAG_SECTION_SUMMARIES", "false")
    settings = RagSettings.from_env()
    assert not settings.enhanced_retrieval
    plan = plan_retrieval("总结全文", settings)
    assert not plan.parent_child and not plan.section_summaries and plan.neighbor_radius == 0
    monkeypatch.setenv("RAG_SECTION_SUMMARIES", "maybe")
    with pytest.raises(ValueError):
        RagSettings.from_env()


def test_persistent_lifecycle_reindex_delete_and_api_diagnostics(isolated_root):
    from local_rag.api.app import create_app
    from fastapi.testclient import TestClient
    ds = DocumentSettings(storage_dir=isolated_root / "docs", chunk_size=100, chunk_overlap=20)
    vs = VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3")
    chat = FakeChat()
    app = create_app(ds, vs, embedder=FakeEmbeddings(), rag_settings=RagSettings(), chat_client=chat)
    with TestClient(app) as client:
        text = "第1章 北馆\n北馆周二开放。\n第2章 南馆\n南馆周四开放。"
        response = client.post("/documents", files={"file": ("guide.txt", text.encode("utf-8"), "text/plain")})
        assert response.status_code == 201
        doc_id = response.json()["data"]["document_id"]
        cache = ds.storage_dir / doc_id / "retrieval.json"
        assert cache.exists()
        def reply(model, prompt, **kwargs):
            contexts = json.loads(prompt)["contexts"]
            return json.dumps({"answerable": True, "answer": "场馆按规定开放。",
                               "evidence": [{"source_id": contexts[0]["source_id"], "quote": contexts[0]["text"]}]}, ensure_ascii=False)
        with patch.object(chat, "chat", side_effect=reply):
            result = client.post("/chat", json={"question": "概括场馆的开放安排"}).json()["data"]
        assert result["status"] == "answered"
        assert result["retrieval"]["strategy"]["scene"] == "overview"
        assert result["retrieval"]["context_types"]["summary"] > 0
        assert result["retrieval"]["used_sections"] == 2
        assert client.post(f"/documents/{doc_id}/reindex").status_code == 200
        count = app.state.index_service.vectors.count_document(doc_id)
    # A new service reads the same children and persisted parent/summary cache.
    index = IndexService(DocumentStore(ds), vs, FakeEmbeddings())
    try:
        result = index.retrieve_context("概括场馆安排", 24, None, plan_retrieval("概括场馆安排", RagSettings()), RagSettings())
        assert result["contexts"]
        assert index.vectors.count_document(doc_id) == count
        index.delete(doc_id)
        assert not cache.exists()
        assert index.vectors.count_document(doc_id) == 0
        assert not index.retrieve_context("概括场馆安排", 24, None, plan_retrieval("概括场馆安排", RagSettings()), RagSettings())["contexts"]
    finally:
        index.close()

def test_summary_prefers_informative_actions_over_short_reactions():
    text = "关公大惊。关公问其姓名。关公在发现埋伏之后，拔剑击退刀斧手，并持刀追赶卞喜，保护随行车队继续前行。"
    spans = extract_summary(text, 0, len(text), "关公做了些什么？", max_sentences=1)
    assert "击退刀斧手" in text[spans[0]["start_index"]:spans[0]["end_index"]]


def test_overview_prompt_preserves_sections_and_allows_natural_synthesis():
    from tests.unit.test_rag import match
    chat = FakeChat()
    rag = RagService(FakeIndex(), RagSettings(), chat)
    source = {**match(), "source_id": "S1"}
    source["metadata"].update(context_type="summary", section_title="第一章")
    rag._generate("概括主要事项", [source], "qwen3:1.7b", comprehensive=True)
    call = chat.calls[-1]
    assert "evidence" not in call["response_format"]["properties"]
    assert call["response_format"]["properties"]["answer"]["maxLength"] == 600
    assert "动作的执行者和承受者" in call["system_prompt"]
    assert json.loads(call["prompt"])["contexts"][0]["section"] == "第一章"
