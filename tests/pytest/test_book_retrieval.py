"""Whole-book retrieval: deterministic contracts on an isolated Chinese corpus."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from unittest.mock import patch

import pytest

from local_rag.core.config import DocumentSettings, RagSettings, VectorSettings
from local_rag.documents.service import DocumentStore
from local_rag.documents.knowledge import build_knowledge
from local_rag.indexing.service import IndexService
from local_rag.rag.book import analyze_question, reciprocal_rank_fusion, knowledge_contexts
from local_rag.rag.scenarios import plan_retrieval, recognize_scene
from local_rag.storage.knowledge import KnowledgeStore
from tests.fixtures.embeddings import FakeEmbeddings

BOOK = """第1章 相识
其人姓关名羽，字长生，后改云长。姓刘名备，字玄德。姓曹名操字孟德。
刘备与关羽结义为兄弟。
关公与云长在本书中指同一位将领，云长离开村庄。
云长又称关公，关公护送车队。
关公即云长，云长救助同行者。
第2章 效力
建安三年，云长投奔曹操。曹操安排云长守城。
第3章 分离
次日，云长离开曹操，前往寻找玄德。
第4章 联合
刘备率军救援关羽。二人合力守住南门。
第5章 交战
刘备与曹操并非父子，二人交战。
第6章 归来
建安五年，关羽回到刘备身边。物资登记编号为WX-9175。
"""


@pytest.fixture
def book_index(isolated_root):
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "documents", chunk_size=100, chunk_overlap=20))
    settings = VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3")
    index = IndexService(store, settings, FakeEmbeddings())
    record = index.ingest(io.BytesIO(BOOK.encode()), "story.txt", "text/plain")
    yield index, record
    index.close()


def payload(index, record):
    return index.knowledge.catalog([index.registry.get(record["document_id"])])[record["document_id"]]


@pytest.mark.parametrize("query,scene", [
    ("关羽的时间线", "timeline"), ("关羽和刘备是什么关系", "relationship"),
    ("关羽有哪些别名", "alias"), ("全书中关羽前后有哪些变化", "cross_chapter"),
    ("跨章节分析曹操和刘备", "cross_chapter"),
])
def test_book_scenes(query, scene):
    assert recognize_scene(query)[0] == scene


def test_explicit_aliases_are_book_local_and_title_hints_are_marked(book_index):
    index, record = book_index
    book = payload(index, record)
    entity = next(e for e in book["entities"] if e["name"] == "关羽")
    assert {"长生", "云长"} <= set(entity["aliases"])
    assert "关公" in entity["retrieval_aliases"]
    assert entity["alias_candidates"][0]["method"] == "same_sentence_title_candidate"
    assert entity["alias_candidates"][0]["support_count"] >= 3
    _, text, _ = index.documents.hierarchy(record["document_id"])
    for declaration in entity["declarations"]:
        excerpt = text[declaration["start_index"]:declaration["end_index"]]
        assert declaration["alias"] in excerpt
    analysis = analyze_question("云长和玄德的关系", {record["document_id"]: book})
    assert {e["name"] for e in analysis["entities"]} == {"关羽", "刘备"}


def test_ambiguous_alias_is_not_silently_merged(isolated_root):
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "docs"))
    record = store.ingest(io.BytesIO("张华，字明远。李强，字明远。二人不同。".encode()), "ambiguous.txt", "text/plain")
    bundle = store.hierarchy(record["document_id"])
    book = build_knowledge(*bundle)
    assert "明远" in book["ambiguous_aliases"]
    analysis = analyze_question("明远做了什么", {record["document_id"]: book})
    assert analysis["entities"] == []
    assert analysis["ambiguous_aliases"]


def test_bm25_retrieves_a_rare_identifier_even_when_dense_ranking_misses_it(book_index):
    index, record = book_index
    semantic = index.search("WX-9175", 1)
    wrong = next(c for c in record["chunks"] if "9175" not in c["text"])
    semantic["matches"] = [{**wrong, "score": .9, "distance": .1, "model_version": semantic["model"]["version"]}]
    with patch.object(index, "search", return_value=semantic):
        result = index.hybrid_search("WX-9175", 5)
    assert any("9175" in hit["text"] for hit in result["matches"])
    assert any("keyword" in hit["metadata"]["retrieval_channels"] for hit in result["matches"])
    assert result["strategy"]["fusion"] == "rrf_k60"


def test_rrf_uses_ranks_and_deduplicates_each_channel():
    def hit(key): return {"chunk_id": key, "metadata": {}, "score": 999, "text": key}
    result = reciprocal_rank_fusion([("semantic", [hit("a"), hit("a"), hit("b")]),
                                     ("keyword", [hit("b"), hit("a")])], 3)
    assert {r["chunk_id"] for r in result} == {"a", "b"}
    a = next(r for r in result if r["chunk_id"] == "a")
    assert a["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert a["metadata"]["score_kind"] == "rrf_normalized"
    assert reciprocal_rank_fusion([], 5) == []


def test_relations_are_source_clues_and_preserve_negation(book_index):
    index, record = book_index
    result = index.retrieve_context("刘备和曹操的关系", 24, None, plan_retrieval("刘备和曹操的关系", RagSettings()), RagSettings())
    spans = [s for s in result["contexts"] if s["metadata"].get("analysis_type") == "relationship"]
    assert spans
    assert any("并非父子" in s["text"] and s["metadata"]["negated"] for s in spans)
    assert all(s["metadata"]["relation_clues"] for s in spans)


def test_timeline_uses_narrative_order_and_original_time_mentions(book_index):
    index, record = book_index
    query = "云长的时间线"
    result = index.retrieve_context(query, 24, None, plan_retrieval(query, RagSettings()), RagSettings())
    spans = result["contexts"]
    assert len({s["metadata"]["section_id"] for s in spans}) >= 4
    assert [s["metadata"]["start_index"] for s in spans] == sorted(s["metadata"]["start_index"] for s in spans)
    assert all(s["metadata"]["ordering"] == "document_narrative_order" for s in spans)
    assert any(m["kind"] == "relative" for s in spans for m in s["metadata"]["time_mentions"])
    _, text, _ = index.documents.hierarchy(record["document_id"])
    for source in spans:
        meta = source["metadata"]
        assert source["text"] == text[meta["start_index"]:meta["end_index"]]
        for time in meta["time_mentions"]:
            assert time["text"] in source["text"]


def test_cross_chapter_plan_has_bounded_variants_and_multi_chapter_context(book_index):
    index, record = book_index
    question = "跨章节分析云长从效力曹操到回归刘备的经历"
    result = index.retrieve_context(question, 24, None, plan_retrieval(question, RagSettings()), RagSettings())
    assert 1 < len(result["strategy"]["query_variants"]) <= 8
    assert result["strategy"]["chapters_considered"] == 6
    assert len({s["metadata"]["section_id"] for s in result["contexts"]}) >= 2
    assert len(result["contexts"]) <= 8


def test_derived_index_survives_restart_rebuild_and_delete(book_index):
    index, record = book_index
    identity = record["document_id"]
    path = index.knowledge.path
    reopened = KnowledgeStore(path)
    rows = [index.registry.get(identity)]
    assert reopened.search("WX-9175", rows)
    old_generation = rows[0]["generation"]
    index.reindex(identity)
    assert index.registry.get(identity)["generation"] != old_generation
    with reopened.connect() as db:
        assert db.execute("SELECT count(*) FROM books WHERE document_id=?", (identity,)).fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM book_chunks WHERE document_id=?", (identity,)).fetchone()[0] == record["chunk_count"]
    index.delete(identity)
    with reopened.connect() as db:
        assert db.execute("SELECT count(*) FROM books").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM book_chunks").fetchone()[0] == 0
    assert index.hybrid_search("云长")["matches"] == []


def test_unpublished_document_is_invisible_to_keyword_channel(book_index):
    index, record = book_index
    identity = record["document_id"]
    index.registry.update(identity, status="indexing", generation=None)
    assert index.hybrid_search("WX-9175")["matches"] == []


def test_document_filter_never_expands_to_a_different_book(book_index):
    index, record = book_index
    other = index.ingest(io.BytesIO("张华，字云长。张华管理茶仓，登记编号为CC-555。".encode()), "other.txt", "text/plain")
    result = index.hybrid_search("云长", 10, other["document_id"])
    assert result["matches"]
    assert all(s["metadata"]["document_id"] == other["document_id"] for s in result["matches"])
    assert all(e["name"] != "关羽" for e in result["strategy"]["resolved_entities"])


def test_optional_features_can_be_disabled_for_baseline(book_index):
    index, record = book_index
    settings = replace(RagSettings(), hybrid_search=False, entity_retrieval=False, cross_chapter=False)
    result = index.retrieve_context("云长做了什么", 24, None, plan_retrieval("云长做了什么", settings), settings)
    assert result["strategy"]["fusion"] == "semantic"
    assert result["strategy"]["resolved_entities"] == []
    assert not result["strategy"]["structured_contexts"]


@pytest.mark.parametrize("name", ["hybrid_search", "entity_retrieval", "cross_chapter"])
def test_feature_flags_require_booleans(name):
    with pytest.raises(ValueError):
        RagSettings(**{name: "false"})

def test_name_declarations_ignore_annotations_and_remove_role_prefixes(isolated_root):
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "docs"))
    text = ("太守陶谦，字恭祖。吴人张温，字惠恕。一是丁谧，字彦靖。"
            "参军傅干，字彦材。手指（上雨下单）。只待半酣（喝酒过半）。"
            "刘备（字玄德），领兵归来。")
    record = store.ingest(io.BytesIO(text.encode()), "names.txt", "text/plain")
    book = build_knowledge(*store.hierarchy(record["document_id"]))
    assert {e["name"] for e in book["entities"]} == {"陶谦", "张温", "丁谧", "傅干", "刘备"}
    assert not book["ambiguous_aliases"]

def test_timeline_keeps_dates_and_outcomes_in_adjacent_sentences(isolated_root):
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "docs"))
    text = "第1章 启程\n林澄，字望舒。建安三年，望舒赴石桥修复水闸。林澄救出船工。\n第2章 返乡\n建安四年，望舒返回青浦。林澄归还器材。"
    record = store.ingest(io.BytesIO(text.encode()), "timeline.txt", "text/plain")
    bundle = store.hierarchy(record["document_id"])
    book = build_knowledge(*bundle)
    query = "林澄的时间线"
    catalog = {record["document_id"]: book}
    contexts = knowledge_contexts(query, analyze_question(query, catalog), catalog,
                                 {record["document_id"]: bundle}, plan_retrieval(query, RagSettings()), 8)
    combined = "".join(s["text"] for s in contexts)
    assert "赴石桥修复水闸" in combined and "返回青浦" in combined
    assert {t["text"] for s in contexts for t in s["metadata"]["time_mentions"]} >= {"建安三年", "建安四年"}
    for s in contexts:
        assert s["text"] == bundle[1][s["metadata"]["start_index"]:s["metadata"]["end_index"]]

def test_relation_query_does_not_use_only_one_of_two_requested_people(book_index):
    index, record = book_index
    book = payload(index, record)
    query = "关羽和曹操是什么关系"
    catalog = {record["document_id"]: book}
    contexts = knowledge_contexts(query, analyze_question(query, catalog), catalog,
                                 {record["document_id"]: index.documents.hierarchy(record["document_id"])},
                                 plan_retrieval(query, RagSettings()), 8)
    assert contexts
    assert not any("结义为兄弟" in s["text"] for s in contexts)

def test_book_context_budget_preserves_adjacent_event_facts(isolated_root):
    from pathlib import Path
    from local_rag.core.config import PROJECT_ROOT
    from local_rag.rag.service import RagService
    dataset = json.loads((PROJECT_ROOT / "benchmarks/book_retrieval_v1/dataset.json").read_text(encoding="utf-8"))
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "docs", chunk_size=160, chunk_overlap=30))
    index = IndexService(store, VectorSettings(chroma_dir=isolated_root / "chroma",
                                              registry_path=isolated_root / "registry.sqlite3"), FakeEmbeddings())
    try:
        doc = dataset["documents"][0]
        record = index.ingest(io.BytesIO(doc["text"].encode()), doc["filename"], "text/markdown")
        case = next(c for c in dataset["cases"] if c["id"] == "timeline")
        config = RagSettings()
        retrieved = index.retrieve_context(case["question"], 24, None, plan_retrieval(case["question"], config), config)
        sources = RagService(index, config)._sources(case["question"], retrieved["contexts"], balanced=True)
        actual = "\n".join(s["text"] for s in sources)
        assert all(any(alias in actual for alias in fact) for fact in case["facts"])
        _, original, _ = index.documents.hierarchy(record["document_id"])
        assert all(s["text"] == original[s["metadata"]["start_index"]:s["metadata"]["end_index"]] for s in sources)
    finally:
        index.close()

def test_alias_context_keeps_the_complete_changed_name_declaration(book_index):
    index, record = book_index
    query = "关羽有哪些别名"
    result = index.retrieve_context(query, 24, None, plan_retrieval(query, RagSettings()), RagSettings())
    text = "\n".join(s["text"] for s in result["contexts"])
    assert "长生" in text and "后改云长" in text
    assert all(s["metadata"]["analysis_type"] == "alias" for s in result["contexts"])

def test_chapter_heading_does_not_create_a_false_person_event(isolated_root):
    store = DocumentStore(DocumentSettings(storage_dir=isolated_root / "docs"))
    text = "第1章 初见\n林澄，字望舒。林澄出发。\n第2章 林澄大败敌军\n陈宁率军赴北仓。"
    record = store.ingest(io.BytesIO(text.encode()), "headings.txt", "text/plain")
    bundle = store.hierarchy(record["document_id"])
    knowledge = build_knowledge(*bundle)
    assert not any(f["section_order"] == 2 for f in knowledge["facts"])

def test_relationship_ranking_prefers_explicit_kinship_over_cooperation(book_index):
    index, record = book_index
    query = "云长和玄德的关系"
    result = index.retrieve_context(query, 24, None, plan_retrieval(query, RagSettings()), RagSettings())
    assert "结义为兄弟" in result["contexts"][0]["text"]
    people = result["contexts"][0]["metadata"]["query_people"]
    assert {p["name"] for p in people} == {"关羽", "刘备"}
