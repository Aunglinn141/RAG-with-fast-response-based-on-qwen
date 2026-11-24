"""Artifact facts, provenance, ambiguity and persistent index lifecycle."""

from io import BytesIO

from fastapi.testclient import TestClient
import pytest

from local_rag.api.app import create_app
from local_rag.artifacts.models import ArtifactWrite
from local_rag.artifacts.service import ArtifactService
from local_rag.core.config import DocumentSettings, RagSettings, RuntimeSettings, VectorSettings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.service import DocumentStore
from local_rag.indexing.service import IndexService
from tests.fixtures.embeddings import FakeEmbeddings
from tests.unit.test_rag import FakeChat


ARCHIVE = """以下为软件测试用虚构文物档案，不代表真实馆藏。
名称：测试云纹壶
年代：测试时期
材料：青铜
用途：用于测试礼仪场景。
文化价值：用于验证档案证据的保存与追溯。
收藏博物馆：测试甲博物馆
馆藏编号：TEST-001
别名：测试云壶
"""
LABELS = {"名称": "name", "年代": "era", "材料": "material", "用途": "purpose", "文化价值": "cultural_value",
          "收藏博物馆": "museum", "馆藏编号": "accession_number", "别名": "alias"}


@pytest.fixture
def catalog(isolated_root):
    settings = VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3", auto_index=False)
    index = IndexService(DocumentStore(DocumentSettings(storage_dir=isolated_root / "documents")), settings, FakeEmbeddings())
    service = ArtifactService(index)
    index.artifacts = service
    try:
        yield service
    finally:
        index.close()


def source_request(service, text=ARCHIVE, source_type="official_archive", build_index=False):
    doc = service.index.ingest(BytesIO(text.encode("utf-8")), "虚构测试档案.txt", "text/plain")
    claims = []
    for line in text.splitlines():
        if "：" not in line:
            continue
        label, value = line.split("：", 1)
        if label in LABELS:
            claims.append({"field": LABELS[label], "value": value, "document_id": doc["document_id"],
                           "quote": line, "source_type": source_type})
    return ArtifactWrite(claims=claims, build_index=build_index), doc


@pytest.mark.parametrize("question,field,expected", [
    ("叫什么名字？", "name", "测试云纹壶"), ("是什么朝代的？", "era", "测试时期"),
    ("用什么材料做的？", "material", "青铜"), ("它用来做什么？", "purpose", "用于测试礼仪场景。"),
    ("有什么文化价值？", "cultural_value", "用于验证档案证据的保存与追溯。"),
    ("保存在哪一个博物馆中？", "museum", "测试甲博物馆"),
    ("馆藏编号是什么？", "accession_number", "TEST-001"),
])
def test_basic_answers_are_literal_sourced_facts_without_models(catalog, question, field, expected):
    request, doc = source_request(catalog)
    record = catalog.save(request)
    before = catalog.index.embedder.calls
    answer = catalog.ask(question, record["artifact_id"])
    assert answer["status"] == "answered"
    assert answer["facts"][field]["value"] == expected
    assert expected in answer["answer"]
    assert answer["sources"][0]["document_id"] == doc["document_id"]
    assert answer["sources"][0]["chunk_ids"]
    assert catalog.index.embedder.calls == before


def test_system_identifier_is_stable_and_not_museum_number(catalog):
    request, _ = source_request(catalog)
    first = catalog.save(request)
    second = catalog.save(request)
    assert first["artifact_id"].startswith("WW-")
    assert first["artifact_id"] != first["facts"]["accession_number"]["value"]
    assert first["artifact_id"] == second["artifact_id"]
    assert second["revision"] == 1
    assert catalog.list()["total"] == 1


def test_bad_quote_or_invented_value_never_writes(catalog):
    request, _ = source_request(catalog)
    request.claims[1].value = "汉代"
    with pytest.raises(DocumentError, match="逐字"):
        catalog.save(request)
    request.claims[1].quote = "年代：汉代"
    with pytest.raises(DocumentError, match="原文不一致"):
        catalog.save(request)
    assert catalog.list()["total"] == 0


def test_repeated_quote_requires_unambiguous_offset(catalog):
    request, _ = source_request(catalog, ARCHIVE + "\n材料：青铜\n")
    with pytest.raises(DocumentError, match="出现多次"):
        catalog.save(request)
    for claim in request.claims:
        if claim.field == "material":
            claim.start_index = ARCHIVE.index("材料：青铜")
    saved = catalog.save(request)
    assert len(saved["facts"]["material"]["claims"]) == 1


def test_missing_facts_and_unsupported_questions_abstain(catalog):
    request, _ = source_request(catalog)
    request.claims = [c for c in request.claims if c.field not in {"material", "museum"}]
    record = catalog.save(request)
    result = catalog.ask("材料是什么，保存在哪个博物馆？", record["artifact_id"])
    assert result["status"] == "insufficient_evidence"
    assert result["answer"].count("暂无资料") == 2
    assert result["sources"] == []
    assert catalog.ask("年代和材料是什么？", record["artifact_id"])["status"] == "partial"
    assert catalog.ask("重量是多少？", record["artifact_id"])["status"] == "insufficient_evidence"
    assert catalog.ask("年代和重量？", record["artifact_id"])["status"] == "partial"


def test_same_name_and_alias_need_identity_or_museum(catalog):
    first, _ = source_request(catalog)
    a = catalog.save(first)
    second, _ = source_request(catalog, ARCHIVE.replace("测试甲博物馆", "测试乙博物馆"))
    b = catalog.save(second)
    for query in ("测试云纹壶的年代？", "测试云壶是什么材料？"):
        answer = catalog.ask(query)
        assert answer["status"] == "needs_clarification"
        assert {c["artifact_id"] for c in answer["candidates"]} == {a["artifact_id"], b["artifact_id"]}
    result = catalog.ask("测试云壶的年代？", museum="测试乙博物馆")
    assert result["status"] == "answered"
    assert result["artifact_id"] == b["artifact_id"]


def test_conflict_is_shown_even_if_one_source_is_official(catalog):
    request, _ = source_request(catalog)
    other, _ = source_request(catalog, ARCHIVE.replace("材料：青铜", "材料：铜"), "museum_guide")
    request.claims += [c for c in other.claims if c.field == "material"]
    record = catalog.save(request)
    assert record["facts"]["material"]["status"] == "conflict"
    assert record["facts"]["material"]["value"] is None
    answer = catalog.ask("材料？", record["artifact_id"])
    assert answer["status"] == "insufficient_evidence"
    assert len(answer["sources"]) == 2
    assert "资料存在冲突" in answer["answer"]


def test_secondary_sources_cannot_be_promoted_and_are_qualified(catalog):
    request, _ = source_request(catalog, "学习型资料 · 非馆方原版出版物\n" + ARCHIVE)
    with pytest.raises(DocumentError, match="二次汇编"):
        catalog.save(request)
    for c in request.claims:
        c.source_type = "secondary"
    record = catalog.save(request)
    assert record["facts"]["museum"]["status"] == "secondary_only"
    assert "尚未经馆方原档案核验" in catalog.ask("在哪个博物馆？", record["artifact_id"])["answer"]


def test_source_deletion_invalidates_facts_and_cleans_artifact_vectors(catalog):
    request, doc = source_request(catalog, build_index=True)
    record = catalog.save(request)
    assert record["index_status"] == "indexed"
    assert catalog.vectors.count_document(record["artifact_id"]) == 8
    catalog.index.delete(doc["document_id"])
    record = catalog.get(record["artifact_id"])
    assert record["index_status"] == "stale"
    assert catalog.vectors.count_document(record["artifact_id"]) == 0
    result = catalog.ask("在哪个博物馆？", record["artifact_id"])
    assert result["status"] == "insufficient_evidence"
    assert result["facts"]["museum"]["status"] == "source_missing"
    assert not catalog.search("青铜")["candidates"]


def test_revisions_duplicates_and_failed_rebuild_preserve_facts(catalog, monkeypatch):
    request, _ = source_request(catalog, build_index=True)
    record = catalog.save(request)
    with pytest.raises(DocumentError, match="expected_revision"):
        catalog.save(request, record["artifact_id"])
    request.expected_revision = record["revision"]
    request.claims = [c for c in request.claims if c.field != "purpose"]
    def unavailable(*args):
        raise RuntimeError("model disconnected")
    monkeypatch.setattr(catalog.index.embedder, "embed_documents", unavailable)
    updated = catalog.save(request, record["artifact_id"])
    assert updated["revision"] == 2
    assert updated["index_status"] == "failed"
    assert updated["index_error"]
    assert catalog.vectors.count_document(record["artifact_id"]) == 0
    assert catalog.ask("用途？", record["artifact_id"])["status"] == "insufficient_evidence"
    assert catalog.ask("年代？", record["artifact_id"])["status"] == "answered"


def test_duplicate_museum_accession_rejected(catalog):
    first, _ = source_request(catalog)
    catalog.save(first)
    second, _ = source_request(catalog, ARCHIVE.replace("测试云纹壶", "测试另一器物"))
    with pytest.raises(DocumentError, match="同馆同编号"):
        catalog.save(second)
    assert catalog.list()["total"] == 1


def test_vectors_are_persistent_isolated_and_model_version_checked(catalog):
    request, _ = source_request(catalog, build_index=True)
    record = catalog.save(request)
    assert catalog.search("礼仪")["candidates"][0]["artifact_id"] == record["artifact_id"]
    assert catalog.index.vectors.count_document(record["artifact_id"]) == 0
    catalog.close()
    reopened = ArtifactService(catalog.index)
    try:
        assert reopened.get(record["artifact_id"])["name"] == "测试云纹壶"
        assert reopened.search("测试云纹壶")["candidates"][0]["artifact_id"] == record["artifact_id"]
        catalog.index.embedder.version = "changed-model"
        assert reopened.search("测试云纹壶")["warnings"]
        result = reopened.ask("无匹配名称的物品是什么年代？")
        assert result["status"] != "answered"
    finally:
        reopened.close()


def test_unknown_entity_does_not_answer_nearest_vector_match(catalog):
    request, _ = source_request(catalog, build_index=True)
    catalog.save(request)
    answer = catalog.ask("不存在的器物是什么材料？")
    assert answer["status"] == "needs_clarification"
    assert "facts" not in answer
    assert answer["sources"] == []


def test_delete_artifact_removes_vectors_but_retains_document(catalog):
    request, doc = source_request(catalog, build_index=True)
    record = catalog.save(request)
    catalog.delete(record["artifact_id"])
    assert catalog.vectors.count_document(record["artifact_id"]) == 0
    assert catalog.list()["total"] == 0
    assert catalog.index.get(doc["document_id"])
    with pytest.raises(DocumentError):
        catalog.get(record["artifact_id"])


def test_http_upload_draft_import_ask_and_invalid_requests(isolated_root):
    app = create_app(DocumentSettings(storage_dir=isolated_root / "docs"),
                     VectorSettings(chroma_dir=isolated_root / "vectors", registry_path=isolated_root / "docs.sqlite3", auto_index=False),
                     embedder=FakeEmbeddings(), chat_client=FakeChat(), rag_settings=RagSettings(),
                     runtime_settings=RuntimeSettings(warmup_on_startup=False))
    with TestClient(app) as client:
        uploaded = client.post("/documents", files={"file": ("测试档案.txt", ARCHIVE.encode("utf-8"), "text/plain")}).json()["data"]
        draft = client.post("/artifacts/draft", json={"document_id": uploaded["document_id"], "source_type": "official_archive"})
        assert draft.status_code == 200, draft.text
        records = draft.json()["data"]["records"]
        assert len(records) == 1
        imported = client.post("/artifacts/import", json={"records": records})
        assert imported.status_code == 200, imported.text
        result = imported.json()["data"]
        assert result["saved"] == 1 and result["failed"] == 0
        artifact = result["results"][0]["artifact"]
        assert artifact["index_status"] == "indexed"
        answer = client.post("/artifacts/ask", json={"question": "测试云纹壶保存在哪一个博物馆中？"})
        assert answer.status_code == 200, answer.text
        assert answer.json()["data"]["answer_policy"] == "documented_facts_only"
        assert answer.json()["data"]["facts"]["museum"]["value"] == "测试甲博物馆"
        assert client.get("/artifacts?museum=测试乙博物馆").json()["data"]["count"] == 0
        for endpoint, body in (("/artifacts/ask", {"question": "   "}), ("/artifacts/search", {"query": "  "}),
                               ("/artifacts", {"claims": []}), ("/artifacts/import", {"records": []})):
            assert client.post(endpoint, json=body).status_code == 422
        assert client.get("/artifacts/absent").status_code == 404
        assert "馆藏编号" in client.get("/artifacts/template").json()["data"]["fields"].values()
        assert client.delete(f"/artifacts/{artifact['artifact_id']}").json()["data"]["remaining_vectors"] == 0


def test_batch_invalid_provenance_prevents_any_write(catalog):
    from local_rag.artifacts.models import ArtifactImport
    good, _ = source_request(catalog)
    bad = good.model_copy(deep=True)
    bad.claims[0].quote = "不存在的摘录"
    with pytest.raises(DocumentError):
        catalog.import_records(ArtifactImport(records=[good, bad]))
    assert catalog.list()["total"] == 0


def test_whitespace_normalization_preserves_original_quote_and_offsets(catalog):
    request, doc = source_request(catalog)
    request.claims[0].quote = "名称： 测试 云纹壶"
    record = catalog.save(request)
    claim = record["facts"]["name"]["claims"][0]
    original = (catalog.index.documents.directory(doc["document_id"]) / "text.txt").read_text(encoding="utf-8")
    assert original[claim["start_index"]:claim["end_index"]] == claim["quote"] == "名称：测试云纹壶"
