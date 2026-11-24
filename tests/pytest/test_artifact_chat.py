"""The regular chat page's JSON and SSE routes generate artifact answers with an LLM."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_rag.api.errors import register_error_handlers
from local_rag.api.routes.ask import create_router
from local_rag.artifacts.retrieval import catalog_contexts
from local_rag.core.config import RagSettings
from local_rag.core.exceptions import DocumentError
from local_rag.rag.service import RagService
from tests.pytest.test_artifacts import ARCHIVE, catalog, source_request
from tests.pytest.test_streaming import StreamChat
from tests.unit.test_rag import FakeChat


def make_rag(catalog, answer="按档案记载，测试云纹壶由测试甲博物馆收藏。"):
    request, doc = source_request(catalog)
    record = catalog.save(request)
    chat = FakeChat({"answerable": True, "answer": answer})
    return RagService(catalog.index, RagSettings(), chat), record, doc


def test_normal_chat_generates_from_catalog_without_calling_template_answer(catalog, monkeypatch):
    rag, record, doc = make_rag(catalog)
    def forbidden(*args, **kwargs):
        raise AssertionError("The chat page must not use the deterministic artifact answer")
    monkeypatch.setattr(catalog, "ask", forbidden)
    result = rag.ask("测试云纹壶保存在哪一个博物馆中？", top_k=None)
    assert result["answer"] == rag.client.reply["answer"]
    assert len(rag.client.calls) == 1
    assert len(rag.client.scope_calls) == 1
    request = rag.client.calls[0]
    assert "测试甲博物馆" in request["prompt"]
    assert "出土地" in request["system_prompt"]
    source = result["sources"][0]
    assert source["metadata"]["artifact_id"] == record["artifact_id"]
    assert source["metadata"]["claims"][0]["document_id"] == doc["document_id"]
    assert result["retrieval"]["context_types"]["artifact"] == 1
    assert not source["truncated"]


def test_chat_sse_streams_the_selected_model_answer(catalog):
    rag, record, _ = make_rag(catalog)
    response = {"answerable": True, "answer": "这件文物的材料为青铜。"}
    client = StreamChat([json.dumps(response, ensure_ascii=False)])
    rag.client = client
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(rag))
    with TestClient(app) as http:
        result = http.post("/chat/stream", json={"question": f"{record['artifact_id']}是什么材料？", "model": "gemma3:1b"})
        assert result.status_code == 200, result.text
        assert "event: delta" in result.text and "event: result" in result.text
        assert response["answer"] in result.text
        assert client.calls[0]["model"] == "gemma3:1b"
        assert "青铜" in client.calls[0]["prompt"]


def test_http_chat_uses_same_catalog_and_generation(catalog):
    rag, _, _ = make_rag(catalog)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(rag))
    with TestClient(app) as http:
        reply = http.post("/chat", json={"question": "测试云壶的收藏博物馆？"})
        assert reply.status_code == 200, reply.text
        assert reply.json()["data"]["answer"] == rag.client.reply["answer"]
        assert rag.client.calls


def test_same_name_ambiguity_reaches_llm_as_clarification_context(catalog):
    rag, _, _ = make_rag(catalog, "资料中有同名文物，请说明你想了解哪家博物馆的藏品。")
    request, _ = source_request(catalog, ARCHIVE.replace("测试甲博物馆", "测试乙博物馆"))
    catalog.save(request)
    answer = rag.ask("测试云纹壶在哪个博物馆？", top_k=None)
    assert answer["answer"] == rag.client.reply["answer"]
    prompt = rag.client.calls[0]["prompt"]
    assert "同名或别名对应多件文物" in prompt
    assert "不要要求用户打开文物档案页面" in rag.client.calls[0]["system_prompt"]
    for source in answer["sources"]:
        if source["metadata"]["context_type"] == "artifact":
            assert source["metadata"]["matched_records"] == 2
            assert not source["truncated"]


def test_conflict_and_secondary_qualification_are_in_model_context(catalog):
    request, _ = source_request(catalog, source_type="secondary")
    other, _ = source_request(catalog, ARCHIVE.replace("材料：青铜", "材料：铜"), "museum_guide")
    request.claims += [c for c in other.claims if c.field == "material"]
    catalog.save(request)
    rag = RagService(catalog.index, RagSettings(), FakeChat({"answerable": True, "answer": "材料记载有冲突，需要核对。"}))
    result = rag.ask("测试云纹壶的材料和收藏馆？")
    assert result["status"] == "answered"
    prompt = rag.client.calls[0]["prompt"]
    assert "来源冲突，不可择一认定" in prompt and "未经馆方原档案核验" in prompt
    assert '青铜' in prompt and '铜' in prompt


def test_museum_named_in_conversation_disambiguates_without_a_picker(catalog):
    make_rag(catalog)
    request, _ = source_request(catalog, ARCHIVE.replace("测试甲博物馆", "测试乙博物馆"))
    second = catalog.save(request)
    contexts = catalog_contexts(catalog, "测试乙博物馆的测试云纹壶是什么年代？")
    assert len(contexts) == 1
    assert contexts[0]["metadata"]["artifact_id"] == second["artifact_id"]
    assert "测试甲博物馆" not in contexts[0]["text"]


def test_source_scope_and_deletion_do_not_leak_catalog_claims(catalog):
    rag, _, doc = make_rag(catalog)
    other, other_doc = source_request(catalog, ARCHIVE.replace("测试甲博物馆", "测试乙博物馆"))
    record = catalog.registry.rows()[0]
    # Put a museum claim from a second document on the same item.
    request, _ = source_request(catalog)
    request.claims = [c for c in request.claims if c.field != "museum"] + [c for c in other.claims if c.field == "museum"]
    request.expected_revision = record["revision"]
    catalog.save(request, record["artifact_id"])
    contexts = catalog_contexts(catalog, "测试云纹壶在哪个博物馆？", document_id=str(request.claims[0].document_id))
    assert contexts and "测试乙博物馆" not in contexts[0]["text"]
    assert all(c["document_id"] == str(request.claims[0].document_id) for c in contexts[0]["metadata"]["claims"])
    catalog.index.delete(str(request.claims[0].document_id))
    assert catalog_contexts(catalog, "测试云纹壶在哪个博物馆？") == []


def test_unavailable_vector_search_can_still_ground_llm_in_live_facts(catalog, monkeypatch):
    rag, _, _ = make_rag(catalog)
    def offline(*args, **kwargs):
        raise DocumentError("embedding_unavailable", "offline", 503)
    monkeypatch.setattr(catalog.index, "search", offline)
    result = rag.ask("测试云纹壶在哪个博物馆？")
    assert result["status"] == "answered" and rag.client.calls
    assert result["warnings"]


def test_structured_conflict_cannot_be_truncated_into_one_sided_answer(catalog):
    rag, _, _ = make_rag(catalog)
    contexts = catalog_contexts(catalog, "测试云纹壶在哪个博物馆？")
    rag.settings.max_context_characters = 100
    assert rag._sources("测试云纹壶在哪个博物馆？", contexts) == []


def test_unrelated_query_keeps_catalog_out_of_general_chat(catalog):
    make_rag(catalog)
    assert catalog_contexts(catalog, "如何使用这个图书馆？") == []


def test_description_reaches_model_without_neighbor_dimensions_or_internal_id(catalog, monkeypatch):
    text = ARCHIVE + '文物介绍：本器尺寸未公开，不能采用邻近器物的尺寸。\n'
    request, doc = source_request(catalog, text)
    from local_rag.artifacts.models import ClaimInput
    request.claims.append(ClaimInput(**{'field': 'description', 'value': '本器尺寸未公开，不能采用邻近器物的尺寸。',
                           'quote': '文物介绍：本器尺寸未公开，不能采用邻近器物的尺寸。',
                           'document_id': doc['document_id'], 'source_type': 'secondary'}))
    record = catalog.save(request)
    rag = RagService(catalog.index, RagSettings(), FakeChat({'answerable': True, 'answer': '资料没有该器的尺寸。'}))
    def unrelated(*args, **kwargs):
        source = {'chunk_id': 'neighbor', 'text': '邻近器物尺寸：长17.5厘米。',
                  'metadata': {'document_id': doc['document_id'], 'filename': doc['filename']}, 'score': 1.0}
        return {'matches': [source], 'contexts': [source], 'model': None, 'warnings': []}
    monkeypatch.setattr(catalog.index, 'retrieve_context', unrelated)
    answer = rag.ask('测试云纹壶尺寸是多少？', top_k=None)
    assert len(answer['sources']) == 1
    prompt = rag.client.calls[0]['prompt']
    assert '尺寸未公开' in prompt and '17.5' not in prompt
    assert record['artifact_id'] not in prompt
    assert answer['sources'][0]['metadata']['artifact_id'] == record['artifact_id']


def test_catalog_budget_cannot_drop_second_object_and_answer_first_only(catalog):
    rag, _, _ = make_rag(catalog)
    other, _ = source_request(catalog, ARCHIVE.replace('测试甲博物馆', '测试乙博物馆'))
    catalog.save(other)
    contexts = catalog_contexts(catalog, '测试云纹壶是什么材料？')
    assert len(contexts) == 2
    rag.settings.max_context_characters = len(contexts[0]['text']) + 1
    assert rag._sources('测试云纹壶是什么材料？', contexts) == []


def test_publication_question_is_not_replaced_with_catalog_candidates(catalog, monkeypatch):
    make_rag(catalog)
    def unrelated(*args, **kwargs):
        raise AssertionError('A publication question must use the document reading notes')
    monkeypatch.setattr(catalog, 'search', unrelated)
    assert catalog_contexts(catalog, '上传的博物馆资料是馆方原版图录吗？') == []


def test_unidentified_bronze_requests_identity_without_candidate_facts(catalog, monkeypatch):
    rag, record, doc = make_rag(catalog)
    monkeypatch.setattr(catalog, 'search', lambda *a, **kw: {'candidates': [{'artifact_id': record['artifact_id']}]})
    contexts = catalog_contexts(catalog, '这件青铜器的用途是什么？')
    assert len(contexts) == 1
    assert '身份未确定' in contexts[0]['text']
    assert '测试甲博物馆' in contexts[0]['text']
    assert '用于测试礼仪场景' not in contexts[0]['text']
    assert all(c['field'] in {'name', 'museum'} for c in contexts[0]['metadata']['claims'])
    def misleading_excerpt(*args, **kwargs):
        source = {'chunk_id': 'unrelated', 'text': '别的器物用于测试礼仪场景。',
                  'metadata': {'document_id': doc['document_id'], 'filename': doc['filename']}, 'score': 1.0}
        return {'matches': [source], 'contexts': [source], 'model': None, 'warnings': []}
    monkeypatch.setattr(catalog.index, 'retrieve_context', misleading_excerpt)
    result = rag.ask('它是什么材料？', top_k=None)
    assert result['sources'] and all(s['metadata']['context_type'] == 'artifact' for s in result['sources'])
    assert '用于测试礼仪场景' not in rag.client.calls[0]['prompt']


def test_editorial_heading_is_evidence_but_never_an_accession_claim(catalog):
    heading = 'NO. 007 馆方资料概述\n测试云纹壶'
    request, _ = source_request(catalog, heading + '\n' + ARCHIVE.replace('馆藏编号：TEST-001\n', ''), 'secondary')
    next(c for c in request.claims if c.field == 'name').quote = heading
    catalog.save(request)
    contexts = catalog_contexts(catalog, '测试云纹壶的 NO. 007 是馆藏编号吗？')
    text = json.loads(contexts[0]['text'])
    assert heading in text['原文条目标题']
    assert text['字段']['馆藏编号']['statements'] == []


def test_two_full_descriptions_and_cultures_fit_normal_comparison_budget(catalog):
    from local_rag.artifacts.models import ClaimInput
    for old, new in [('测试云纹壶', '测试红陶壶'), ('测试云纹壶', '测试白陶壶')]:
        description = '形制与工艺：器身圆腹，表面有彩绘纹样，底部平坦。' * 3
        text = ARCHIVE.replace(old, new).replace('馆藏编号：TEST-001\n', '') + '文物介绍：' + description + '\n'
        request, doc = source_request(catalog, text, 'secondary')
        request.claims.append(ClaimInput(field='description', value=description, quote='文物介绍：' + description,
                                        document_id=doc['document_id'], source_type='secondary'))
        catalog.save(request)
    rag = RagService(catalog.index, RagSettings(), FakeChat({'answerable': True, 'answer': '据汇编，两件均有彩绘。'}))
    result = rag.ask('测试红陶壶和测试白陶壶的形制、文化和材料有何区别？', top_k=None)
    assert len(result['sources']) == 2
    assert all(s['metadata']['has_description'] and not s['truncated'] for s in result['sources'])
    assert all('形制与工艺' in s['text'] for s in result['sources'])
    assert len(rag.client.calls) == 1


def test_work_title_without_artist_matches_but_shared_title_stays_ambiguous(catalog):
    first, _ = source_request(catalog, ARCHIVE.replace('测试云纹壶', '画家甲《测试山水》卷'))
    catalog.save(first)
    contexts = catalog_contexts(catalog, '《测试山水》是什么年代？')
    assert len(contexts) == 1
    assert '测试时期' in contexts[0]['text']
    second, _ = source_request(catalog, ARCHIVE.replace('测试云纹壶', '画家乙《测试山水》轴').replace('测试甲博物馆', '测试乙博物馆'))
    catalog.save(second)
    contexts = catalog_contexts(catalog, '介绍《测试山水》。')
    assert len(contexts) == 2
    assert all('同名或别名对应多件文物' in s['text'] for s in contexts)


def test_static_collection_cannot_answer_current_gallery_even_if_model_would(catalog):
    rag, _, _ = make_rag(catalog, '今天在第2号展厅展出。')
    contexts = catalog_contexts(catalog, '测试云纹壶目前在哪个展厅展出？', exact_only=True)
    sources = rag._sources('测试云纹壶目前在哪个展厅展出？', contexts)
    answer, version = rag._generate('测试云纹壶目前在哪个展厅展出？', sources, rag.settings.model_name)
    assert answer is None and version
    assert not rag.client.calls and not rag.client.scope_calls
    # A source about a past exhibition is a normal document question, not live data.
    from local_rag.rag.scope import requires_live_exhibition_evidence
    assert not requires_live_exhibition_evidence('测试云纹壶在2010年的展览中怎样陈列？')
