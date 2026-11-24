import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_rag.api.errors import register_error_handlers
from local_rag.api.routes.ask import create_router
from local_rag.artifacts.presentation import public_artifact_text, PublicArtifactEvents
from local_rag.core.config import RagSettings
from local_rag.rag.service import RagService
from tests.pytest.test_artifacts import catalog, source_request
from tests.pytest.test_streaming import StreamChat
from tests.unit.test_rag import FakeChat

IDENTIFIER = 'WW-FCDEB84CC7464E1F83EEE7A8295C900E'
SOURCES = [{'source_id': 'S1', 'text': '名称：金蝉玉叶饰件', 'metadata': {
    'artifact_id': IDENTIFIER, 'claims': [{'field': 'name', 'value': '金蝉玉叶饰件'}]}}]


def test_prompt_uses_name_for_resolved_id_and_legacy_history_without_mutation():
    history = [{'question': '介绍这件文物', 'answer': IDENTIFIER + '是金蝉玉叶饰件。'}]
    prompt = json.loads(RagService._prompt('它是什么材料？', SOURCES, history=history,
                                          resolved_question=IDENTIFIER + '：它是什么材料？'))
    assert prompt['question'] == '金蝉玉叶饰件：它是什么材料？'
    assert IDENTIFIER not in json.dumps(prompt)
    assert history[0]['answer'].startswith(IDENTIFIER)
    assert SOURCES[0]['metadata']['artifact_id'] == IDENTIFIER


def test_prompt_preserves_question_only_history_and_omission_marker():
    from local_rag.rag.conversation import bounded_history
    history = [{'question': IDENTIFIER + '是什么？', 'answer': '很长的介绍。' * 200}]
    bounded = bounded_history(history, 200)
    assert bounded == [{'question': history[0]['question'], 'answer_omitted': True}]
    prompt = json.loads(RagService._prompt('它是什么材料？', SOURCES, history=bounded))
    assert prompt['conversation'] == [{'question': '金蝉玉叶饰件是什么？', 'answer_omitted': True}]
    assert bounded[0]['question'].startswith(IDENTIFIER)


@pytest.mark.parametrize('path', ['/chat', '/chat/stream'])
def test_long_answer_history_can_be_omitted_in_json_and_streaming_followups(catalog, path):
    request, _ = source_request(catalog)
    record = catalog.save(request)
    reply = {'answerable': True, 'answer': record['artifact_id'] + '的材料是青铜。'}
    client = StreamChat([json.dumps(reply, ensure_ascii=False)])
    client.reply = reply
    rag = RagService(catalog.index, RagSettings(), client)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(create_router(rag))
    with TestClient(app) as http:
        response = http.post(path, json={'question': '它是什么材料？', 'history': [
            {'question': '介绍测试云纹壶', 'answer': '上一轮很长的介绍。' * 200}]})
    assert response.status_code == 200
    if path.endswith('stream'):
        assert 'event: error' not in response.text
        frames = response.text.replace('\r\n', '\n').split('\n\n')
        result_frame = next(f for f in frames if f.startswith('event: result\n'))
        result = json.loads(next(line[6:] for line in result_frame.splitlines() if line.startswith('data: ')))['data']
    else:
        result = response.json()['data']
    assert result['answer'] == '测试云纹壶的材料是青铜。'
    assert result['conversation']['truncated']
    prompt = json.loads(client.calls[-1]['prompt'])
    assert prompt['conversation'] == [{'question': '介绍测试云纹壶', 'answer_omitted': True}]


@pytest.mark.parametrize('split', range(1, len(IDENTIFIER)))
def test_stream_hides_id_at_every_fragment_boundary_and_preserves_real_numbers(split):
    events = []
    stream = PublicArtifactEvents(lambda *e: events.append(e), SOURCES)
    stream('delta', {'text': '藏品：' + IDENTIFIER[:split]})
    assert ''.join(data['text'] for _, data in events) == '藏品：'
    stream('delta', {'text': IDENTIFIER[split:] + '，馆藏号 TEST-001 / WW-1234。'})
    stream.finish()
    assert ''.join(data['text'] for _, data in events) == '藏品：金蝉玉叶饰件，馆藏号 TEST-001 / WW-1234。'


def test_stream_reset_drops_pending_id_and_flush_keeps_short_real_number():
    events = []
    stream = PublicArtifactEvents(lambda *e: events.append(e))
    stream('delta', {'text': 'WW-ABCDE'})
    stream('reset', {'reason': 'generation_retry'})
    stream('delta', {'text': '馆藏号 WW-1234'})
    stream.finish()
    assert events == [('reset', {'reason': 'generation_retry'}),
                      ('delta', {'text': '馆藏号 '}), ('delta', {'text': 'WW-1234'})]
    assert public_artifact_text(IDENTIFIER + '是一件文物。') == '该文物是一件文物。'
    assert public_artifact_text(IDENTIFIER.lower(), SOURCES) == '金蝉玉叶饰件'
    assert public_artifact_text(IDENTIFIER, SOURCES, language='en') == 'the artifact'


@pytest.mark.parametrize('streaming', [False, True])
def test_chat_preserves_identity_for_retrieval_but_never_displays_it(catalog, streaming):
    request, _ = source_request(catalog)
    record = catalog.save(request)
    identifier = record['artifact_id']
    raw = {'answerable': True, 'answer': f'{identifier}的材料是青铜，正式馆藏编号是 TEST-001。'}
    client = StreamChat([json.dumps(raw, ensure_ascii=False)]) if streaming else FakeChat(raw)
    rag = RagService(catalog.index, RagSettings(), client)
    events = []
    result = rag.ask('它是什么材料？', top_k=None,
                     history=[{'question': '介绍测试云纹壶', 'answer': identifier + '是一件文物。'}],
                     on_event=(lambda *e: events.append(e)) if streaming else None)
    assert result['sources'][0]['metadata']['artifact_id'] == identifier
    assert identifier in result['conversation']['resolved_question']
    assert identifier not in client.calls[-1]['prompt']
    assert result['answer'] == '测试云纹壶的材料是青铜，正式馆藏编号是 TEST-001。'
    if streaming:
        assert ''.join(data['text'] for name, data in events if name == 'delta') == result['answer']
