"""Demo provenance, isolation, reproducibility and honest scoring contracts."""

from copy import deepcopy
from dataclasses import asdict
import json
import shutil

import pytest

from local_rag.artifacts.service import ArtifactService
from local_rag.documents.service import DocumentStore
from local_rag.evaluation import museum_demo as demo
from local_rag.indexing.service import IndexService
from tests.fixtures.embeddings import FakeEmbeddings


@pytest.fixture
def sandbox(isolated_root, monkeypatch):
    monkeypatch.setattr(demo, 'DEMOS_ROOT', isolated_root / 'demos')
    def open_index(doc, vec):
        index = IndexService(DocumentStore(doc), vec, FakeEmbeddings())
        index.artifacts = ArtifactService(index)
        return index
    monkeypatch.setattr(demo, '_open_index', open_index)
    monkeypatch.setattr(demo, '_models', lambda index, settings: {
        'embedding': asdict(index.embedder.identity()), 'chat': {'name': settings.model_name, 'version': 'fake-chat-v1'}})
    return isolated_root


@pytest.fixture
def prepared(sandbox):
    return demo.prepare(sandbox / 'demos/collection', progress=lambda _: None)


def copy_dataset(root):
    target = root / 'dataset'
    shutil.copytree(demo.DATASET.parent, target)
    return target / 'dataset.json'


def test_corpus_is_complete_and_every_claim_has_source_evidence():
    frozen = demo.load_dataset()
    assert len(frozen['documents']) == 14
    assert len(frozen['data']['artifacts']) == 12
    assert len(frozen['data']['cases']) == 30
    assert sum(c['critical'] for c in frozen['data']['cases']) == 9


@pytest.mark.parametrize('corruption', ['quote', 'path', 'gate', 'audience', 'disclaimer'])
def test_validation_rejects_corrupt_provenance_or_scenarios(isolated_root, corruption):
    path = copy_dataset(isolated_root)
    data = demo.read_json(path)
    if corruption == 'quote':
        data['artifacts'][0]['claims'][0]['quote'] = 'invented evidence'
    elif corruption == 'path':
        data['sources'][0]['file'] = '../outside.txt'
    elif corruption == 'gate':
        data['cases'][5]['critical'] = False
    elif corruption == 'audience':
        data['cases'][0]['audience'] = 'staff'
    else:
        (path.parent / data['sources'][0]['file']).write_text('unmarked fiction', encoding='utf-8')
    demo.write_json(path, data)
    with pytest.raises(ValueError):
        demo.load_dataset(path)


def test_prepare_isolates_storage_and_preserves_source_states(prepared, sandbox, monkeypatch):
    # Even hostile default env paths cannot redirect the manifest's configured stores.
    monkeypatch.setenv('DOCUMENT_STORAGE_DIR', str(sandbox / 'real-library'))
    root, manifest, frozen, doc, vec, rag = demo.load_prepared(prepared)
    assert doc.storage_dir == root / 'documents'
    assert vec.registry_path == root / 'registry/documents.sqlite3'
    assert not (sandbox / 'real-library').exists()
    assert manifest['status'] == 'ready'
    index = demo._open_index(doc, vec)
    try:
        assert len(index.list_documents()) == 14
        assert index.artifacts.list()['total'] == 12
        fish = index.artifacts.get(manifest['artifacts']['fish_jade']['artifact_id'])
        assert fish['facts']['era']['status'] == 'conflict'
        cup = index.artifacts.get(manifest['artifacts']['white_cup']['artifact_id'])
        assert cup['facts']['era']['status'] == 'missing'
        leaf = index.artifacts.get(manifest['artifacts']['leaf_jade']['artifact_id'])
        assert leaf['facts']['material']['status'] == 'secondary_only'
        assert all(d['status'] == 'indexed' for d in index.list_documents())
    finally:
        index.close()
    before = prepared.read_bytes()
    with pytest.raises(ValueError, match='populated'):
        demo.prepare(root)
    assert prepared.read_bytes() == before


def test_preparation_rejects_outside_path_before_mutation(sandbox):
    target = sandbox / 'production'
    with pytest.raises(ValueError, match='child'):
        demo.prepare(target)
    assert not target.exists()


def test_failed_preparation_retains_diagnostics(sandbox, monkeypatch):
    def broken(*_):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(demo, '_models', broken)
    target = sandbox / 'demos/failed'
    with pytest.raises(RuntimeError, match='model unavailable'):
        demo.prepare(target)
    manifest = demo.read_json(target / 'manifest.json')
    assert manifest['status'] == 'failed'
    assert manifest['error']['message'] == 'model unavailable'
    assert (target / 'dataset/dataset.json').is_file()
    with pytest.raises(ValueError, match='ready'):
        demo.load_prepared(target / 'manifest.json')


def test_frozen_dataset_or_storage_redirect_is_rejected(prepared, sandbox):
    manifest = demo.read_json(prepared)
    altered = deepcopy(manifest)
    altered['configuration']['vectors']['registry_path'] = str(sandbox / 'production.sqlite3')
    demo.write_json(prepared, altered)
    with pytest.raises(ValueError, match='paths'):
        demo.load_prepared(prepared)
    assert not (sandbox / 'production.sqlite3').exists()
    demo.write_json(prepared, manifest)
    corpus = prepared.parent / 'dataset/corpus/cloud_bronze.txt'
    corpus.write_text(corpus.read_text(encoding='utf-8') + '新增资料\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen corpus'):
        demo.load_prepared(prepared)


class RecordingRag:
    def __init__(self):
        self.calls = []

    def ask(self, question, *, top_k, model, history):
        self.calls.append({'question': question, 'top_k': top_k, 'history': deepcopy(history)})
        if question == '演示星纹镜是什么材料？':
            raise RuntimeError('deliberate failure')
        return {'answer': '=actual answer ' + str(len(self.calls)), 'status': 'answered',
                'sources': [{'text': 'actual retrieved evidence'}], 'model': {'version': 'fake-chat-v1'}}


def test_evaluation_uses_actual_history_resets_and_keeps_errors(prepared, monkeypatch):
    rag = RecordingRag()
    monkeypatch.setattr(demo, '_open_rag', lambda *_: rag)
    path = demo.evaluate(prepared, progress=lambda _: None)
    report = demo.read_json(path)
    assert len(report['results']) == 30
    assert report['summary']['execution_errors'] == 2
    assert report['summary']['acceptance'] == 'pending_review'
    assert all(c['top_k'] is None for c in rag.calls)
    by_id = {r['id']: r for r in report['results']}
    assert by_id['v05']['turns'][1]['history_sent'] == []  # failed first turn
    follow = by_id['s08']['turns']
    assert follow[1]['history_sent'] == [{'question': follow[0]['gold']['question'],
                                        'answer': follow[0]['response']['answer']}]
    assert by_id['v10']['turns'][1]['history_sent'] == []
    assert by_id['r09']['turns'][1]['history_sent'] == []
    assert by_id['v01']['turns'][0]['response']['sources']
    assert "'=actual answer" in path.with_suffix('.csv').read_text(encoding='utf-8-sig')
    with pytest.raises(ValueError, match='new .json'):
        demo.evaluate(prepared, path)
    review = {'run_id': report['run_id'], 'dataset_sha256': report['dataset_sha256'],
              'scenarios': [{'id': r['id'], **dict.fromkeys(demo.REVIEW_FIELDS, True),
                             'reviewer': 'Test agent', 'reviewer_kind': 'agent', 'notes': 'Fixture judgment'}
                            for r in report['results']]}
    review['scenarios'][5]['groundedness'] = False
    review_path = path.with_name('review.json')
    demo.write_json(review_path, review)
    summary = demo.apply_review(path, review_path)
    assert summary['acceptance'] == 'not_met'  # critical failure even with 27 passes
    updated = demo.read_json(path)
    assert updated['results'][0]['turns'] == report['results'][0]['turns']
    assert updated['results'][0]['semantic_review']['reviewer_kind'] == 'agent'


def test_model_change_fails_before_questions_and_retains_report(prepared, monkeypatch):
    monkeypatch.setattr(demo, '_models', lambda *_: {'chat': {'version': 'replacement'}})
    monkeypatch.setattr(demo, '_open_rag', lambda *_: pytest.fail('Must not ask with a changed model'))
    path = prepared.parent / 'reports/drift.json'
    with pytest.raises(ValueError, match='model versions'):
        demo.evaluate(prepared, path)
    report = demo.read_json(path)
    assert report['status'] == 'failed'
    assert report['results'] == []
    assert report['summary']['acceptance'] == 'not_met'


def mock_report():
    cases = demo.load_dataset()['data']['cases']
    return {'status': 'completed', 'run_id': 'one', 'dataset_sha256': 'hash', 'model_versions_stable': True,
            'results': [{'id': c['id'], 'critical': c['critical'], 'turns': [{'error': None}],
                         'semantic_review': None} for c in cases]}


def test_acceptance_requires_semantic_reviews_and_every_critical_case():
    report = mock_report()
    demo.summarize(report)
    assert report['summary']['acceptance'] == 'pending_review'
    for result in report['results']:
        result['semantic_review'] = dict.fromkeys(demo.REVIEW_FIELDS, True)
    report['results'][0]['semantic_review']['completeness'] = False
    demo.summarize(report)
    assert report['summary']['semantic_passed'] == 29
    assert report['summary']['acceptance'] == 'met'
    next(r for r in report['results'] if r['critical'])['semantic_review']['groundedness'] = False
    demo.summarize(report)
    assert report['summary']['semantic_passed'] == 28
    assert report['summary']['acceptance'] == 'not_met'


def test_review_rejects_stale_run_duplicate_ids_and_ambiguous_reviewer(sandbox):
    report = mock_report()
    path = sandbox / 'demos/collection/reports/test.json'
    demo.write_json(path, report)
    review = {'run_id': 'old', 'dataset_sha256': 'hash', 'scenarios': []}
    review_path = sandbox / 'review.json'
    demo.write_json(review_path, review)
    with pytest.raises(ValueError, match='completed run'):
        demo.apply_review(path, review_path)
    review['run_id'] = 'one'
    review['scenarios'] = [{'id': r['id'], **dict.fromkeys(demo.REVIEW_FIELDS, True),
                            'reviewer': 'Test agent', 'reviewer_kind': 'agent', 'notes': 'Test judgment'}
                           for r in report['results']]
    review['scenarios'][-1]['id'] = review['scenarios'][0]['id']
    demo.write_json(review_path, review)
    with pytest.raises(ValueError, match='exactly once'):
        demo.apply_review(path, review_path)
    review['scenarios'][-1]['id'] = report['results'][-1]['id']
    review['scenarios'][0]['reviewer_kind'] = 'automatic_accuracy'
    demo.write_json(review_path, review)
    with pytest.raises(ValueError, match='reviewer'):
        demo.apply_review(path, review_path)


def test_validate_only_needs_no_models_or_storage(sandbox, monkeypatch, capsys):
    from local_rag.cli.museum_demo import main
    monkeypatch.setattr(demo, '_open_index', lambda *_: pytest.fail('Must not create storage'))
    assert main(['--validate-only']) == 0
    assert json.loads(capsys.readouterr().out)['valid'] is True
    assert not (sandbox / 'demos').exists()


def test_no_generation_version_is_not_mistaken_for_model_drift():
    before = {'chat': {'version': 'a'}}
    results = [{'turns': [{'response': {'model': {'version': None}}},
                          {'response': {'model': {'version': 'a'}}}]}]
    assert demo.model_versions_stable(before, before, results)
    results[0]['turns'][1]['response']['model']['version'] = 'b'
    assert not demo.model_versions_stable(before, before, results)
