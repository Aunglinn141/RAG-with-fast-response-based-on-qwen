"""Existing PDF sources retain provenance, source rank, page context and privacy."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from local_rag.artifacts.service import ArtifactService
from local_rag.documents.models import ReadDocument, Extraction, TextUnit
from local_rag.documents.service import DocumentStore
from local_rag.evaluation import museum_demo as demo, museum_library as library
from local_rag.indexing.service import IndexService
from tests.fixtures.embeddings import FakeEmbeddings
from tests.fixtures.reader_documents import unicode_pdf_bytes


def pages(museum):
    result = ['仅供软件测试。不是馆方出版的原版图录。学习型资料 · 非馆方原版出版物']
    for page in range(4):
        text = museum + ' 馆藏文物资料汇编 / 重点文物\n'
        for n in range(page * 2 + 1, page * 2 + 3):
            text += (f'NO. {n:03} 馆方资料概述\n测试器物{n}\n测试时期 · 精品详解 · 铜、玉\n'
                     '公开尺寸/形制记载：未公开尺寸\n文物介绍\n测试器物呈圆形。\n'
                     '阅读观察 可比较图案设计。\n查看原始资料 ↗\n')
        result.append(text + '学习型资料 · 非馆方原版出版物')
    return result


def reading(museum):
    return ReadDocument(str(uuid4()), 'sample.pdf', 'pdf', 1,
                        Extraction([TextUnit(text, page_number=n) for n, text in enumerate(pages(museum), 1)], 5))


def test_featured_parser_uses_explicit_page_context_not_sequence_or_research_prompts():
    result = library.featured_claims(reading('测试馆'), 'sample.pdf', '测试馆')
    assert len(result) == 8
    assert {c['field'] for r in result for c in r['claims']} == {'name', 'era', 'material', 'museum', 'description'}
    assert all('可比较图案' not in c['value'] for r in result for c in r['claims'])
    assert result[0]['source_page'] == 2
    assert result[0]['editorial_sequence'] == '001'
    assert any('未公开尺寸' in c['value'] for c in result[0]['claims'] if c['field'] == 'description')
    assert all('测试馆' in c['quote'] and '测试器物' in c['quote'] for r in result for c in r['claims'] if c['field'] == 'museum')


@pytest.mark.parametrize('damage', ['museum', 'rank', 'count', 'layout'])
def test_featured_parser_fails_closed_on_ambiguous_layout(damage):
    doc = reading('测试馆')
    if damage == 'museum':
        doc.extraction.units[1].text = doc.extraction.units[1].text.replace('测试馆', '别的馆')
    elif damage == 'rank':
        doc.extraction.units[0].text = '正式档案'
    elif damage == 'count':
        doc.extraction.units.pop()
    else:
        doc.extraction.units[1].text = doc.extraction.units[1].text.replace(' · 精品详解 · ', '不明确的分隔')
    with pytest.raises(ValueError):
        library.featured_claims(doc, 'sample.pdf', '测试馆')


@pytest.fixture
def pdf_library(isolated_root, monkeypatch):
    monkeypatch.setattr(library, 'RUNTIME_DIR', isolated_root / 'runtime')
    monkeypatch.setattr(demo, 'DEMOS_ROOT', isolated_root / 'demos')
    source = isolated_root / 'library'
    for filename, museum in library.COLLECTIONS.items():
        identifier = str(uuid4())
        folder = source / identifier
        folder.mkdir(parents=True)
        raw = unicode_pdf_bytes(pages(museum))
        (folder / 'original.pdf').write_bytes(raw)
        demo.write_json(folder / 'document.json', {'filename': filename, 'document_id': identifier, 'sha256': demo.digest(raw)})
    return source


def test_library_snapshot_and_prepare_keep_pdf_pages_and_secondary_claims(pdf_library, monkeypatch, isolated_root):
    before = {str(p): demo.digest(p.read_bytes()) for p in pdf_library.rglob('*') if p.is_file()}
    path = library.build_dataset(pdf_library)
    frozen = demo.load_dataset(path)
    assert frozen['data']['synthetic'] is False
    assert len(frozen['data']['artifacts']) == 32
    assert len(frozen['documents']) == 4
    assert all(s['source_type'] == 'secondary' and s['original_document_id'] for s in frozen['data']['sources'])
    def open_index(doc, vec):
        index = IndexService(DocumentStore(doc), vec, FakeEmbeddings())
        index.artifacts = ArtifactService(index)
        return index
    monkeypatch.setattr(demo, '_open_index', open_index)
    monkeypatch.setattr(demo, '_models', lambda i, r: {'embedding': asdict(i.embedder.identity()), 'chat': {'name': r.model_name, 'version': 'fake'}})
    manifest_path = demo.prepare(isolated_root / 'demos/pdf', path, progress=lambda _: None)
    root, manifest, _, doc, vec, _ = demo.load_prepared(manifest_path)
    assert manifest['synthetic'] is False
    index = open_index(doc, vec)
    try:
        record = index.artifacts.get(next(iter(manifest['artifacts'].values()))['artifact_id'])
        assert record['facts']['material']['status'] == 'secondary_only'
        assert record['facts']['name']['claims'][0]['page_numbers'] == [2]
        assert record['facts']['accession_number']['status'] == 'missing'
        assert len(index.list_documents()) == 4
    finally:
        index.close()
    assert before == {str(p): demo.digest(p.read_bytes()) for p in pdf_library.rglob('*') if p.is_file()}


def test_pdf_hash_or_source_upgrade_is_rejected(pdf_library):
    path = library.build_dataset(pdf_library)
    original = demo.read_json(path)
    for alteration in ('hash', 'rank'):
        data = deepcopy(original)
        if alteration == 'hash':
            data['sources'][0]['sha256'] = 'wrong'
        else:
            data['sources'][0]['source_type'] = 'official_archive'
        demo.write_json(path, data)
        with pytest.raises(ValueError, match='hash and secondary'):
            demo.load_dataset(path)


def test_changed_original_is_rejected_without_snapshot(pdf_library):
    original = next(pdf_library.glob('*/original.pdf'))
    original.write_bytes(original.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        library.build_dataset(pdf_library)


def test_duplicate_source_is_rejected_before_snapshot(pdf_library):
    import shutil
    original = next(pdf_library.iterdir())
    shutil.copytree(original, pdf_library / str(uuid4()))
    with pytest.raises(ValueError, match='Multiple library documents'):
        library.build_dataset(pdf_library)
