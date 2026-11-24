"""Frozen museum collections, isolated preparation, and auditable chat evaluation."""

from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

from local_rag.core.config import DocumentSettings, RagSettings, VectorSettings, PROJECT_ROOT, RUNTIME_DIR

DATASET = PROJECT_ROOT / 'benchmarks/museum_demo_v1/dataset.json'
DEMOS_ROOT = RUNTIME_DIR / 'demos'
CRITICAL_TAGS = {'ambiguity', 'conflict', 'missing_evidence', 'isolation'}
REVIEW_FIELDS = ('correctness', 'completeness', 'groundedness', 'audience_suitability')
PROFILES = {'museum_demo_v1': (True, 12, 3), 'museum_documents_v1': (False, 32, 4)}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_dataset(path=DATASET):
    """Validate all evidence before creating storage or connecting to a model."""
    from local_rag.artifacts.models import ArtifactWrite

    path = Path(path).resolve()
    data = read_json(path)
    profile = PROFILES.get(data.get('benchmark_id'))
    if data.get('version') != 1 or not profile or data.get('synthetic') is not profile[0]:
        raise ValueError('Require a supported version 1 museum dataset with the correct source classification.')
    if len(data['artifacts']) != profile[1] or len(set(data['museums'])) != profile[2]:
        raise ValueError(f'Require {profile[1]} artifacts across {profile[2]} museums for this dataset.')
    documents, source_types, texts = {}, {}, {}
    hashes = {'dataset.json': digest(path.read_bytes())}
    for source in data['sources']:
        name = source['file']
        target = (path.parent / name).resolve()
        if (target.parent != (path.parent / 'corpus').resolve() or target.suffix not in {'.txt', '.pdf'}
                or Path(name).as_posix() != 'corpus/' + target.name or name in documents):
            raise ValueError(f'Invalid or duplicate corpus path: {name}')
        raw = target.read_bytes()
        if target.suffix == '.pdf':
            from local_rag.documents.reader import DocumentReader
            text = DocumentReader().read(target).text
        else:
            from local_rag.documents.extractors import clean_text
            text = clean_text(raw.decode('utf-8'))
        if data['synthetic'] and (not text.startswith(data['disclaimer']) or '虚构' not in data['disclaimer']):
            raise ValueError(f'Missing synthetic disclosure: {name}')
        if not data['synthetic'] and (source.get('sha256') != digest(raw)
                or source['source_type'] != 'secondary' or '非馆方原版出版物' not in text):
            raise ValueError(f'Existing compilation must retain its original hash and secondary-source status: {name}')
        if source['source_type'] not in {'official_archive', 'museum_guide', 'secondary'}:
            raise ValueError(f'Invalid source type: {name}')
        documents[name], source_types[name], hashes[name] = raw, source['source_type'], digest(raw)
        texts[name] = text
    keys, museums = set(), set()
    for artifact in data['artifacts']:
        if artifact['key'] in keys:
            raise ValueError('Duplicate artifact key.')
        keys.add(artifact['key'])
        inputs = []
        for claim in artifact['claims']:
            text = texts[claim['file']]
            if text.count(claim['quote']) != 1 or claim['value'] not in claim['quote']:
                raise ValueError(f'Claim must have one exact source match: {artifact["key"]}')
            inputs.append({k: claim[k] for k in ('field', 'value', 'quote')} | {
                'document_id': '00000000-0000-0000-0000-000000000001',
                'source_type': source_types[claim['file']]})
            if claim['field'] == 'museum':
                museums.add(claim['value'])
        ArtifactWrite(claims=inputs)
    if museums != set(data['museums']):
        raise ValueError('Artifact museum claims do not match the collection.')
    if len(data['cases']) != 30 or Counter(c['audience'] for c in data['cases']) != dict(visitor=10, staff=10, researcher=10):
        raise ValueError('Require ten scenarios for each of three audiences.')
    ids = set()
    for case in data['cases']:
        if case['id'] in ids or not case['turns']:
            raise ValueError('Each scenario needs a unique ID and turns.')
        ids.add(case['id'])
        if case['critical'] != bool(set(case['tags']) & CRITICAL_TAGS):
            raise ValueError('Critical scenarios must retain their acceptance gate.')
        for turn in case['turns']:
            if (not turn['question'].strip() or not turn['reference_answer'].strip()
                    or type(turn['reset_history']) is not bool
                    or any(not group or any(not isinstance(x, str) or not x for x in group) for group in turn['fact_groups'])
                    or any(name not in documents for name in turn['source_files'])):
                raise ValueError(f'Invalid question or expected evidence: {case["id"]}')
    sha = digest(json.dumps(hashes, sort_keys=True).encode())
    return {'data': data, 'documents': documents, 'texts': texts, 'hashes': hashes, 'sha256': sha, 'path': path}


def demo_root(path):
    root = Path(path).resolve()
    if root == DEMOS_ROOT.resolve() or not root.is_relative_to(DEMOS_ROOT.resolve()):
        raise ValueError(f'Demo storage must be a child of {DEMOS_ROOT}.')
    return root


def _open_index(doc_settings, vector_settings):
    from local_rag.artifacts.service import ArtifactService
    from local_rag.documents.service import DocumentStore
    from local_rag.indexing.service import IndexService
    index = IndexService(DocumentStore(doc_settings), vector_settings)
    try:
        index.artifacts = ArtifactService(index)
        return index
    except Exception:
        index.close()
        raise


def _open_rag(index, settings):
    from local_rag.rag.service import RagService
    return RagService(index, settings)


def _models(index, settings):
    from local_rag.clients.chat import create_chat_client
    client = create_chat_client(settings)
    client.require_local_model(settings.model_name, 'completion')
    return {'embedding': asdict(index.embedder.identity()),
            'chat': {'name': settings.model_name, 'version': client.model_digest(settings.model_name)}}


def error_record(exc):
    return {'type': type(exc).__name__, 'code': getattr(exc, 'code', None), 'message': str(exc)}


def model_versions_stable(before, after, results):
    # A no-sources refusal does not call generation and legitimately has version=None.
    # Only observed non-null versions can establish an in-flight model change.
    observed = [turn['response'].get('model', {}).get('version')
                for result in results for turn in result['turns'] if turn.get('response')]
    return before == after and all(version is None or version == before['chat']['version'] for version in observed)


def prepare(root=None, dataset=DATASET, model='qwen3:1.7b', progress=print):
    frozen = load_dataset(dataset)
    root = demo_root(root or DEMOS_ROOT / ('museum-' + uuid4().hex))
    if root.exists() and any(root.iterdir()):
        raise ValueError('Demo directory is populated; choose a new directory. No files were changed.')
    doc = replace(DocumentSettings.from_env(), storage_dir=root / 'documents')
    vec = replace(VectorSettings.from_env(), chroma_dir=root / 'chroma',
                  registry_path=root / 'registry/documents.sqlite3', auto_index=True)
    rag = replace(RagSettings.from_env(), model_name=model)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    # Exclusive reservation also prevents two concurrent preparations in an empty directory.
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'initializing', 'synthetic': frozen['data']['synthetic']}, stream)
    manifest = {'version': 1, 'benchmark_id': frozen['data']['benchmark_id'],
                'synthetic': frozen['data']['synthetic'], 'status': 'preparing',
                'started_at': now(), 'root': str(root), 'dataset_sha256': frozen['sha256'],
                'file_hashes': frozen['hashes'], 'documents': {}, 'artifacts': {},
                'configuration': {'documents': asdict(doc), 'vectors': asdict(vec), 'rag': asdict(rag)}}
    index = None
    try:
        (root / 'dataset/corpus').mkdir(parents=True)
        shutil.copyfile(frozen['path'], root / 'dataset/dataset.json')
        for name, raw in frozen['documents'].items():
            (root / 'dataset' / name).write_bytes(raw)
        write_json(manifest_path, manifest)
        index = _open_index(doc, vec)
        manifest['models'] = _models(index, rag)
        from local_rag.artifacts.models import ArtifactWrite
        source_types = {s['file']: s['source_type'] for s in frozen['data']['sources']}
        for name, raw in frozen['documents'].items():
            content_type = 'application/pdf' if Path(name).suffix == '.pdf' else 'text/plain'
            record = index.ingest(io.BytesIO(raw), Path(name).name, content_type)
            manifest['documents'][name] = {'document_id': record['document_id'], 'sha256': record['sha256'],
                                         'status': record['status'], 'index': record['index']}
            write_json(manifest_path, manifest)
            if record['status'] != 'indexed':
                raise RuntimeError(f'Document indexing failed: {name}')
            progress(f'Indexed source: {name}')
        for artifact in frozen['data']['artifacts']:
            claims = [{k: c[k] for k in ('field', 'value', 'quote')} | {
                'document_id': manifest['documents'][c['file']]['document_id'],
                'source_type': source_types[c['file']]} for c in artifact['claims']]
            record = index.artifacts.save(ArtifactWrite(claims=claims))
            manifest['artifacts'][artifact['key']] = {'artifact_id': record['artifact_id'],
                                                    'revision': record['revision'], 'index_status': record['index_status']}
            write_json(manifest_path, manifest)
            if record['index_status'] != 'indexed':
                raise RuntimeError(f'Artifact indexing failed: {artifact["key"]}: {record["index_error"]}')
            progress(f'Indexed artifact: {artifact["key"]}')
        manifest['models_after'] = _models(index, rag)
        if manifest['models_after'] != manifest['models']:
            raise RuntimeError('Model version changed during preparation; prepare a fresh demo.')
        manifest['status'] = 'ready'
    except Exception as exc:
        manifest.update(status='failed', error=error_record(exc))
        raise
    finally:
        manifest['finished_at'] = now()
        write_json(manifest_path, manifest)
        if index is not None:
            index.close()
    return manifest_path


def load_prepared(path):
    path = Path(path).resolve()
    root = demo_root(path.parent)
    if path.name != 'manifest.json':
        raise ValueError('Use the prepared manifest.json.')
    manifest = read_json(path)
    if (manifest.get('status') != 'ready' or type(manifest.get('synthetic')) is not bool
            or manifest.get('version') != 1 or Path(manifest['root']).resolve() != root):
        raise ValueError('Require a ready, unmodified demo manifest at its original location.')
    frozen = load_dataset(root / 'dataset/dataset.json')
    if manifest.get('benchmark_id') != frozen['data']['benchmark_id'] or manifest['synthetic'] != frozen['data']['synthetic']:
        raise ValueError('Manifest and dataset source classification differ.')
    if frozen['sha256'] != manifest['dataset_sha256'] or frozen['hashes'] != manifest['file_hashes']:
        raise ValueError('Frozen corpus changed; prepare a new version instead of altering a baseline.')
    cfg = manifest['configuration']
    doc, vec, rag = DocumentSettings(**cfg['documents']), VectorSettings(**cfg['vectors']), RagSettings(**cfg['rag'])
    if (doc.storage_dir != root / 'documents' or vec.chroma_dir != root / 'chroma'
            or vec.registry_path != root / 'registry/documents.sqlite3'):
        raise ValueError('Manifest storage paths do not belong to this demo.')
    if not vec.registry_path.is_file() or not vec.registry_path.with_suffix('.artifacts.sqlite3').is_file():
        raise ValueError('Prepared databases are missing.')
    return root, manifest, frozen, doc, vec, rag


def proxy_metrics(turn, response):
    answer = response.get('answer', '')
    groups = turn['fact_groups']
    hits = sum(any(alias.casefold() in answer.casefold() for alias in group) for group in groups)
    return {'fact_groups_hit': hits, 'fact_groups_total': len(groups),
            'fact_coverage': hits / len(groups) if groups else None,
            'note': 'Lexical proxy only; no semantic pass is inferred.'}


def summarize(report):
    results = report['results']
    reviewed = [r for r in results if r.get('semantic_review') is not None]
    passed = [r for r in reviewed if all(r['semantic_review'][k] for k in REVIEW_FIELDS)
              and all(t['error'] is None for t in r['turns'])]
    critical = [r for r in results if r['critical']]
    critical_passed = sum(r in passed for r in critical)
    complete = report['status'] == 'completed' and len(results) == 30
    report['summary'] = {'scenarios': len(results), 'expected_scenarios': 30,
                         'execution_errors': sum(t['error'] is not None for r in results for t in r['turns']),
                         'reviewed': len(reviewed), 'semantic_passed': len(passed),
                         'critical_scenarios': len(critical), 'critical_passed': critical_passed,
                         'acceptance_target': '27/30 semantic passes and every critical scenario passes',
                         'acceptance': ('pending_review' if complete and len(reviewed) < 30 else
                                        'met' if complete and len(passed) >= 27 and critical_passed == len(critical)
                                        and report.get('model_versions_stable') is True else 'not_met')}


def write_report(path, report):
    summarize(report)
    write_json(path, report)
    fields = ['case_id', 'audience', 'critical', 'turn', 'question', 'reference_answer', 'answer', 'status',
              'error', 'fact_coverage', 'elapsed_seconds', *REVIEW_FIELDS, 'reviewer', 'reviewer_kind', 'notes']
    with Path(path).with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in report['results']:
            for i, turn in enumerate(result['turns'], 1):
                response = turn.get('response') or {}
                row = dict(case_id=result['id'], audience=result['audience'], critical=result['critical'], turn=i,
                           question=turn['gold']['question'], reference_answer=turn['gold']['reference_answer'],
                           answer=response.get('answer'), status=response.get('status'),
                           error=json.dumps(turn['error'], ensure_ascii=False) if turn['error'] else '',
                           fact_coverage=turn['automatic']['fact_coverage'], elapsed_seconds=turn['elapsed_seconds'])
                row.update(result.get('semantic_review') or {})
                row = {k: ("'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
                       for k, v in row.items()}
                writer.writerow(row)


def evaluate(manifest_path, output=None, progress=print):
    root, manifest, frozen, doc, vec, rag_settings = load_prepared(manifest_path)
    run_id = uuid4().hex
    output = Path(output or root / 'reports' / f'evaluation-{run_id}.json').resolve()
    if output.parent != root / 'reports' or output.suffix != '.json' or output.exists() or output.with_suffix('.csv').exists():
        raise ValueError('Use a new .json report directly inside this demo\'s reports directory.')
    report = {'version': 1, 'run_id': run_id, 'benchmark_id': manifest['benchmark_id'],
              'synthetic': manifest['synthetic'], 'status': 'running', 'started_at': now(),
              'dataset_sha256': frozen['sha256'], 'configuration': manifest['configuration'],
              'models': manifest['models'], 'execution': {'interface': 'RagService.ask shared chat workflow',
              'top_k': None, 'concurrency': 1, 'history': 'actual completed answers; reset at each scenario and marked turn'},
              'review_rubric': {'correctness': 'Facts and identity are correct.',
              'completeness': 'All requested facts and required qualifications are included.',
              'groundedness': 'Every factual claim is supported; missing or conflicting evidence is respected.',
              'audience_suitability': 'Depth and language suit the requested audience.'},
              'limitations': [('Fictional small corpus; not real museum accuracy.' if manifest['synthetic'] else
                               'User-provided secondary compilations; not independently verified museum records.'),
                              'Automatic fact matching is not semantic review. Reviewer kind is recorded explicitly.'],
              'results': []}
    index = None
    write_report(output, report)
    try:
        index = _open_index(doc, vec)
        models = _models(index, rag_settings)
        if models != manifest['models']:
            raise ValueError('Installed model versions differ from preparation. Prepare a fresh demo.')
        for name, saved in manifest['documents'].items():
            record = index.get(saved['document_id'])
            actual = index.documents.directory(saved['document_id']) / 'text.txt'
            if (record['status'] != 'indexed' or record['sha256'] != saved['sha256']
                    or actual.read_text(encoding='utf-8').strip() != frozen['texts'][name].strip()):
                raise ValueError(f'Prepared source changed or is not indexed: {name}')
        for saved in manifest['artifacts'].values():
            record = index.artifacts.get(saved['artifact_id'])
            if record['revision'] != saved['revision'] or record['index_status'] != 'indexed':
                raise ValueError('Prepared artifact changed or is not indexed.')
        rag = _open_rag(index, rag_settings)
        for case in frozen['data']['cases']:
            result = {k: deepcopy(case[k]) for k in ('id', 'audience', 'tags', 'critical')}
            result.update(turns=[], semantic_review=None)
            report['results'].append(result)
            history = []
            for gold in case['turns']:
                if gold['reset_history']:
                    history = []
                turn = {'gold': gold, 'history_sent': deepcopy(history), 'response': None, 'error': None}
                start = time.perf_counter()
                try:
                    response = rag.ask(gold['question'], top_k=None, model=rag_settings.model_name, history=deepcopy(history))
                    turn['response'] = response
                    history.append({'question': gold['question'], 'answer': response['answer']})
                    observed = response.get('model', {}).get('version')
                    if observed and observed != models['chat']['version']:
                        raise ValueError('Chat model version changed during an answer.')
                except Exception as exc:
                    turn['error'] = error_record(exc)
                    # A failed turn cannot supply memory to subsequent turns.
                    history = turn['history_sent']
                turn['elapsed_seconds'] = round(time.perf_counter() - start, 4)
                turn['automatic'] = proxy_metrics(gold, turn['response'] or {})
                result['turns'].append(turn)
                write_report(output, report)
            progress(f'{case["id"]}: {len(result["turns"])} turns recorded')
        report['models_after'] = _models(index, rag_settings)
        report['model_versions_stable'] = model_versions_stable(models, report['models_after'], report['results'])
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', setup_error=error_record(exc))
        raise
    finally:
        report['finished_at'] = now()
        write_report(output, report)
        if index is not None:
            index.close()
    return output


def apply_review(report_path, review_path):
    """Import explicit semantic judgments without changing answers or lexical scores."""
    report_path = Path(report_path).resolve()
    demo_root(report_path.parent.parent)
    if report_path.parent.name != 'reports':
        raise ValueError('Review only a demo evaluation report.')
    report, review = read_json(report_path), read_json(review_path)
    if (report.get('status') != 'completed' or review.get('run_id') != report['run_id']
            or review.get('dataset_sha256') != report['dataset_sha256']):
        raise ValueError('Review must identify this completed run and frozen dataset.')
    entries = review['scenarios']
    if len(entries) != len(report['results']) or {r['id'] for r in entries} != {r['id'] for r in report['results']}:
        raise ValueError('Review must cover every scenario exactly once.')
    for entry in entries:
        if (any(type(entry.get(k)) is not bool for k in REVIEW_FIELDS)
                or entry.get('reviewer_kind') not in {'human', 'agent'}
                or not entry.get('reviewer', '').strip() or not entry.get('notes', '').strip()):
            raise ValueError('Review needs four boolean judgments, reviewer identity/kind, and notes.')
    by_id = {entry['id']: entry for entry in entries}
    for result in report['results']:
        entry = by_id[result['id']]
        result['semantic_review'] = {k: entry[k] for k in (*REVIEW_FIELDS, 'reviewer', 'reviewer_kind', 'notes')}
    if report.get('models') and report.get('models_after'):
        report['model_versions_stable'] = model_versions_stable(report['models'], report['models_after'], report['results'])
    report['reviewed_at'] = now()
    write_report(report_path, report)
    return report['summary']
