"""Prepare and evaluate isolated museum collections from fixtures or existing PDFs."""

import argparse
import json
from pathlib import Path
import sys

from local_rag.evaluation import museum_demo as demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Validate the frozen corpus without models or writes')
    parser.add_argument('--dataset', type=Path, default=demo.DATASET)
    commands = parser.add_subparsers(dest='command')
    prepare = commands.add_parser('prepare', help='Create a new isolated demo and both indexes')
    prepare.add_argument('--root', type=Path, help='New directory beneath .runtime/demos/')
    prepare.add_argument('--model', default='qwen3:1.7b')
    library = commands.add_parser('prepare-library', help='Use the four existing museum compilation PDFs')
    library.add_argument('--source-dir', type=Path, default=demo.RUNTIME_DIR / 'documents')
    library.add_argument('--root', type=Path, help='New directory beneath .runtime/demos/')
    library.add_argument('--model', default='qwen3:1.7b')
    evaluate = commands.add_parser('evaluate', help='Run all 30 scenarios on a prepared demo')
    evaluate.add_argument('--manifest', required=True, type=Path)
    evaluate.add_argument('--output', type=Path, help='New JSON path inside the demo reports directory')
    review = commands.add_parser('review', help='Attach explicit semantic review and compute acceptance')
    review.add_argument('--report', required=True, type=Path)
    review.add_argument('--review', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            dataset = demo.load_dataset(args.dataset)
            print(json.dumps({'valid': True, 'synthetic': dataset['data']['synthetic'], 'sha256': dataset['sha256'],
                              'sources': len(dataset['documents']), 'artifacts': len(dataset['data']['artifacts']),
                              'scenarios': len(dataset['data']['cases'])}, ensure_ascii=False))
        elif args.command == 'prepare':
            path = demo.prepare(args.root, args.dataset, args.model)
            print(f'Manifest: {path}')
        elif args.command == 'prepare-library':
            from local_rag.evaluation.museum_library import build_dataset
            dataset = build_dataset(args.source_dir)
            path = demo.prepare(args.root, dataset, args.model)
            print(f'Dataset: {dataset}\nManifest: {path}')
        elif args.command == 'evaluate':
            path = demo.evaluate(args.manifest, args.output)
            report = demo.read_json(path)
            print(json.dumps(report['summary'], ensure_ascii=False))
            print(f'JSON: {path}\nCSV: {path.with_suffix(".csv")}')
            return 1 if report['summary']['execution_errors'] or not report.get('model_versions_stable') else 0
        elif args.command == 'review':
            summary = demo.apply_review(args.report, args.review)
            print(json.dumps(summary, ensure_ascii=False))
            return 0 if summary['acceptance'] == 'met' else 2
        else:
            parser.error('Choose prepare, prepare-library, evaluate, review, or --validate-only.')
        return 0
    except Exception as exc:
        print(json.dumps({'success': False, 'error': demo.error_record(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


def cli():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())


if __name__ == '__main__':
    cli()
