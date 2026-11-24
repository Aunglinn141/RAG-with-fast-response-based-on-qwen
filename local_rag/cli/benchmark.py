"""python -m local_rag.cli.benchmark --output .runtime/reports/benchmark-v1.json"""

import argparse
import json
from pathlib import Path
import sys

from local_rag.core.config import RUNTIME_DIR
from local_rag.evaluation.artifacts import validate_report_path
from local_rag.evaluation.dataset import DEFAULT_DATASET, load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定中文 RAG 基准：本地 BGE-M3 + Qwen，独立测试数据库。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "reports/benchmark-v1.json")
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--top-k", type=int, default=4, choices=range(1, 11), metavar="1..10")
    parser.add_argument("--validate-only", action="store_true", help="只验证冻结语料与标准答案，不连接模型")
    args = parser.parse_args(argv)
    try:
        args.output = validate_report_path(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    dataset = load_dataset(args.dataset)
    if args.validate_only:
        print(json.dumps({"benchmark_id": dataset.manifest["benchmark_id"], "sha256": dataset.sha256,
                          "documents": len(dataset.documents), "cases": len(dataset.cases)}, ensure_ascii=False, indent=2))
        return 0
    from local_rag.evaluation.runner import run_benchmark
    report = run_benchmark(dataset, args.output, args.repeats, args.top_k)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output.resolve()}\nCSV: {args.output.with_suffix('.csv').resolve()}")
    print("自动事实覆盖率是词面匹配指标；请结合标准答案与原始证据进行人工质量评分。")
    return 0 if report["operational_success"] else 1


def cli():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
