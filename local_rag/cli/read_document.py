"""Read one local document: python -m local_rag.cli.read_document path [--output report.json]."""

import argparse
import json
from pathlib import Path
import sys

from local_rag.cli.document_output import output_destination as _output_destination, save_report as _save_report
from local_rag.core.config import DocumentSettings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.reader import DocumentReader


def main(argv=None):
    parser = argparse.ArgumentParser(description="读取 PDF、Word、Excel、文本或图片；扫描件使用本地 OCR，输出全文与来源，不建立向量索引。")
    parser.add_argument("path", type=Path, help="需要读取的本地文档路径")
    parser.add_argument("--output", type=Path, help="可选的新 JSON 文件；已有文件不会被覆盖")
    parser.add_argument("--document-id", help="可选的文档 UUID；省略时自动生成")
    parser.add_argument("--ocr", choices=["auto", "always", "off"], help="覆盖本次读取的 OCR 模式")
    args = parser.parse_args(argv)
    try:
        destination = _output_destination(args.path, args.output)
        settings = DocumentSettings.from_env()
        if args.ocr is not None:
            settings.ocr_mode = args.ocr
        document = DocumentReader(settings).read(args.path, document_id=args.document_id)
        payload = {"success": True, "data": document.to_dict(), "error": None}
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        if destination is not None:
            _save_report(destination, serialized)
    except DocumentError as exc:
        payload = {"success": False, "data": None, "error": {"code": exc.code, "message": exc.message}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    except ValueError:
        payload = {"success": False, "data": None,
                   "error": {"code": "invalid_configuration", "message": "文档读取配置无效，请检查 DOCUMENT_* 环境变量。"}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    print(serialized)
    return 0


def cli():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
