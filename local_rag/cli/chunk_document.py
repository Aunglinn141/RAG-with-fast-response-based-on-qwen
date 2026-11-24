"""Split one local document: python -m local_rag.cli.chunk_document path."""

import argparse
import json
from pathlib import Path
import sys

from local_rag.cli.document_output import output_destination, save_report
from local_rag.core.config import DocumentSettings
from local_rag.core.exceptions import DocumentError
from local_rag.documents.reader import DocumentReader
from local_rag.documents.splitters import DocumentChunker


def main(argv=None):
    parser = argparse.ArgumentParser(description="使用 LangChain 切分 PDF、Word、Excel、文本或 OCR 结果，保留原文来源；无需 Ollama 服务。")
    parser.add_argument("path", type=Path, help="需要切分的本地文档路径")
    parser.add_argument("--chunk-size", type=int, help="每块字符上限；优先于 DOCUMENT_CHUNK_SIZE，默认 700")
    parser.add_argument("--chunk-overlap", type=int, help="目标重叠字符数；优先于 DOCUMENT_CHUNK_OVERLAP，默认 100")
    parser.add_argument("--document-id", help="可选的文档 UUID；省略时自动生成")
    parser.add_argument("--output", type=Path, help="可选的新 JSON 文件；已有文件不会被覆盖")
    parser.add_argument("--ocr", choices=["auto", "always", "off"], help="覆盖本次读取的 OCR 模式")
    args = parser.parse_args(argv)
    try:
        destination = output_destination(args.path, args.output)
        settings = DocumentSettings.from_env(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        if args.ocr is not None:
            settings.ocr_mode = args.ocr
        document = DocumentReader(settings).read(args.path, document_id=args.document_id)
        result = DocumentChunker(settings.chunk_size, settings.chunk_overlap).split(document)
        serialized = json.dumps({"success": True, "data": result.to_dict(), "error": None},
                                ensure_ascii=False, indent=2)
        if destination is not None:
            save_report(destination, serialized)
    except DocumentError as exc:
        payload = {"success": False, "data": None, "error": {"code": exc.code, "message": exc.message}}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    except ValueError:
        payload = {
            "success": False, "data": None,
            "error": {"code": "invalid_configuration",
                      "message": "文档读取或切分配置无效，请检查 DOCUMENT_* 环境变量及切分参数；"
                                 "要求 50 <= chunk_size <= 10000，0 <= chunk_overlap < chunk_size。"},
        }
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
