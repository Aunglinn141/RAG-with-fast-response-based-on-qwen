"""Exercise real local BGE-M3 + Qwen on an isolated, disposable knowledge base."""

import argparse
import io
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

from local_rag.clients.chat import create_chat_client
from local_rag.core.config import DocumentSettings, VectorSettings, RagSettings, RUNTIME_DIR
from local_rag.documents.service import DocumentStore
from local_rag.indexing.service import IndexService
from local_rag.rag.service import RagService


def verify_sources(response, records):
    chunks = {chunk["chunk_id"]: chunk for record in records for chunk in record["chunks"]}
    sources = {s["source_id"]: s for s in response["sources"]}
    for source in sources.values():
        original = chunks.get(source["chunk_id"])
        if original is None or source["metadata"] != original["metadata"] or source["text"] not in original["text"]:
            raise RuntimeError("A response source does not match the indexed original document")
    for item in response["evidence"]:
        if item["source_id"] not in sources or item["quote"] not in sources[item["source_id"]]["text"]:
            raise RuntimeError("A model quotation does not match its actual retrieved source")
    if response["status"] == "answered" and not sources:
        raise RuntimeError("Answered without retrieved knowledge-base context")


def main():
    parser = argparse.ArgumentParser(description="真实本地 BGE-M3 + Qwen 知识库问答验收；使用独立测试数据。")
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "reports/rag-verification.json")
    args = parser.parse_args()
    index = None
    report = {"passed": False, "embedding_provider": "real_local_bge_m3", "chat_provider": "real_local_qwen", "results": {}}
    try:
        vector_settings, rag_settings = VectorSettings.from_env(), RagSettings.from_env()
        chat = create_chat_client(rag_settings)
        report["chat_provider"] = rag_settings.chat_backend
        chat.require_local_model(rag_settings.model_name, "completion")
        report["chat_model"] = {"name": rag_settings.model_name, "version": chat.model_digest(rag_settings.model_name)}
        root = RUNTIME_DIR / "tests" / ("live-rag-" + uuid4().hex)
        report["test_directory"] = str(root)
        vector_settings.chroma_dir = root / "chroma"
        vector_settings.registry_path = root / "registry/documents.sqlite3"
        vector_settings.auto_index = True
        index = IndexService(DocumentStore(DocumentSettings(storage_dir=root / "documents")), vector_settings)
        rag = RagService(index, rag_settings, chat)
        samples = [
            ("青岚实验室指南.txt", "青岚实验室的样品接收窗口每周二和周四开放，时间为14:20至16:40。送检前必须填写样品编号和保存温度。项目联络人是林杉。"),
            ("云岭茶仓说明.txt", "云岭茶叶仓库的干茶须密封避光保存，仓库相对湿度保持在45%至55%。仓库盘点安排在每月最后一个星期五。"),
            ("山谷图书室.txt", "山谷图书室每次可借阅三本书，借期十四天。借阅人应在服务台登记书名和归还日期。"),
            ("雨桐花圃.txt", "雨桐花圃的浇水安排在清晨进行。每个花盆贴有植物名称和上次浇水日期。"),
        ]
        records = [index.ingest(io.BytesIO(text.encode("utf-8")), filename, "text/plain") for filename, text in samples]
        print("[PASS] 四份样本文档已使用真实 BGE-M3 建立索引", flush=True)

        answer = rag.ask("青岚实验室的样品接收窗口在每周哪几天开放？")
        report["results"]["supported_question"] = answer
        verify_sources(answer, records)
        if answer["status"] != "answered" or not all(word in answer["answer"] for word in ("周二", "周四")):
            raise RuntimeError("Known-answer question did not produce the expected supported Chinese answer")
        if not any(s["metadata"]["document_id"] == records[0]["document_id"] for s in answer["sources"]):
            raise RuntimeError("Known-answer question did not retrieve the correct document")
        if not re.search(r"[\u4e00-\u9fff]", answer["answer"]):
            raise RuntimeError("Answer is not Chinese")
        print("[PASS] 中文答案包含预期事实，检索来源对应真实文档；不要求逐字引用", flush=True)

        missing = rag.ask("青岚实验室今年的年度预算是多少元？")
        report["results"]["missing_fact"] = missing
        verify_sources(missing, records)
        if missing["status"] != "insufficient_evidence" or missing["evidence"]:
            raise RuntimeError("Absent budget information should produce insufficient_evidence")
        print("[PASS] 知识库缺少年度预算，明确返回无法确定", flush=True)

        scoped = rag.ask("青岚实验室的样品接收窗口在每周哪几天开放？", document_id=records[1]["document_id"])
        report["results"]["restricted_document"] = scoped
        verify_sources(scoped, records)
        if scoped["status"] != "insufficient_evidence" or any(s["metadata"]["document_id"] != records[1]["document_id"] for s in scoped["sources"]):
            raise RuntimeError("Document scope was not respected")
        print("[PASS] 指定无关文档时不越过检索范围回答", flush=True)

        for record in records:
            index.delete(record["document_id"])
        empty = rag.ask("青岚实验室的样品接收窗口在每周哪几天开放？")
        report["results"]["after_delete"] = empty
        if empty["status"] != "insufficient_evidence" or empty["sources"]:
            raise RuntimeError("Deleted sources appeared in a new answer")
        report["passed"] = True
        print("[PASS] 删除后没有残留来源；测试文档及向量已清理", flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        print(f"[FAIL] {exc}", file=sys.stderr, flush=True)
    finally:
        if index is not None:
            index.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"报告：{args.output.resolve()}", flush=True)
    return 0 if report["passed"] else 1


def cli():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
