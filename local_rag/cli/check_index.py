"""Real BGE-M3 acceptance check across fresh Python processes and isolated local data."""

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from local_rag.clients.ollama import OllamaError
from local_rag.core.config import DocumentSettings, VectorSettings, PROJECT_ROOT, RUNTIME_DIR
from local_rag.core.exceptions import DocumentError
from local_rag.documents.service import DocumentStore
from local_rag.embeddings.bge import BGEEmbeddings
from local_rag.indexing.service import IndexService


def phase(root, operation):
    config = VectorSettings.from_env()
    config.chroma_dir = root / "chroma"
    config.registry_path = root / "registry/documents.sqlite3"
    config.auto_index = True
    index = IndexService(DocumentStore(DocumentSettings(storage_dir=root / "documents")), config)
    state_path = root / "state.json"
    try:
        if operation == "create":
            records = []
            samples = [
                ("果园说明.txt", "苹果果园的采收与储存：成熟苹果采收后放入低温仓库，保持新鲜。"),
                ("航天说明.txt", "航天飞船通过火箭进入太空，宇航员在空间站进行科学实验。"),
            ]
            for filename, text in samples:
                records.append(index.ingest(io.BytesIO(text.encode()), filename, "text/plain"))
            state = {"document_ids": [r["document_id"] for r in records],
                     "model": {key: records[0]["index"][key]
                               for key in ("model_name", "model_version", "embedding_dimension")}}
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return state
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if operation == "search":
            results = index.search("苹果采收之后怎样保存新鲜？", limit=1)["matches"]
            if not results or results[0]["metadata"]["document_id"] != state["document_ids"][0]:
                raise RuntimeError("Fresh-process search did not rank the orchard document first")
            record = index.reindex(state["document_ids"][0])
            if index.vectors.count_document(record["document_id"]) != record["chunk_count"]:
                raise RuntimeError("Reindex left duplicate or missing vectors")
            if not index.search("苹果", document_id=record["document_id"])["matches"]:
                raise RuntimeError("Reindexed document cannot be searched")
            return {"restart_search": True, "reindex": True, "match": results[0]}
        if operation == "delete":
            for document_id in state["document_ids"]:
                index.delete(document_id)
                if index.vectors.count_document(document_id) or (root / "documents" / document_id).exists():
                    raise RuntimeError("Document deletion left files or vectors")
            return {"deleted": True}
        if index.search("苹果")["matches"]:
            raise RuntimeError("Deleted documents appeared after restart")
        for document_id in state["document_ids"]:
            if index.vectors.count_document(document_id):
                raise RuntimeError("Deleted vectors survived restart")
        return {"restart_after_delete": True, "remaining_vectors": 0}
    finally:
        index.close()


def main():
    parser = argparse.ArgumentParser(description="真实 BGE-M3 + Chroma + SQLite 跨进程验收；使用独立测试目录。")
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "reports/index-verification.json")
    parser.add_argument("--phase", choices=["create", "search", "delete", "verify_deleted"], help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.phase:
            print(json.dumps(phase(args.root, args.phase), ensure_ascii=False))
            return 0
        # Fail before creating data if the actual local BGE-M3 model is unavailable.
        config = VectorSettings.from_env()
        provider = BGEEmbeddings(config)
        identity = provider.identity()
        provider.embed(["验证本地 BGE-M3 生成真实向量。"])
        print(f"[PASS] 本地 {identity.name}，维度 {identity.dimension}，版本 {identity.version}", flush=True)
        root = RUNTIME_DIR / "tests" / ("live-index-" + uuid4().hex)
        root.mkdir(parents=True)
        results = {}
        for operation in ("create", "search", "delete", "verify_deleted"):
            child = subprocess.run(
                [sys.executable, "-m", "local_rag.cli.check_index", "--phase", operation, "--root", str(root)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=max(600, config.timeout * 10))
            if child.returncode:
                raise RuntimeError(f"{operation} failed; test data kept at {root}\n{child.stderr or child.stdout}")
            results[operation] = json.loads(child.stdout)
            print(f"[PASS] {operation}（独立 Python 进程）", flush=True)
        report = {"passed": True, "embedding_provider": "real_local_bge_m3", "test_directory": str(root),
                  "results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"验收通过，报告：{args.output.resolve()}")
        return 0
    except (OllamaError, DocumentError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


def cli():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
