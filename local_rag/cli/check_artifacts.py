"""Isolated real BGE-M3 smoke check using explicitly fictional archival records."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

from local_rag.artifacts.importing import draft_records
from local_rag.artifacts.models import ArtifactWrite, DraftRequest
from local_rag.artifacts.service import ArtifactService
from local_rag.core.config import DocumentSettings, RagSettings, RUNTIME_DIR, VectorSettings
from local_rag.documents.service import DocumentStore
from local_rag.indexing.service import IndexService
from local_rag.rag.service import RagService

SAMPLE = """软件验收的虚构档案，不代表真实文物或博物馆。
名称：验收云纹壶
年代：虚构测试时期
材料：青铜
用途：用于礼仪活动中盛放液体。
文化价值：用于演示器物档案的字段追溯流程。
收藏博物馆：虚构验收博物馆
馆藏编号：TEST-ARTIFACT-001
"""


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "reports" / "artifact-verification.json")
    parser.add_argument("--with-chat", action="store_true", help="同时通过普通问答链路验证真实本地 LLM")
    args = parser.parse_args()
    root = RUNTIME_DIR / "tests" / ("live-artifacts-" + uuid4().hex)
    root.mkdir(parents=True)
    settings = VectorSettings.from_env()
    settings.chroma_dir, settings.registry_path = root / "chroma", root / "registry.sqlite3"
    settings.auto_index = False
    index = IndexService(DocumentStore(DocumentSettings(storage_dir=root / "documents")), settings)
    service = ArtifactService(index)
    index.artifacts = service
    report = {"passed": False, "synthetic_records": True, "llm_used": False, "storage": str(root), "checks": []}
    try:
        doc = index.ingest(BytesIO(SAMPLE.encode("utf-8")), "虚构验收档案.txt", "text/plain")
        draft = draft_records(index, DraftRequest(document_id=doc["document_id"], source_type="official_archive"))
        record = service.save(ArtifactWrite.model_validate(draft["records"][0]))
        assert record["index_status"] == "indexed", record["index_error"]
        report["embedding_model"] = index.embedder.identity().__dict__
        report["checks"].append("real BGE-M3 1024-dimensional artifact indexing")
        answer = service.ask("介绍这件文物", record["artifact_id"])
        assert answer["status"] == "answered" and len(answer["facts"]) == 6
        report["answer"] = answer
        report["checks"].append("six factual fields with verified source excerpt positions")
        matches = service.search("在礼仪中使用的容器")["candidates"]
        assert matches and matches[0]["artifact_id"] == record["artifact_id"] and matches[0]["score"] is not None
        report["checks"].append("real semantic candidate retrieval")
        assert service.ask("重量多少？", record["artifact_id"])["status"] == "insufficient_evidence"
        report["checks"].append("missing facts abstain without a chat model")
        if args.with_chat:
            rag = RagService(index, RagSettings.from_env())
            result = rag.ask("验收云纹壶保存在哪一个博物馆中？", top_k=None)
            assert result["status"] == "answered" and "虚构验收博物馆" in result["answer"], result["answer"]
            assert any(s["metadata"].get("context_type") == "artifact" for s in result["sources"])
            report["llm_used"] = True
            report["chat_result"] = result
            report["checks"].append("shared chat workflow generates a grounded answer with the real selected LLM")
        service.close()
        service = ArtifactService(index)
        index.artifacts = service
        assert service.search("在礼仪中使用的容器")["candidates"][0]["artifact_id"] == record["artifact_id"]
        report["checks"].append("persisted SQLite and Chroma reopen successfully")
        service.delete(record["artifact_id"])
        assert service.vectors.count_document(record["artifact_id"]) == 0
        index.delete(doc["document_id"])
        report["checks"].append("sample records and vectors removed after verification")
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        index.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "report": str(args.output), "checks": report["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    cli()
