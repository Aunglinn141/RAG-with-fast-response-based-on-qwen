"""Reproducible local retrieval ablations; optionally compare generated answers."""

import argparse
from dataclasses import asdict, replace
import io
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time
from uuid import uuid4

from local_rag.core.config import DocumentSettings, RagSettings, VectorSettings, PROJECT_ROOT, RUNTIME_DIR
from local_rag.documents.service import DocumentStore
from local_rag.indexing.service import IndexService
from local_rag.rag.retrieval import select_context_candidates
from local_rag.rag.scenarios import plan_retrieval, recognize_scene
from local_rag.rag.service import RagService


def coverage(text, facts):
    if not facts:
        return None
    return sum(any(alias in text for alias in fact) for fact in facts) / len(facts)


def verify_sources(sources, documents):
    for source in sources:
        meta = source["metadata"]
        record, text = documents[meta["document_id"]]
        assert text[meta["start_index"]:meta["end_index"]].startswith(source["text"])
        if "source_chunk_ids" in meta:
            assert text[meta["start_index"]:meta["end_index"]] == source["text"]
            ids = {c["chunk_id"] for c in record["chunks"]}
            assert set(meta["source_chunk_ids"]) <= ids


def main():
    parser = argparse.ArgumentParser(description="本地检索场景基准与四项策略消融；数据与正式库隔离。")
    parser.add_argument("--generate", action="store_true", help="额外对基线和完整策略运行本地聊天模型")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "reports/retrieval-scenarios.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "benchmarks/retrieval_scenarios_v1/dataset.json")
    args = parser.parse_args()
    dataset_path = args.dataset
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    root = RUNTIME_DIR / "tests" / ("retrieval-scenarios-" + uuid4().hex)
    settings = replace(RagSettings.from_env(), model_name=args.model, enhanced_retrieval=True,
                       parent_child=True, neighbor_radius=1, section_summaries=True,
                       hybrid_search=True, entity_retrieval=True, cross_chapter=True)
    modes = {
        "baseline": replace(settings, enhanced_retrieval=False),
        "full": settings,
        "no_parent": replace(settings, parent_child=False),
        "no_neighbors": replace(settings, neighbor_radius=0),
        "no_summaries": replace(settings, section_summaries=False),
        "no_hybrid": replace(settings, hybrid_search=False),
        "no_entities": replace(settings, entity_retrieval=False),
        "no_cross_chapter": replace(settings, cross_chapter=False),
    }
    report = {"completed": False, "dataset": dataset["version"], "test_directory": str(root),
              "embedding_provider": "real_local_bge_m3", "generate": args.generate,
              "settings": asdict(settings), "cases": [], "cleanup": [],
              "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
              "metric_note": "Literal reference fact coverage measures retrieval, not factual answer correctness."}
    records, documents, index = [], {}, None
    try:
        vector = replace(VectorSettings.from_env(), chroma_dir=root / "chroma",
                         registry_path=root / "registry.sqlite3", auto_index=True)
        index = IndexService(DocumentStore(DocumentSettings(storage_dir=root / "documents",
                                                           chunk_size=160, chunk_overlap=30)), vector)
        for doc in dataset["documents"]:
            record = index.ingest(io.BytesIO(doc["text"].encode("utf-8")), doc["filename"], "text/markdown")
            records.append(record)
            documents[record["document_id"]] = (record, (index.documents.directory(record["document_id"]) / "text.txt").read_text(encoding="utf-8"))
        for case in dataset["cases"]:
            question = case["question"]
            result = {"id": case["id"], "question": question, "expected_scene": case["scene"], "modes": {}}
            for mode, config in modes.items():
                started = time.perf_counter()
                rag = RagService(index, config)
                if config.enhanced_retrieval:
                    retrieved = index.retrieve_context(question, config.retrieval_candidates, None,
                                                       plan_retrieval(question, config), config)
                    contexts = retrieved["contexts"]
                else:
                    retrieved = index.search(question, config.retrieval_candidates)
                    contexts = select_context_candidates(retrieved["matches"], config.context_chunks)
                sources = rag._sources(question, contexts, balanced=config.enhanced_retrieval)
                verify_sources(sources, documents)
                result["modes"][mode] = {
                    "seconds": round(time.perf_counter() - started, 4),
                    "context_fact_coverage": coverage("\n".join(s["text"] for s in sources), case["facts"]),
                    "source_count": len(sources),
                    "section_count": len({s["metadata"].get("section_id") for s in sources if s["metadata"].get("section_id")}),
                    "sources": sources,
                    "strategy": retrieved.get("strategy"),
                }
            if args.generate:
                for mode in ("baseline", "full"):
                    started = time.perf_counter()
                    answer = RagService(index, modes[mode]).ask(question, top_k=None, model=args.model)
                    verify_sources(answer["sources"], documents)
                    by_id = {s["source_id"]: s["text"] for s in answer["sources"]}
                    assert all(e["quote"] in by_id[e["source_id"]] for e in answer["evidence"])
                    row = result["modes"][mode]
                    row["generation_seconds"] = round(time.perf_counter() - started, 4)
                    row["answer_fact_coverage"] = coverage(answer["answer"], case["facts"])
                    row["correct_abstention"] = answer["status"] == "insufficient_evidence" if case.get("unanswerable") else None
                    row["answer"] = answer
            assert recognize_scene(question)[0] == case["scene"]
            report["cases"].append(result)
            print(json.dumps({"case": case["id"], "coverage": {m: row["context_fact_coverage"] for m, row in result["modes"].items()}}, ensure_ascii=False), flush=True)
        report["summary"] = {}
        for mode in modes:
            rows = [c["modes"][mode] for c in report["cases"]]
            report["summary"][mode] = {
                "mean_context_fact_coverage": statistics.mean(r["context_fact_coverage"] for r in rows if r["context_fact_coverage"] is not None),
                "median_retrieval_seconds": statistics.median(r["seconds"] for r in rows),
            }
        report["completed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        print(f"FAIL: {exc}", flush=True)
    finally:
        if index:
            for record in records:
                report["cleanup"].append(index.delete(record["document_id"]))
            report["remaining_documents"] = len(index.list_documents())
            index.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(str(args.output.resolve()), flush=True)
    # Completion means the run and source checks succeeded, not perfect quality.
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
