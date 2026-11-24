"""Run real local models against an isolated corpus and retain auditable results."""

from copy import deepcopy
import csv
import ctypes
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

from local_rag import __version__
from local_rag.clients.ollama import OllamaClient
from local_rag.clients.chat import create_chat_client
from local_rag.core.config import DocumentSettings, VectorSettings, RagSettings, RUNTIME_DIR
from local_rag.documents.service import DocumentStore
from local_rag.evaluation.artifacts import validate_report_path
from local_rag.evaluation.metrics import aggregate, evaluate_case, HUMAN_REVIEW_RUBRIC, METRIC_DEFINITIONS
from local_rag.indexing.service import IndexService
from local_rag.rag.service import RagService


def hardware_info():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or os.getenv("PROCESSOR_IDENTIFIER"),
            "logical_cpus": os.cpu_count(), "ram_total_bytes": None, "ram_available_bytes": None}
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ("total_phys", "available_phys", "total_page", "available_page",
                               "total_virtual", "available_virtual", "available_extended")]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            info.update(ram_total_bytes=status.total_phys, ram_available_bytes=status.available_phys)
        elif Path("/proc/meminfo").exists():
            rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            info["ram_total_bytes"] = int(rows["MemTotal"].split()[0]) * 1024
            info["ram_available_bytes"] = int(rows["MemAvailable"].split()[0]) * 1024
        else:
            info["ram_total_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        info["memory_measurement_error"] = str(exc)
    return info


class RecordingIndex:
    """Capture the one retrieval actually consumed by RagService."""

    def __init__(self, index):
        self.index = index
        self.reset()

    def reset(self):
        self.result, self.elapsed, self.calls = None, None, 0

    def search(self, *args, **kwargs):
        self.calls += 1
        start = time.perf_counter()
        try:
            self.result = self.index.search(*args, **kwargs)
            return self.result
        finally:
            self.elapsed = time.perf_counter() - start


class RecordingChat:
    """Keep raw model output even if service JSON/citation validation rejects it."""

    def __init__(self, client):
        self.client = client
        self.reset()

    def reset(self):
        self.raw, self.elapsed, self.calls = None, None, 0
        self.scope_raw, self.scope_calls = None, 0

    def model_digest(self, model):
        return self.client.model_digest(model)

    def classify(self, *args, **kwargs):
        self.scope_calls += 1
        start = time.perf_counter()
        try:
            self.scope_raw = self.client.classify(*args, **kwargs)
            return self.scope_raw
        finally:
            self.elapsed = (self.elapsed or 0) + time.perf_counter() - start

    def chat(self, *args, **kwargs):
        self.calls += 1
        start = time.perf_counter()
        try:
            self.raw = self.client.chat(*args, **kwargs)
            return self.raw
        finally:
            self.elapsed = (self.elapsed or 0) + time.perf_counter() - start


def error_record(exc):
    return {"type": type(exc).__name__, "code": getattr(exc, "code", None), "message": str(exc)}


def observed_version_mismatches(report):
    """A -> B -> A switches still invalidate a run even if final digests match."""
    mismatches = []
    def compare(kind, actual, location):
        expected = report["models"][kind]["digest"]
        if actual is not None and actual != expected:
            mismatches.append({"model": kind, "expected": expected, "observed": actual, **location})
    for document in report["documents"]:
        compare("embedding", document.get("index", {}).get("model_version"),
                {"stage": "index", "filename": document["filename"]})
    for result in report["results"]:
        location = {"case_id": result["case_id"], "repeat": result["repeat"]}
        retrieved_model = (result.get("retrieval_result") or {}).get("model") or {}
        answer_model = (result.get("response") or {}).get("model") or {}
        compare("embedding", retrieved_model.get("version"), {"stage": "retrieval", **location})
        compare("chat", answer_model.get("version"), {"stage": "generation", **location})
    for kind, actual in report.get("model_digests_after", {}).items():
        compare(kind, actual, {"stage": "after_run"})
    return mismatches


def write_report(output, report):
    """Checkpoint after each question; JSON is authoritative, CSV is for review."""
    output = validate_report_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(output)
    fields = ["case_id", "repeat", "question", "reference_answer", "answer", "status", "error",
              "hit_at_k", "recall_at_k", "reciprocal_rank", "fact_group_coverage", "decision_correct",
              "citation_source_precision", "quote_validity", "total_seconds", "retrieval_seconds",
              "model_call_seconds", "correctness_0_to_2", "completeness_0_to_2", "groundedness_0_to_2",
              "chinese_clarity_0_to_2", "reviewer", "notes"]
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            response, gold = result.get("response") or {}, result["gold"]
            row = {"case_id": result["case_id"], "repeat": result["repeat"], "question": gold["question"],
                   "reference_answer": gold["reference_answer"], "answer": response.get("answer"),
                   "status": response.get("status"), "error": json.dumps(result["error"], ensure_ascii=False) if result["error"] else ""}
            row.update({key: value for key, value in result["metrics"].items() if key in fields})
            row.update({key + "_seconds": value for key, value in result["latency_seconds"].items()})
            row.update(result["human_review"])
            # A model answer is untrusted spreadsheet content. Keep JSON untouched.
            for key, value in row.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    row[key] = "'" + value
            writer.writerow(row)


def run_benchmark(dataset, output, repeats=1, top_k=4, progress=print):
    output = validate_report_path(output)
    if type(repeats) is not int or not 1 <= repeats <= 100 or type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("Require 1 <= repeats <= 100 and 1 <= top_k <= 10")
    test_root = RUNTIME_DIR / "tests" / ("benchmark-" + uuid4().hex)
    started = time.perf_counter()
    report = {
        "report_version": 1, "benchmark_id": dataset.manifest["benchmark_id"],
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "dataset": {"sha256": dataset.sha256, "file_sha256": dataset.file_sha256,
                    "version": dataset.manifest["version"], "description": dataset.manifest["description"]},
        "parameters": {"top_k": top_k, "repeats": repeats, "order": "fixed manifest order, sequential",
                       "warmup_queries": 0, "model_temperature": 0},
        "execution": {"interface": "direct_service", "effective_concurrency": 1,
                      "api_limiter_applicable": False,
                      "ollama_resource_settings": {"verification": "not_introspected_by_runner",
                                                   "max_loaded_models": None, "num_parallel": None,
                                                   "max_queue": None},
                      "note": "CLI calls services directly and sequentially. Server startup settings require separate startup-log verification; client environment variables do not establish server settings."},
        "test_directory": str(test_root), "hardware": hardware_info(),
        "versions": {"python": sys.version, "local_rag": __version__},
        "metric_definitions": METRIC_DEFINITIONS, "human_review_rubric": HUMAN_REVIEW_RUBRIC,
        "limitations": ["Small synthetic corpus; not an estimate of production or long-document quality.",
                        "Fact matching is a lexical proxy; manual review is needed for semantic correctness.",
                        "Sequential runs include model loading and swapping; p95 with one repeat is descriptive only.",
                        "Metadata/quote validity proves origin, not entailment or completeness."],
        "models": {}, "documents": [], "results": [], "cleanup": {"status": "not_started"},
    }
    for package in ("chromadb", "langchain-text-splitters", "pydantic", "pypdf", "python-docx"):
        try:
            report["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["versions"][package] = None
    index, original_chunks, doc_ids = None, {}, {}
    completed_all, setup_error = False, None
    try:
        doc_settings = replace(DocumentSettings.from_env(), storage_dir=test_root / "documents")
        vector_settings = replace(VectorSettings.from_env(), chroma_dir=test_root / "chroma",
                                  registry_path=test_root / "registry/documents.sqlite3", auto_index=True)
        rag_settings = RagSettings.from_env()
        report["configuration"] = {"documents": asdict(doc_settings), "vectors": asdict(vector_settings),
                                   "rag": asdict(rag_settings)}
        chat_client = create_chat_client(rag_settings)
        embedding_client = OllamaClient(vector_settings.ollama_url, vector_settings.timeout)
        chat_client.require_local_model(rag_settings.model_name, "completion")
        embedding_client.require_local_model(vector_settings.model_name, "embedding")
        report["versions"][rag_settings.chat_backend + "_chat"] = chat_client.version()
        if rag_settings.chat_backend == "vllm":
            report["limitations"].append("vLLM model identities describe server metadata, not verified weight checksums.")
        report["versions"]["ollama_embedding"] = embedding_client.version()
        report["models"] = {
            "chat": {"name": rag_settings.model_name, "digest": chat_client.model_digest(rag_settings.model_name)},
            "embedding": {"name": vector_settings.model_name, "digest": embedding_client.model_digest(vector_settings.model_name), "dimension": 1024},
        }
        index = IndexService(DocumentStore(doc_settings), vector_settings)
        for filename in dataset.manifest["corpus_files"]:
            start = time.perf_counter()
            record = index.ingest(io.BytesIO(dataset.documents[filename]), filename,
                                  "text/markdown" if filename.endswith(".md") else "text/plain")
            doc_ids[filename] = record["document_id"]
            original_chunks.update({chunk["chunk_id"]: chunk for chunk in record["chunks"]})
            report["documents"].append({"filename": filename, "document_id": record["document_id"],
                                        "sha256": record["sha256"], "chunks": deepcopy(record["chunks"]),
                                        "index": record["index"], "index_seconds": time.perf_counter() - start})
            progress(f"Indexed {filename}: {record['chunk_count']} chunks")
        report["setup_seconds"] = time.perf_counter() - started
        recorder, chat = RecordingIndex(index), RecordingChat(chat_client)
        rag = RagService(recorder, rag_settings, chat)
        write_report(output, report)
        for repeat in range(1, repeats + 1):
            for case in dataset.cases:
                recorder.reset()
                chat.reset()
                response, error = None, None
                start = time.perf_counter()
                try:
                    response = rag.ask(case["question"], top_k=top_k,
                                       document_id=doc_ids.get(case.get("document_scope")))
                except Exception as exc:
                    error = error_record(exc)
                elapsed = time.perf_counter() - start
                matches = (recorder.result or {}).get("matches", [])[:top_k]
                result = {
                    "case_id": case["id"], "repeat": repeat, "attempted": True,
                    "gold": deepcopy(case), "response": response, "raw_model_reply": chat.raw, "raw_scope_reply": chat.scope_raw,
                    "retrieved": deepcopy(matches), "retrieval_result": recorder.result,
                    "calls": {"retrieval": recorder.calls, "chat": chat.calls, "scope": chat.scope_calls}, "error": error,
                    "latency_seconds": {"total": elapsed, "retrieval": recorder.elapsed, "model_call": chat.elapsed},
                    "metrics": evaluate_case(case, response, matches, original_chunks, error),
                    "human_review": {"correctness_0_to_2": None, "completeness_0_to_2": None,
                                     "groundedness_0_to_2": None, "chinese_clarity_0_to_2": None,
                                     "reviewer": None, "notes": None},
                }
                report["results"].append(result)
                report["summary"] = aggregate(report["results"])
                write_report(output, report)
                progress(f"{case['id']} repeat={repeat}: {(response or {}).get('status', 'error')}, {elapsed:.2f}s")
        report["model_digests_after"] = {
            "chat": chat_client.model_digest(rag_settings.model_name),
            "embedding": embedding_client.model_digest(vector_settings.model_name),
        }
        report["model_version_mismatches"] = observed_version_mismatches(report)
        report["model_versions_stable"] = not report["model_version_mismatches"]
        if not report["model_versions_stable"]:
            raise RuntimeError("Model digest changed during benchmark; results cannot share one baseline")
        completed_all = True
    except Exception as exc:
        setup_error = error_record(exc)
        report["run_error"] = setup_error
    finally:
        report.setdefault("setup_seconds", time.perf_counter() - started)
        existing = {(item["repeat"], item["case_id"]) for item in report["results"]}
        for repeat in range(1, repeats + 1):
            for case in dataset.cases:
                if (repeat, case["id"]) in existing:
                    continue
                error = setup_error or {"type": "Interrupted", "code": None, "message": "Run interrupted before question"}
                report["results"].append({
                    "case_id": case["id"], "repeat": repeat, "attempted": False,
                    "gold": deepcopy(case), "response": None, "raw_model_reply": None,
                    "retrieved": [], "retrieval_result": None, "calls": {"retrieval": 0, "chat": 0},
                    "error": error, "latency_seconds": {"total": None, "retrieval": None, "model_call": None},
                    "metrics": evaluate_case(case, None, [], original_chunks, error),
                    "human_review": {"correctness_0_to_2": None, "completeness_0_to_2": None,
                                     "groundedness_0_to_2": None, "chinese_clarity_0_to_2": None,
                                     "reviewer": None, "notes": None},
                })
        # Keep failed run databases for diagnosis. On completion delete only this UUID's samples.
        if index is not None:
            if completed_all:
                cleanup_errors = []
                for filename, document_id in doc_ids.items():
                    try:
                        index.delete(document_id)
                    except Exception as exc:
                        cleanup_errors.append({"filename": filename, **error_record(exc)})
                report["cleanup"] = {"status": "failed" if cleanup_errors else "deleted_samples",
                                     "errors": cleanup_errors, "retained": "SQLite tombstones, Chroma DB and report original chunks"}
            else:
                report["cleanup"] = {"status": "retained_for_diagnosis", "directory": str(test_root)}
            try:
                index.close()
            except Exception as exc:
                report["close_error"] = error_record(exc)
        report["summary"] = aggregate(report["results"])
        report["operational_success"] = bool(completed_all and not report["summary"]["errors"]
                                              and report["cleanup"]["status"] == "deleted_samples"
                                              and not report.get("close_error"))
        report["status"] = "completed" if report["operational_success"] else "completed_with_errors"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["run_seconds"] = time.perf_counter() - started
        write_report(output, report)
    return report
