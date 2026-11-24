"""Gold-data integrity and independently calculated metric expectations; no model downloads."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from local_rag.evaluation.dataset import DEFAULT_DATASET, load_dataset
from local_rag.evaluation.metrics import aggregate, distribution, evaluate_case
from local_rag.evaluation.runner import RecordingChat, RecordingIndex, hardware_info, observed_version_mismatches, run_benchmark, write_report


def gold_case(abstain=False):
    return {"id": "unit", "question": "借期是多少天？", "reference_answer": "14天。",
            "expected_source_files": [] if abstain else ["a.txt", "b.txt"],
            "required_fact_groups": [] if abstain else [["14天", "十四天"], ["3本", "三本"]],
            "should_abstain": abstain}


def chunk(chunk_id, filename, text="最多3本，初次借期为14天。"):
    return {"chunk_id": chunk_id, "text": text,
            "metadata": {"filename": filename, "document_id": "doc-" + filename, "chunk_index": 1}}


def result_row(case, response=None, retrieved=None, error=None, seconds=1.0, attempted=True):
    retrieved = retrieved or []
    return {"case_id": case["id"], "repeat": 1, "gold": case, "response": response,
            "error": error, "attempted": attempted, "human_review": {},
            "latency_seconds": {"total": seconds, "retrieval": None, "model_call": None},
            "metrics": evaluate_case(case, response, retrieved, {c["chunk_id"]: c for c in retrieved}, error)}


def test_frozen_chinese_corpus_gold_quotes_and_counts():
    dataset = load_dataset()
    assert len(dataset.documents) == 10
    assert len(dataset.cases) == 22
    assert sum(not case["should_abstain"] for case in dataset.cases) == 16
    assert sum(case["should_abstain"] for case in dataset.cases) == 6
    assert dataset.sha256 == (DEFAULT_DATASET.parent / "SHA256").read_text().strip()
    assert len(dataset.file_sha256) == 11
    assert any(len(case["expected_source_files"]) == 2 for case in dataset.cases)


def test_dataset_rejects_changed_gold_quote(isolated_root):
    tmp_path = isolated_root
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus/a.txt").write_text("借期为14天。", encoding="utf-8")
    manifest = {"corpus_files": ["a.txt"], "cases": [{
        "id": "q", "question": "借期？", "reference_answer": "14天", "should_abstain": False,
        "expected_source_files": ["a.txt"], "required_fact_groups": [["14天"]],
        "support_quotes": [{"file": "a.txt", "text": "借期为99天。"}],
    }]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not verbatim"):
        load_dataset(path)


def test_retrieval_uses_chunk_rank_and_distinct_gold_document_recall():
    retrieved = [chunk("x1", "x.txt"), chunk("x2", "x.txt"), chunk("a1", "a.txt"), chunk("a2", "a.txt")]
    metrics = evaluate_case(gold_case(), None, retrieved, {})
    assert metrics["hit_at_k"] == 1
    assert metrics["recall_at_k"] == 0.5
    assert metrics["reciprocal_rank"] == pytest.approx(1 / 3)


def test_alias_groups_normalize_width_spacing_and_do_not_claim_semantics():
    response = {"status": "answered", "answer": "可以借 １４ 天，同时三本。"}
    metrics = evaluate_case(gold_case(), response, [], {})
    assert metrics["fact_group_coverage"] == 1
    assert metrics["all_fact_groups_present"] is True
    # Lexical match alone cannot establish provenance: this answer has no citations.
    assert metrics["all_answer_citations_valid"] is False
    assert metrics["quote_validity"] is None


def test_real_quote_and_expected_document_are_independent_metrics():
    good, distractor = chunk("a1", "a.txt"), chunk("x1", "x.txt")
    sources = [{**good, "source_id": "S1"}, {**distractor, "source_id": "S2"}]
    response = {"status": "answered", "answer": "14天，3本。", "sources": sources,
                "evidence": [{"source_id": "S1", "quote": "借期为14天"},
                             {"source_id": "S2", "quote": "借期为14天"}]}
    metrics = evaluate_case(gold_case(), response, [good, distractor], {"a1": good, "x1": distractor})
    assert metrics["citation_source_precision"] == .5
    assert metrics["quote_validity"] == 1
    assert metrics["all_answer_citations_valid"] is True


@pytest.mark.parametrize("tamper", ["metadata", "text", "not_retrieved", "unknown_source", "invented_quote"])
def test_citation_provenance_rejects_fabricated_sources(tamper):
    original = chunk("a1", "a.txt")
    source = {**deepcopy(original), "source_id": "S1"}
    retrieved = [deepcopy(original)]
    item = {"source_id": "S1", "quote": "借期为14天"}
    if tamper == "metadata":
        source["metadata"]["filename"] = "forged.txt"
    elif tamper == "text":
        source["text"] = "借期为14天，伪造结论。"
    elif tamper == "not_retrieved":
        retrieved = []
    elif tamper == "unknown_source":
        item["source_id"] = "S99"
    else:
        item["quote"] = "借期为99天"
    response = {"status": "answered", "answer": "14天。", "sources": [source], "evidence": [item]}
    metrics = evaluate_case(gold_case(), response, retrieved, {"a1": original})
    assert metrics["quote_validity"] == 0
    assert metrics["all_answer_citations_valid"] is False


def test_errors_count_against_answers_and_abstentions_and_remain_in_latency():
    rows = [result_row(gold_case(), {"status": "answered", "answer": "14天，3本。"}, seconds=1),
            result_row(gold_case(), error={"message": "model disconnected"}, seconds=9),
            result_row(gold_case(True), {"status": "insufficient_evidence"}, seconds=2),
            result_row(gold_case(True), error={"message": "timeout"}, seconds=8)]
    summary = aggregate(rows)
    assert summary["errors"] == 2
    assert summary["decision_accuracy"] == .5
    assert summary["abstention_accuracy"] == .5
    assert summary["fact_group_coverage_proxy"] == .5
    assert summary["answerable_cases"] == 2
    assert summary["latency_seconds"] == {"count": 4, "mean": 5, "p50": 5, "p95": pytest.approx(8.85), "min": 1, "max": 9}


def test_unanswerable_excluded_from_retrieval_denominator_and_no_citation_is_undefined():
    summary = aggregate([result_row(gold_case(True), {"status": "insufficient_evidence"})])
    assert summary["retrieval_hit_at_k"] is None
    assert summary["retrieval_recall_at_k"] is None
    assert summary["retrieval_mrr_at_k"] is None
    assert summary["citation_source_precision"] is None
    assert summary["quote_validity"] is None
    assert summary["abstention_accuracy"] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_latency_rejects_invalid_samples(bad):
    with pytest.raises(ValueError):
        distribution([bad])


def test_scoped_error_is_not_credited_as_scope_success():
    case = {**gold_case(True), "document_scope": "a.txt"}
    metrics = evaluate_case(case, None, [], {}, {"message": "failure"})
    assert metrics["scope_respected"] is False


def test_recorders_capture_actual_one_call_and_raw_reply_even_if_service_later_rejects():
    class Client:
        def search(self, question, limit, document_id):
            return {"query": question, "matches": [chunk("a1", "a.txt")]}
        def chat(self, *args, **kwargs):
            return "not valid JSON"
    index, chat = RecordingIndex(Client()), RecordingChat(Client())
    assert index.search("q", 4, None) is index.result
    assert index.calls == 1 and index.elapsed >= 0
    assert chat.chat("model", "question") == "not valid JSON"
    assert chat.raw == "not valid JSON" and chat.calls == 1
    index.reset()
    chat.reset()
    assert index.result is None and index.elapsed is None and index.calls == 0
    assert chat.raw is None and chat.elapsed is None and chat.calls == 0


def test_setup_connection_failure_creates_complete_error_report_without_indexing(isolated_root, monkeypatch):
    tmp_path = isolated_root
    import local_rag.evaluation.runner as runner
    class UnavailableClient:
        def __init__(self, *args):
            pass
        def require_local_model(self, *args):
            raise ConnectionError("offline test")
    monkeypatch.setattr(runner, "OllamaClient", UnavailableClient)
    monkeypatch.setattr(runner, "create_chat_client", lambda settings: UnavailableClient())
    monkeypatch.setattr(runner, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(runner, "IndexService", lambda *args: pytest.fail("No indexing before model preflight"))
    output = tmp_path / "report.json"
    report = run_benchmark(load_dataset(), output, progress=lambda _: None)
    assert report["operational_success"] is False
    assert report["summary"]["errors"] == 22
    assert report["summary"]["not_attempted"] == 22
    assert report["summary"]["latency_seconds"]["count"] == 0
    assert report["summary"]["decision_accuracy"] == 0
    assert report["summary"]["fact_group_coverage_proxy"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["run_error"]["type"] == "ConnectionError"
    assert output.with_suffix(".csv").exists()


def test_runner_success_uses_isolated_paths_one_retrieval_and_cleans_only_its_samples(isolated_root, monkeypatch):
    tmp_path = isolated_root
    import local_rag.evaluation.runner as runner
    from hashlib import sha256
    from uuid import uuid4
    stores = []
    class LocalClient:
        def __init__(self, *args):
            pass
        def require_local_model(self, *args):
            return {}
        def model_digest(self, model):
            return "digest-" + model
        def version(self):
            return "test-version"
        def classify(self, *args, **kwargs):
            return '{"within_scope":true}'
        def chat(self, *args, **kwargs):
            return '{"answerable":false,"answer":"","evidence":[]}'
    class IsolatedIndex:
        def __init__(self, documents, settings):
            self.records, self.deleted, self.searches = {}, [], 0
            self.closed = False
            assert documents.root.is_relative_to(tmp_path / "tests")
            assert settings.chroma_dir.is_relative_to(tmp_path / "tests")
            assert settings.registry_path.is_relative_to(tmp_path / "tests")
            stores.append(self)
        def ingest(self, stream, filename, content_type):
            data, document_id = stream.read(), str(uuid4())
            part = {"chunk_id": document_id + ":1", "text": data.decode("utf-8"),
                    "metadata": {"document_id": document_id, "filename": filename, "chunk_index": 1}}
            record = {"document_id": document_id, "filename": filename, "sha256": sha256(data).hexdigest(),
                      "chunk_count": 1, "chunks": [part], "index": {"status": "indexed"}}
            self.records[document_id] = record
            return record
        def search(self, question, limit, document_id):
            self.searches += 1
            records = [self.records[document_id]] if document_id else self.records.values()
            return {"query": question, "matches": [r["chunks"][0] for r in records][:limit], "model": {"name": "stub"}}
        def delete(self, document_id):
            self.deleted.append(document_id)
            del self.records[document_id]
        def close(self):
            self.closed = True
    monkeypatch.setattr(runner, "OllamaClient", LocalClient)
    monkeypatch.setattr(runner, "create_chat_client", lambda settings: LocalClient())
    monkeypatch.setattr(runner, "IndexService", IsolatedIndex)
    monkeypatch.setattr(runner, "RUNTIME_DIR", tmp_path)
    output = tmp_path / "success.json"
    report = run_benchmark(load_dataset(), output, progress=lambda _: None)
    assert report["operational_success"] is True
    assert report["summary"]["attempted"] == 22
    assert stores[0].searches == 22
    assert len(stores[0].deleted) == 10 and not stores[0].records and stores[0].closed
    assert report["cleanup"]["status"] == "deleted_samples"
    assert all(result["calls"] == {"retrieval": 1, "chat": 1, "scope": 1} for result in report["results"])
    assert all(result["raw_model_reply"] for result in report["results"])
    assert len(report["documents"]) == 10  # Original chunks survive sample cleanup for review.
    assert report["summary"]["answerable_answer_rate"] == 0  # Execution success is not quality success.
    assert report["summary"]["abstention_accuracy"] == 1
    assert report["model_versions_stable"] is True
    assert report["execution"]["interface"] == "direct_service"
    assert report["execution"]["effective_concurrency"] == 1
    assert report["execution"]["api_limiter_applicable"] is False
    assert report["execution"]["ollama_resource_settings"]["num_parallel"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"


def test_csv_preserves_raw_json_but_escapes_formula_content(isolated_root):
    tmp_path = isolated_root
    row = result_row(gold_case(), {"status": "answered", "answer": "=1+1"})
    output = tmp_path / "report.json"
    write_report(output, {"results": [row]})
    assert json.loads(output.read_text(encoding="utf-8"))["results"][0]["response"]["answer"] == "=1+1"
    assert "'=1+1" in output.with_suffix(".csv").read_text(encoding="utf-8-sig")


def test_hardware_is_measured_without_optional_dependencies():
    info = hardware_info()
    assert info["logical_cpus"] is None or info["logical_cpus"] > 0
    assert info["ram_total_bytes"] is None or info["ram_total_bytes"] > 0
    if info["ram_available_bytes"] is not None:
        assert 0 <= info["ram_available_bytes"] <= info["ram_total_bytes"]


@pytest.mark.parametrize("filename", ["report.csv", "report.CSV", "report.txt", "report"])
def test_non_json_output_rejected_before_models_or_overwriting_files(isolated_root, monkeypatch, filename):
    import local_rag.evaluation.runner as runner
    from local_rag.cli.benchmark import main
    output = isolated_root / filename
    output.write_text("keep this existing artifact", encoding="utf-8")
    monkeypatch.setattr(runner, "IndexService", lambda *args: pytest.fail("Invalid output must not index"))
    monkeypatch.setattr(runner, "OllamaClient", lambda *args: pytest.fail("Invalid output must not call models"))
    with pytest.raises(ValueError, match=r"\.json"):
        write_report(output, {"results": []})
    with pytest.raises(ValueError, match=r"\.json"):
        run_benchmark(load_dataset(), output)
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(output)])
    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep this existing artifact"


def test_models_switching_away_and_back_are_not_reported_as_stable():
    report = {
        "models": {"embedding": {"digest": "embed-a"}, "chat": {"digest": "chat-a"}},
        "model_digests_after": {"embedding": "embed-a", "chat": "chat-a"},
        "documents": [{"filename": "a.txt", "index": {"model_version": "embed-b"}}],
        "results": [
            {"case_id": "q1", "repeat": 1, "retrieval_result": {"model": {"version": "embed-a"}},
             "response": {"model": {"version": "chat-a"}}},
            {"case_id": "q2", "repeat": 1, "retrieval_result": {"model": {"version": "embed-b"}},
             "response": {"model": {"version": "chat-b"}}},
            {"case_id": "q3", "repeat": 1, "retrieval_result": {"model": {"version": "embed-a"}},
             "response": {"model": {"version": "chat-a"}}},
        ],
    }
    mismatches = observed_version_mismatches(report)
    assert [item["stage"] for item in mismatches] == ["index", "retrieval", "generation"]
    assert [item["observed"] for item in mismatches] == ["embed-b", "embed-b", "chat-b"]
    assert all(item.get("case_id") == "q2" for item in mismatches[1:])


def test_knowledge_only_answers_do_not_report_quote_validation_as_pass_or_failure():
    response = {"status": "answered", "answer": "14天，3本。", "answer_policy": "knowledge_base_only",
                "sources": [], "evidence": []}
    row = result_row(gold_case(), response)
    assert row["metrics"]["fact_group_coverage"] == 1
    assert row["metrics"]["all_answer_citations_valid"] is None
    assert aggregate([row])["answerable_valid_citation_rate"] is None
