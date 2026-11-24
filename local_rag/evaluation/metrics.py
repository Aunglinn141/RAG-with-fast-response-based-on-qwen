"""Inspectable retrieval/answer proxies; exact quote provenance is not entailment."""

import math
import statistics
import unicodedata


METRIC_DEFINITIONS = {
    "retrieval": "Answerable cases only, using actual top-K chunks. Hit is any gold document; recall is fraction of distinct gold documents; reciprocal rank uses the first relevant chunk rank (duplicates still occupy ranks).",
    "fact_group_coverage": "Fraction of required alias groups found in the normalized answer; case-macro average. This lexical proxy does not prove semantic correctness, number/subject alignment, completeness or absence of hallucinations. Errors and abstentions score zero on answerable cases.",
    "decision_accuracy": "Whether response.status is answered for answerable gold or insufficient_evidence for abstain gold. Any error is incorrect; this does not judge answer semantics.",
    "citation_source_precision": "Fraction of distinct cited source IDs whose document filename is gold and whose metadata/text correspond to the original indexed chunk and actual retrieved result. No citations means undefined, not perfect.",
    "quote_validity": "Fraction of evidence items whose quote is verbatim in the displayed source AND original indexed chunk, with metadata and retrieval provenance checked. This does not prove that the quote supports the answer.",
    "answerable_valid_citation_rate": "Legacy citation metric; not applicable (null) to knowledge_base_only answers, which do not generate quotations.",
    "latency": "Seconds from RagService.ask start through validation, including failed attempts; no warm-up discarded. Retrieval and chat request stages recorded once. Setup/not-attempted cases are counted as errors but excluded from timing with explicit attempted counts. Mean/p50/p95 use linear interpolation for percentiles.",
}

HUMAN_REVIEW_RUBRIC = {
    "correctness_0_to_2": "0: wrong/contradictory or failed; 1: partly correct; 2: all requested gold facts correct, including subject/number alignment.",
    "completeness_0_to_2": "0: no useful answer; 1: important omissions; 2: covers all requested facts or clearly abstains when gold says abstain.",
    "groundedness_0_to_2": "0: material invented facts; 1: partly unsupported; 2: every factual claim follows from the provided knowledge-base content.",
    "chinese_clarity_0_to_2": "0: unreadable/non-Chinese; 1: awkward or ambiguous; 2: clear Chinese answer.",
    "instruction": "Review raw answer, gold answer, retrieved texts and any legacy quotations together. Do not turn the automatic lexical proxy into a semantic correctness label. Leave scores null until a person has reviewed them.",
}


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def evaluate_case(case, response, retrieved, original_chunks, error=None):
    response = response or {}
    expected = set(case["expected_source_files"])
    names = [item.get("metadata", {}).get("filename") for item in retrieved]
    relevant_ranks = [rank for rank, name in enumerate(names, 1) if name in expected]
    answerable = not case["should_abstain"]
    answered = not error and response.get("status") == "answered"
    abstained = not error and response.get("status") == "insufficient_evidence"
    answer = normalize(response.get("answer", "")) if answered else ""
    groups = case["required_fact_groups"]
    fact_hits = [any(normalize(alias) in answer for alias in group) for group in groups]
    sources = {s["source_id"]: s for s in response.get("sources", [])}
    retrieved_by_id = {m["chunk_id"]: m for m in retrieved}

    def valid_source(source):
        if not source:
            return False
        original = original_chunks.get(source.get("chunk_id"))
        actual = retrieved_by_id.get(source.get("chunk_id"))
        return bool(original and actual and source.get("text")
                    and source.get("metadata") == original.get("metadata") == actual.get("metadata")
                    and source["text"] in original["text"] and source["text"] in actual["text"])

    evidence = response.get("evidence", [])
    cited_ids = set(item["source_id"] for item in evidence)
    correct_citations = sum(valid_source(sources.get(sid))
                            and sources[sid]["metadata"].get("filename") in expected for sid in cited_ids)
    valid_quotes = sum(valid_source(sources.get(item["source_id"])) and bool(item.get("quote"))
                       and item["quote"] in sources[item["source_id"]]["text"]
                       and item["quote"] in original_chunks[sources[item["source_id"]]["chunk_id"]]["text"]
                       for item in evidence)
    return {
        "answerable": answerable,
        "hit_at_k": int(bool(relevant_ranks)) if answerable else None,
        "recall_at_k": len(expected.intersection(names)) / len(expected) if answerable else None,
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else (0.0 if answerable else None),
        "fact_group_hits": fact_hits,
        "fact_group_coverage": sum(fact_hits) / len(groups) if groups else None,
        "all_fact_groups_present": bool(fact_hits) and all(fact_hits) if answerable else None,
        "answered": bool(answered),
        "abstained": bool(abstained),
        "decision_correct": bool(abstained if case["should_abstain"] else answered),
        "citation_count": len(cited_ids), "correct_source_citations": correct_citations,
        "citation_source_precision": correct_citations / len(cited_ids) if cited_ids else None,
        "quote_count": len(evidence), "valid_quote_count": valid_quotes,
        "quote_validity": valid_quotes / len(evidence) if evidence else None,
        "all_answer_citations_valid": (None if response.get("answer_policy") == "knowledge_base_only"
                                       else bool(answered and evidence and valid_quotes == len(evidence))),
        "scope_respected": bool(not error and all(item.get("metadata", {}).get("filename") == case["document_scope"]
                                                 for item in [*retrieved, *sources.values()])) if case.get("document_scope") else None,
    }


def distribution(values):
    values = sorted(float(x) for x in values if x is not None)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Latency samples must be finite and nonnegative")
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    def percentile(q):
        position = (len(values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"count": len(values), "mean": statistics.fmean(values), "p50": percentile(.5),
            "p95": percentile(.95), "min": values[0], "max": values[-1]}


def aggregate(results):
    answerable = [r for r in results if r["metrics"]["answerable"]]
    unanswerable = [r for r in results if not r["metrics"]["answerable"]]
    def mean(rows, key):
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        return statistics.fmean(values) if values else None
    citations = sum(r["metrics"]["citation_count"] for r in results)
    quotes = sum(r["metrics"]["quote_count"] for r in results)
    attempted = [r for r in results if r["attempted"]]
    return {
        "cases": len(results), "attempted": len(attempted), "not_attempted": len(results) - len(attempted),
        "errors": sum(bool(r.get("error")) for r in results),
        "answerable_cases": len(answerable), "abstain_cases": len(unanswerable),
        "retrieval_hit_at_k": mean(answerable, "hit_at_k"),
        "retrieval_recall_at_k": mean(answerable, "recall_at_k"),
        "retrieval_mrr_at_k": mean(answerable, "reciprocal_rank"),
        "fact_group_coverage_proxy": mean(answerable, "fact_group_coverage"),
        "all_fact_groups_present_rate_proxy": mean(answerable, "all_fact_groups_present"),
        "decision_accuracy": mean(results, "decision_correct"),
        "answerable_answer_rate": mean(answerable, "answered"),
        "abstention_accuracy": mean(unanswerable, "abstained"),
        "citation_source_precision": sum(r["metrics"]["correct_source_citations"] for r in results) / citations if citations else None,
        "citation_count": citations,
        "quote_validity": sum(r["metrics"]["valid_quote_count"] for r in results) / quotes if quotes else None,
        "quote_count": quotes,
        "answerable_valid_citation_rate": mean(answerable, "all_answer_citations_valid"),
        "scoped_case_compliance": mean(results, "scope_respected"),
        "latency_seconds": distribution(r["latency_seconds"]["total"] for r in attempted),
        "retrieval_latency_seconds": distribution(r["latency_seconds"].get("retrieval") for r in attempted),
        "model_call_latency_seconds": distribution(r["latency_seconds"].get("model_call") for r in attempted),
    }
