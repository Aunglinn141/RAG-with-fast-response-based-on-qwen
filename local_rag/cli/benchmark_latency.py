"""Measure existing indexes over FastAPI; never upload or rebuild a corpus."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from local_rag.clients.ollama import _NoRedirect
from local_rag.core.config import RUNTIME_DIR

QUESTIONS = ["关羽和刘备是什么关系？", "孙悟空是谁？", "林冲为什么上梁山？", "贾宝玉和林黛玉是什么关系？"]


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * .95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"samples": len(values), "p50_ms": round(median(values), 3),
            "p95_ms": round(p95, 3), "max_ms": round(values[-1], 3)}


class LocalAPI:
    def __init__(self, url):
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("Use a loopback HTTP API URL, e.g. http://127.0.0.1:8000")
        host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
        self.url = "http://" + host + ":" + str(parsed.port or 8000)
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, path, data=None):
        req = Request(self.url + path, data=None if data is None else json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
        start = perf_counter()
        with self.opener.open(req, timeout=600) as response:
            payload = json.load(response)
        if payload.get("success") is not True:
            raise RuntimeError(str(payload.get("error")))
        return payload["data"], round((perf_counter() - start) * 1000, 3)


def benchmark(api, *, repeats, model=None, include_chat=False, save=lambda report: None):
    before, _ = api.request("/documents")
    if not any(r["status"] == "indexed" for r in before["documents"]):
        raise ValueError("Index documents before benchmarking; this command never indexes them.")
    health, _ = api.request("/health")
    report = {"repeats": repeats, "target_ms": 1000, "indexed_documents_before": before,
              "startup_warmup": health.get("warmup"), "search": [], "chat": [],
              "chat_backend": health.get("chat_backend", "ollama"),
              "measurement": "Each search embeds its query and reads existing Chroma vectors. No query-result cache.",
              "model": model or health["chat_model"]}
    try:
        report["warmup"], _ = api.request("/warmup", {"model": model} if model else {})
        for repeat in range(repeats):
            for mode in ("semantic", "hybrid"):
                for question in QUESTIONS:
                    result, wall = api.request("/search", {"query": question, "mode": mode, "top_k": 4})
                    if not result["matches"] or "embedding_chroma_ms" not in result.get("timings", {}):
                        raise RuntimeError("Search has no matches or timing data; verify the API and index.")
                    report["search"].append({"repeat": repeat, "mode": mode, "question": question,
                                             "http_ms": wall, "timings": result["timings"]})
            if include_chat:
                question = QUESTIONS[repeat % len(QUESTIONS)]
                result, wall = api.request("/chat", {"question": question, "model": report["model"]})
                report["chat"].append({"repeat": repeat, "question": question, "http_ms": wall,
                                       "status": result["status"], "timings": result["timings"]})
                result, wall = api.request("/search", {"query": question, "top_k": 4})
                report["search"].append({"repeat": repeat, "mode": "after_chat", "question": question,
                                         "http_ms": wall, "timings": result["timings"]})
            save(report)
        report["summary"] = {mode: {
            "embedding_chroma": distribution([s["timings"]["embedding_chroma_ms"] for s in report["search"] if s["mode"] == mode]),
            "retrieval": distribution([s["timings"]["retrieval_ms"] for s in report["search"] if s["mode"] == mode]),
            "http": distribution([s["http_ms"] for s in report["search"] if s["mode"] == mode])
        } for mode in ("semantic", "hybrid", "after_chat")}
        report["generation"] = distribution([s["timings"]["generation_ms"] for s in report["chat"]])
        report["all_retrieval_samples_within_target"] = all(s["timings"]["embedding_chroma_ms"] <= 1000 for s in report["search"])
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        try:
            after, _ = api.request("/documents")
            report["index_records_unchanged"] = before == after
        finally:
            save(report)
    return report


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chat", action="store_true", help="Also measure LLM stages and search immediately after each chat")
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "reports" / "latency.json")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be 1..20")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(report):
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        report = benchmark(LocalAPI(args.api_url), repeats=args.repeats, model=args.model,
                           include_chat=args.chat, save=save)
    except Exception as exc:
        print("Benchmark failed: " + str(exc))
        return 1
    print(json.dumps({key: report[key] for key in ("summary", "generation", "all_retrieval_samples_within_target",
                                                  "index_records_unchanged")}, ensure_ascii=True, indent=2))
    print("Report: " + str(args.output.resolve()))
    return 0 if report["index_records_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(cli())
