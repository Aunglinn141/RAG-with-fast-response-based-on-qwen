"""Measure actual visible answer and completion latency over the RAG SSE API."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request

from local_rag.cli.benchmark_latency import LocalAPI, distribution


def measure(api, question, *, model=None):
    payload = {"question": question, **({"model": model} if model else {})}
    request = Request(api.url + "/chat/stream", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = perf_counter()
    first_answer_ms, event, data_lines, total_bytes = None, "", [], 0
    resets = 0
    with api.opener.open(request, timeout=180) as response:
        while True:
            line = response.readline(65537)
            total_bytes += len(line)
            if not line:
                raise RuntimeError("SSE ended without a final result")
            if len(line) > 65536 or total_bytes > 4 * 1024 * 1024 or perf_counter() - started > 180:
                raise RuntimeError("SSE exceeded benchmark time or size limit")
            line = line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
            elif not line and data_lines:
                packet = json.loads("\n".join(data_lines))
                data_lines = []
                if packet.get("success") is not True or event == "error":
                    raise RuntimeError(str(packet.get("error", "SSE request failed")))
                data = packet["data"]
                elapsed = (perf_counter() - started) * 1000
                if event == "reset":
                    first_answer_ms = None  # A discarded draft does not meet the target.
                    resets += 1
                elif event == "delta" and data.get("text", "").strip() and first_answer_ms is None:
                    first_answer_ms = elapsed  # Stage events / headers do not count.
                elif event == "result":
                    answered = data.get("status") == "answered" and bool(data.get("answer", "").strip())
                    return {"question": question, "first_answer_ms": round(first_answer_ms if first_answer_ms is not None else elapsed, 3),
                            "complete_ms": round(elapsed, 3), "status": data.get("status"),
                            "eligible": answered and data.get("interaction") != "greeting",
                            "answer": data.get("answer"), "resets": resets,
                            "timings": data.get("timings", {})}
                event = ""


def summarize(samples, target_ms, metric):
    valid = [sample for sample in samples if sample.get("eligible")]
    return {"metric": metric, "target_ms": target_ms, "samples": len(samples),
            "valid_answers": len(valid), "first_answer": distribution([s["first_answer_ms"] for s in valid]),
            "complete": distribution([s["complete_ms"] for s in valid]),
            "passed": bool(samples) and len(valid) == len(samples)
                      and all(s[metric] <= target_ms for s in valid)}


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", action="append", required=True, help="Repeat for several real knowledge-base questions")
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=1000)
    parser.add_argument("--metric", choices=["first_answer_ms", "complete_ms"], default="first_answer_ms")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100 or not 0 < args.target_ms < 180000:
        parser.error("Require repeats 1..100 and target-ms in (0, 180000)")
    if any(not q.strip() or len(q) > 2000 for q in args.question):
        parser.error("Questions must contain 1..2000 characters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new artifact; never overwrite a previous baseline.
    try:
        report_file = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error("Output exists; choose a new report path")
    report = {"samples": [], "measurement": "HTTP request to first visible answer / final result, including retrieval and scope check",
              "limitations": ["Warm model tests do not measure startup. Review answer correctness separately.",
                              "Small samples and repeated questions do not establish a production SLA."],
              "warmup_enabled": not args.no_warmup, "passed": False}
    code = 1
    try:
        api = LocalAPI(args.api_url)
        report["health"], _ = api.request("/health")
        if not args.no_warmup:
            report["warmup"], _ = api.request("/warmup", {"model": args.model} if args.model else {})
        for repeat in range(args.repeats):
            for question in args.question:
                sample = measure(api, question, model=args.model)
                sample["repeat"] = repeat
                report["samples"].append(sample)
                print(json.dumps({k: sample[k] for k in ("repeat", "first_answer_ms", "complete_ms", "status")}), flush=True)
        report["summary"] = summarize(report["samples"], args.target_ms, args.metric)
        report["passed"] = report["summary"]["passed"]
        print(json.dumps(report["summary"], indent=2))
        code = 0 if report["passed"] else 2
    except Exception as exc:
        report["error"] = str(exc)
        print("Benchmark failed: " + str(exc))
    finally:
        with report_file:
            json.dump(report, report_file, ensure_ascii=False, indent=2)
    return code


if __name__ == "__main__":
    raise SystemExit(cli())
