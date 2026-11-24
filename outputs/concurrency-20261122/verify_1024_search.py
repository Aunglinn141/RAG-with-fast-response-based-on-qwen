"""Verify 1,024 simultaneous, completed hybrid searches on a fresh local RAG API.

Uses the existing project models, a private Ollama, a fresh index and independent loopback
connections. --after waits until a preceding benchmark's JSON contains a finished timestamp.
"""

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from uuid import uuid4

import httpx
import psutil

import replay
from verify_100_clients import CONFIG_FIELDS, default_api_environment, prepare_corpus, read_health
from verify_elastic_clients import observe, wait_for_count


HOST = "127.0.0.1"
API_PORT = 18603
MODEL_PORT = 18604
CLIENTS = 1024
BATCH_TIMEOUT = 600
OPEN_BATCH = 32


def wait_for_benchmark(marker):
    if marker is None:
        return
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("finished"):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for {marker}")


@contextmanager
def private_ollama(work):
    with socket.socket() as probe:
        probe.bind((HOST, MODEL_PORT))
    models_dir = replay.ROOT / ".runtime" / "ollama" / "models"
    executable = shutil.which("ollama")
    if not executable or not models_dir.is_dir():
        raise RuntimeError("Project Ollama executable or model directory is unavailable")
    env = dict(
        os.environ, OLLAMA_HOST=f"{HOST}:{MODEL_PORT}", OLLAMA_MODELS=str(models_dir),
        OLLAMA_NO_CLOUD="1", OLLAMA_CONTEXT_LENGTH="4096", OLLAMA_MAX_LOADED_MODELS="2",
        OLLAMA_KEEP_ALIVE="-1", OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_QUEUE="1",
    )
    with (work / "ollama.log").open("wb") as log:
        proc = subprocess.Popen(
            [executable, "serve"], env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 30
            with httpx.Client(base_url=f"http://{HOST}:{MODEL_PORT}", trust_env=False, timeout=2) as probe:
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Private Ollama stopped; see {log.name}")
                    try:
                        response = probe.get("/api/tags")
                        response.raise_for_status()
                        break
                    except httpx.TransportError:
                        time.sleep(0.2)
                else:
                    raise TimeoutError("Private Ollama startup timed out")
            yield
        finally:
            if proc.poll() is None:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                proc.terminate()
                proc.wait(timeout=10)
                _, alive = psutil.wait_procs(children, timeout=3)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=5)


def wire_request(case):
    payload = json.dumps({"query": case["question"], "top_k": 4, "mode": "hybrid"}).encode("utf-8")
    header = (
        f"POST /search HTTP/1.1\r\nHost: {HOST}:{API_PORT}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    return header + payload


async def read_response(reader, writer, index, case, sent, start):
    row = {"client": index + 1, "case": case["id"], "kind": "search", "start_s": sent - start}
    try:
        header = await reader.readuntil(b"\r\n\r\n")
        lines = header.split(b"\r\n")
        row["http"] = int(lines[0].split()[1])
        headers = dict(line.lower().split(b":", 1) for line in lines[1:] if line)
        body = await reader.readexactly(int(headers[b"content-length"]))
        received = time.perf_counter()
        envelope = json.loads(body)
        replay.collect(envelope, "search", case, row)
        if row["outcome"] == "ok":
            data = envelope["data"]
            identity = data.get("model") or {}
            row["model"] = identity.get("name")
            row["model_version"] = identity.get("version")
            if (
                row["http"] != 200 or data.get("query") != case["question"]
                or row["model"] != "bge-m3:latest" or not row["model_version"]
            ):
                row["outcome"] = "unexpected_response"
    except (OSError, EOFError, asyncio.TimeoutError, ValueError, KeyError) as exc:
        received = time.perf_counter()
        row["outcome"] = type(exc).__name__
    finally:
        # The response specifies Connection: close. Release each TCP peer as its body
        # finishes, including early successes that would otherwise sit idle for minutes.
        writer.close()
    row["ms"] = (received - sent) * 1000
    return row


async def search_wave(monitor):
    peers = []
    tasks = []
    peak = {"active": 0, "waiting": 0}
    completed_mark = 128
    wire = [wire_request(case) for case in replay.CASES]
    setup_start = time.perf_counter()
    try:
        for _ in range(0, CLIENTS, OPEN_BATCH):
            batch = await asyncio.gather(
                *(asyncio.wait_for(asyncio.open_connection(HOST, API_PORT), 5) for _ in range(OPEN_BATCH)),
                return_exceptions=True,
            )
            peers.extend(peer for peer in batch if isinstance(peer, tuple))
            for peer in batch:
                if isinstance(peer, BaseException):
                    raise peer
        ports = {writer.get_extra_info("sockname")[1] for _, writer in peers}
        assert len(ports) == CLIENTS
        start = time.perf_counter()
        for index, (reader, writer) in enumerate(peers):
            case_index = index % len(replay.CASES)
            sent = time.perf_counter()
            writer.write(wire[case_index])
            response = read_response(reader, writer, index, replay.CASES[case_index], sent, start)
            tasks.append(asyncio.create_task(response))
        dispatch_done = time.perf_counter()
        deadline = start + BATCH_TIMEOUT
        while not all(task.done() for task in tasks):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Search wave exceeded its time budget")
            current = await observe(monitor)
            if sum(current.values()) > sum(peak.values()):
                peak = current
            completed = sum(task.done() for task in tasks)
            if completed >= completed_mark:
                print(f"Search {completed}/{CLIENTS}; peak {peak}", flush=True)
                completed_mark += 128
            await asyncio.sleep(0.01 if sum(peak.values()) < CLIENTS - 8 else 0.15)
        rows = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for _, writer in peers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for _, writer in peers), return_exceptions=True)

    finished = [row["start_s"] + row["ms"] / 1000 for row in rows]
    seconds = max(finished)
    return {
        "clients": CLIENTS,
        "independent_tcp_connections": len(ports),
        "socket_setup_s": round(start - setup_start, 3),
        "dispatch_ms": round((dispatch_done - start) * 1000, 3),
        "peak": peak,
        "peak_admitted": sum(peak.values()),
        "stats": replay.summarize(rows, seconds),
        "first_complete_s": round(min(finished), 3),
        "all_complete_s": round(seconds, 3),
        "verified_by_case": dict(Counter(row["case"] for row in rows if row["outcome"] == "ok")),
        "samples": rows,
    }


def embedding_http_calls(work):
    log = (work / "ollama.log").read_bytes()
    return len(re.findall(rb'POST\s+"/api/(?:embed|embeddings)"', log))


async def measure(url, result, work):
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=60) as monitor:
        health = await read_health(monitor)
        result["config"] = {key: health[key] for key in CONFIG_FIELDS}
        assert health["max_admitted_requests"] == 0 and health["max_concurrent_requests"] == 2
        result["warmup"] = await prepare_corpus(monitor)
        before = embedding_http_calls(work)
        result["batch"] = batch = await search_wave(monitor)
        batch["ollama_embedding_http_calls"] = embedding_http_calls(work) - before
        result["final_counts"] = await wait_for_count(monitor, 0, 10)
        recovery = await replay.request(monitor, "search", replay.CASES[0], time.perf_counter())
        result["recovery"] = recovery
        result["passed"] = (
            batch["stats"]["ok"] == CLIENTS and batch["stats"]["http"] == {"200": CLIENTS}
            and batch["ollama_embedding_http_calls"] >= CLIENTS
            and recovery["http"] == 200 and recovery["outcome"] == "ok"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, help="Wait for this preceding benchmark JSON to finish")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stamp = datetime.now().astimezone()
    destination = args.out or Path(__file__).with_name(f"search-real-1024-{stamp:%Y%m%d-%H%M%S}.json")
    if destination.exists():
        raise FileExistsError(destination)
    work = replay.ROOT / ".runtime" / "capacity-runs" / uuid4().hex
    work.mkdir(parents=True)
    print(f"Queueing run: {destination}; private working data: {work}", flush=True)
    result = {
        "queued": stamp.isoformat(), "workdir": str(work), "passed": False,
        "api_port": API_PORT, "model_port": MODEL_PORT, "api_workers": 1,
        "workload": "hybrid search, top_k=4, real BGE-M3 and private Chroma, no answer generation",
        "expected_sources": {case["id"]: case["expected_source_files"] for case in replay.CASES},
    }
    try:
        wait_for_benchmark(args.after)
        result["started"] = datetime.now().astimezone().isoformat()
        with default_api_environment(), private_ollama(work):
            with replay.api(work, None, API_PORT, MODEL_PORT) as url:
                asyncio.run(measure(url, result, work))
    finally:
        result["finished"] = datetime.now().astimezone().isoformat()
        with destination.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, separators=(",", ":"))
        print(destination, result.get("batch", {}).get("stats"), result["passed"], flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
