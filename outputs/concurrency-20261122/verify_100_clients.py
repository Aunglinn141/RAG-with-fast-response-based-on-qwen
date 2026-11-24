"""Replay the fixed 100-client admission test with two execution slots in an isolated API."""

import argparse
import asyncio
from collections import Counter
from contextlib import AsyncExitStack, contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx

import replay


CLIENT_COUNT = 100
EXECUTION_SLOTS = 2
CONFIG_FIELDS = (
    "chat_model",
    "embedding_model",
    "fast_mode",
    "max_admitted_requests",
    "max_concurrent_requests",
    "concurrency_scope",
)


@contextmanager
def default_api_environment():
    # replay.api sets the isolated storage paths, model names and model server URL.
    # Ignore shell overrides for the application's other document and inference settings.
    prefixes = ("RAG_", "DOCUMENT_", "VLLM_", "OLLAMA_", "CHROMA_", "EMBED_")
    overrides = {
        key: value for key, value in os.environ.items()
        if key.startswith(prefixes) or key == "LOCAL_OLLAMA_URL"
    }
    try:
        for key in overrides:
            del os.environ[key]
        yield
    finally:
        os.environ.update(overrides)


async def read_health(client):
    response = await client.get("/health")
    response.raise_for_status()
    return response.json()["data"]


async def prepare_corpus(client):
    listing = await client.get("/documents")
    listing.raise_for_status()
    assert listing.json()["data"]["count"] == 0
    for name in replay.DATASET["corpus_files"]:
        mime = "text/markdown" if name.endswith(".md") else "text/plain"
        content = (replay.DATA / "corpus" / name).read_bytes()
        response = await client.post("/documents", files={"file": (name, content, mime)})
        assert response.status_code == 201
        assert response.json()["data"]["status"] == "indexed"

    response = await client.post("/warmup")
    response.raise_for_status()
    warmup = response.json()["data"]
    assert warmup["both_models_loaded"] and warmup["indexed_documents"] == 10
    return warmup


async def ask(client, index, start):
    kind = "sse" if index % 2 else "chat"
    case = replay.CASES[(index // 2) % len(replay.CASES)]
    row = await replay.request(client, kind, case, start)
    row["client"] = index + 1
    return row


async def run_batch(clients, monitor):
    start = perf_counter()
    requests = asyncio.gather(*(ask(client, index, start) for index, client in enumerate(clients)))
    observations = []
    overflow = None
    try:
        while not requests.done():
            health = await read_health(monitor)
            running = health["active_heavy_requests"]
            waiting = health["queued_heavy_requests"]
            observations.append({
                "seconds": round(perf_counter() - start, 3),
                "running": running,
                "waiting": waiting,
            })
            if running + waiting == CLIENT_COUNT and overflow is None:
                # The separate monitor connection sends the 101st request at full occupancy.
                response = await monitor.post(
                    "/chat", json={"question": replay.CASES[0]["question"]}
                )
                error = response.json().get("error") or {}
                overflow = {
                    "http": response.status_code,
                    "code": error.get("code"),
                    "retry_after": response.headers.get("retry-after"),
                }
                print("100 admitted", running, waiting, "overflow", overflow, flush=True)
            await asyncio.sleep(0.25 if overflow is not None else 0.025)
        rows = await requests
    finally:
        if not requests.done():
            requests.cancel()
            await asyncio.gather(requests, return_exceptions=True)
    return rows, observations, overflow


def summarize_batch(rows, observations):
    start_times = [row["start_s"] for row in rows]
    finish_times = [row["start_s"] + row["ms"] / 1000 for row in rows]
    wall = max(finish_times)
    valid_routes = Counter(row["kind"] for row in rows if row["outcome"] == "ok")
    return {
        "samples": rows,
        "queue_samples": observations,
        "stats": replay.summarize(rows, wall),
        "peak_admitted": max(point["running"] + point["waiting"] for point in observations),
        "peak_active": max(point["running"] for point in observations),
        "peak_waiting": max(point["waiting"] for point in observations),
        "last_client_seconds": round(wall, 3),
        "first_client_seconds": round(min(finish_times), 3),
        "launch_span_ms": round((max(start_times) - min(start_times)) * 1000, 2),
        "valid_by_route": dict(valid_routes),
    }


async def check(url, result):
    async with AsyncExitStack() as stack:
        monitor = await stack.enter_async_context(
            httpx.AsyncClient(base_url=url, trust_env=False, timeout=180)
        )
        health = await read_health(monitor)
        result["config"] = {key: health[key] for key in CONFIG_FIELDS}
        assert health["max_admitted_requests"] == CLIENT_COUNT
        assert health["max_concurrent_requests"] == EXECUTION_SLOTS
        result["warmup"] = await prepare_corpus(monitor)

        limits = httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=30)
        clients = []
        for _ in range(CLIENT_COUNT):
            client = httpx.AsyncClient(base_url=url, trust_env=False, timeout=600, limits=limits)
            clients.append(await stack.enter_async_context(client))

        # Warm one separate TCP connection per client before releasing the group.
        responses = await asyncio.gather(*(client.get("/health") for client in clients))
        assert all(response.status_code == 200 for response in responses)
        rows, observations, overflow = await run_batch(clients, monitor)

        for _ in range(30):
            health = await read_health(monitor)
            if health["active_heavy_requests"] == health["queued_heavy_requests"] == 0:
                break
            await asyncio.sleep(0.05)
        recovery = await replay.request(monitor, "chat", replay.CASES[0], perf_counter())

    result.update(summarize_batch(rows, observations))
    result.update(
        overflow=overflow,
        final_active=health["active_heavy_requests"],
        final_waiting=health["queued_heavy_requests"],
        recovery_outcome=recovery["outcome"],
    )
    assert result["peak_admitted"] == CLIENT_COUNT
    assert result["peak_active"] == EXECUTION_SLOTS
    assert result["peak_waiting"] == CLIENT_COUNT - EXECUTION_SLOTS
    assert overflow == {"http": 503, "code": "service_busy", "retry_after": "1"}
    assert result["valid_by_route"] == {"sse": 50, "chat": 50}
    assert all(row["http"] == 200 for row in rows)
    assert all(row.get("first_delta_ms", 0) > 0 for row in rows if row["kind"] == "sse")
    assert result["recovery_outcome"] == "ok"
    assert result["final_active"] == result["final_waiting"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-port", type=int, default=11435)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    workdir = replay.ROOT / ".runtime" / "capacity-runs" / uuid4().hex
    workdir.mkdir(parents=True)
    destination = args.out or workdir / "100-clients.json"
    if destination.exists():
        raise FileExistsError(destination)

    result = {
        "started": datetime.now().astimezone().isoformat(),
        "clients": CLIENT_COUNT,
        "workdir": str(workdir),
    }
    try:
        with default_api_environment(), replay.api(
            workdir, EXECUTION_SLOTS, 18593, args.model_port, admitted=CLIENT_COUNT
        ) as url:
            asyncio.run(check(url, result))
        result["passed"] = True
    finally:
        result["finished"] = datetime.now().astimezone().isoformat()
        destination.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(destination, result.get("stats"), flush=True)


if __name__ == "__main__":
    main()
