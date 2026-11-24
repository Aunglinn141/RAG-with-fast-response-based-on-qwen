"""Measure two fixed workloads on a private loopback API with real warmed models."""

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx

import replay
from verify_100_clients import ask, default_api_environment, prepare_corpus, read_health, CONFIG_FIELDS


HOST = "127.0.0.1"
API_PORT = 18597
ADMISSION_PROBE = 1024
FIRST_GROUP = 156
SECOND_GROUP = 100


async def observe(client):
    health = await read_health(client)
    return {"active": health["active_heavy_requests"], "waiting": health["queued_heavy_requests"]}


async def wait_for_count(client, target, timeout):
    deadline = perf_counter() + timeout
    while True:
        sample = await observe(client)
        if sum(sample.values()) == target:
            return sample
        if perf_counter() >= deadline:
            raise RuntimeError(f"Expected {target} requests in the API; last observed {sample}")
        await asyncio.sleep(0.02)


async def admission_probe(monitor):
    # Fill real /chat using small valid requests. Close the burst once it is observed.
    # This measures admission and disconnect recovery separately from answer timing.
    peers = []
    transport_start = perf_counter()
    try:
        for _ in range(ADMISSION_PROBE // 32):
            batch = await asyncio.gather(
                *(asyncio.wait_for(asyncio.open_connection(HOST, API_PORT), 5) for _ in range(32)),
                return_exceptions=True,
            )
            peers.extend(pair for pair in batch if isinstance(pair, tuple))
            for result in batch:
                if isinstance(result, BaseException):
                    raise result
        ports = {writer.get_extra_info("sockname")[1] for _, writer in peers}
        assert len(ports) == ADMISSION_PROBE
        body = json.dumps({"question": replay.CASES[0]["question"]}).encode("utf-8")
        headers = (
            f"POST /chat HTTP/1.1\r\nHost: {HOST}:{API_PORT}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
        start = perf_counter()
        for _, writer in peers:
            writer.write(headers + body)
        await asyncio.gather(*(writer.drain() for _, writer in peers))
        snapshot = await wait_for_count(monitor, ADMISSION_PROBE, 3)
        reached = perf_counter()
        print("Live API, before controlled disconnect:", snapshot, flush=True)
    finally:
        for _, writer in peers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for _, writer in peers), return_exceptions=True)
    drained = await wait_for_count(monitor, 0, 60)
    recovery = await replay.request(monitor, "chat", replay.CASES[0], perf_counter())
    assert recovery["outcome"] == "ok"
    return {
        "clients": len(ports), "socket_setup_s": round(start - transport_start, 3),
        "peak": snapshot, "time_to_verified_peak_s": round(reached - start, 3),
        "probe_behavior": "Clients close after simultaneous admission; no completion claim",
        "drained": drained, "recovery": recovery["outcome"],
    }


async def answers(url, monitor):
    async with AsyncExitStack() as stack:
        clients = []
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=60)
        for _ in range(FIRST_GROUP + SECOND_GROUP):
            client = httpx.AsyncClient(
                base_url=url, trust_env=False, limits=limits, timeout=httpx.Timeout(900, connect=10)
            )
            clients.append(await stack.enter_async_context(client))
        for offset in range(0, len(clients), 32):
            probes = await asyncio.gather(*(c.get("/health") for c in clients[offset:offset + 32]))
            assert all(r.status_code == 200 for r in probes)
        start = perf_counter()
        pending = []
        samples = []

        async def ask_once(i):
            client = clients[i]
            row = await ask(client, i, start)
            # Each warmed connection serves one measured answer. Close it after timing and validation.
            await client.aclose()
            return row

        try:
            for i in range(FIRST_GROUP):
                pending.append(asyncio.create_task(ask_once(i)))
            first = await wait_for_count(monitor, FIRST_GROUP, 3)
            samples.append(dict(elapsed=round(perf_counter()-start, 3), **first))
            for i in range(FIRST_GROUP, len(clients)):
                pending.append(asyncio.create_task(ask_once(i)))
            peak = await wait_for_count(monitor, len(clients), 3)
            print("Live API, extra 100 submitted:", first, "=>", peak, flush=True)
            samples.append(dict(elapsed=round(perf_counter()-start, 3), **peak))
            next_progress = 50
            while not all(t.done() for t in pending):
                sample = await observe(monitor)
                samples.append(dict(elapsed=round(perf_counter() - start, 3), **sample))
                completed = sum(t.done() for t in pending)
                if completed >= next_progress:
                    elapsed = round(perf_counter() - start, 1)
                    print("Completed", completed, "/", len(clients), "elapsed", elapsed, flush=True)
                    next_progress += 50
                await asyncio.sleep(0.4)
            rows = await asyncio.gather(*pending)
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    finish_times = [row["start_s"] + row["ms"] / 1000 for row in rows]
    end = max(finish_times)
    first_end = min(finish_times)
    launch_times = [row["start_s"] for row in rows]
    return {
        "first_group": FIRST_GROUP, "second_group": SECOND_GROUP,
        "before_extra_100": first, "peak": peak, "stats": replay.summarize(rows, end),
        "first_complete_s": round(first_end, 3), "all_complete_s": round(end, 3),
        "extra_group_successes": sum(r["outcome"] == "ok" for r in rows[FIRST_GROUP:]),
        "max_launch_span_ms": round(1000 * (max(launch_times) - min(launch_times)), 2),
        "samples": rows, "queue_samples": samples,
    }


async def measure(url, result, persist):
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=180) as monitor:
        health = await read_health(monitor)
        result["config"] = {key: health[key] for key in CONFIG_FIELDS}
        assert health["max_admitted_requests"] == 0 and health["max_concurrent_requests"] == 2
        result["warmup"] = await prepare_corpus(monitor)
        result["admission_probe"] = await admission_probe(monitor)
        persist()
        result["answer_batch"] = await answers(url, monitor)
        result["final_counts"] = await wait_for_count(monitor, 0, 10)
        recovery = await replay.request(monitor, "chat", replay.CASES[0], perf_counter())
        result["recovery"] = recovery["outcome"]
        rows = result["answer_batch"]["samples"]
        valid = all(
            r["http"] == 200 and r["outcome"] == "ok"
            and (r["kind"] != "sse" or r.get("first_delta_ms", 0) > 0)
            for r in rows
        )
        result["passed"] = len(rows) == FIRST_GROUP + SECOND_GROUP and valid and recovery["outcome"] == "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-port", type=int, default=11435)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    work = replay.ROOT / ".runtime/capacity-runs" / uuid4().hex
    work.mkdir(parents=True)
    dest = args.out or work / "elastic-clients.json"
    if dest.exists():
        raise FileExistsError(dest)
    result = {"started": datetime.now().astimezone().isoformat(), "workdir": str(work)}

    def persist():
        dest.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    try:
        with default_api_environment(), replay.api(work, None, API_PORT, args.model_port) as url:
            asyncio.run(measure(url, result, persist))
    finally:
        result["finished"] = datetime.now().astimezone().isoformat()
        persist()
        print(dest, result.get("passed"), flush=True)
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
