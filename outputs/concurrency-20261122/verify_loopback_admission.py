"""Measure local HTTP admission using the production middleware and a gated echo worker.

Run from the project root with the project Python. No model process is required.
Only loopback connections to a private, ephemeral port are used.
"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import platform
import socket
import sys
from time import perf_counter

import httpx
import psutil
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_rag.api.limits import HeavyRequestMiddleware, RequestLimiter


HOST = "127.0.0.1"
LEVELS = (100, 256, 512, 1024, 2048)
EXECUTION_SLOTS = 2
OPEN_BATCH = 48
MIB = 1024 ** 2
MAX_PROCESS_RSS = 512 * MIB
MIN_FREE_MEMORY = 1024 * MIB


class EchoWorker:
    def __init__(self, limiter):
        self.limiter = limiter
        self.gate = asyncio.Event()

    async def __call__(self, scope, receive, send):
        if scope["path"] == "/state":
            running, waiting = self.limiter.counts()
            response = JSONResponse({"running": running, "waiting": waiting})
        else:
            body = await Request(scope, receive).body()
            await self.gate.wait()
            response = Response(body, media_type="application/octet-stream")
        await response(scope, receive, send)


class ResourceGuard:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = 0

    def check(self):
        rss = self.process.memory_info().rss
        self.peak = max(self.peak, rss)
        if rss >= MAX_PROCESS_RSS or psutil.virtual_memory().available <= MIN_FREE_MEMORY:
            raise RuntimeError("Stopped at the benchmark's process RSS or free-memory limit")


async def exchange(reader, writer, port, payload):
    request = (
        f"POST /chat HTTP/1.1\r\nHost: {HOST}:{port}\r\nConnection: close\r\n"
        f"Content-Type: application/octet-stream\r\nContent-Length: {len(payload)}\r\n\r\n"
    ).encode("ascii")
    try:
        writer.write(request + payload)
        await writer.drain()
        header = await reader.readuntil(b"\r\n\r\n")
        lines = header.split(b"\r\n")
        status = int(lines[0].split()[1])
        headers = dict(line.lower().split(b":", 1) for line in lines[1:] if line)
        body = await reader.readexactly(int(headers[b"content-length"]))
        return status, status == 200 and body == payload
    finally:
        writer.close()


async def read_state(client):
    response = await client.get("/state")
    response.raise_for_status()
    state = response.json()
    return state["running"], state["waiting"]


async def wave(width, port, worker, monitor):
    sockets = []
    tasks = []
    errors = Counter()
    statuses = Counter()
    guard = ResourceGuard()
    peak = (0, 0)
    verified = 0
    worker.gate.clear()
    started = perf_counter()
    try:
        for offset in range(0, width, OPEN_BATCH):
            guard.check()
            batch = [
                asyncio.wait_for(asyncio.open_connection(HOST, port, limit=16 * 1024), timeout=5)
                for _ in range(min(OPEN_BATCH, width - offset))
            ]
            for peer in await asyncio.gather(*batch, return_exceptions=True):
                if isinstance(peer, BaseException):
                    errors[f"connect_{type(peer).__name__}"] += 1
                else:
                    sockets.append(peer)
        dispatched = perf_counter()
        for index, (reader, writer) in enumerate(sockets):
            # Each connection carries a distinct body so responses cannot be interchanged.
            payload = f"client-{width}-{index}".encode("ascii")
            request = exchange(reader, writer, port, payload)
            tasks.append(asyncio.create_task(asyncio.wait_for(request, timeout=30)))

        deadline = dispatched + 15
        while perf_counter() < deadline:
            state = await read_state(monitor)
            if sum(state) > sum(peak):
                peak = state
            guard.check()
            if sum(peak) >= len(sockets) or all(task.done() for task in tasks):
                break
            await asyncio.sleep(0.01)
        gate_released = perf_counter()
        worker.gate.set()

        for outcome in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(outcome, BaseException):
                errors[f"http_{type(outcome).__name__}"] += 1
            else:
                status, matches = outcome
                statuses[str(status)] += 1
                verified += int(matches)
        finished = perf_counter()
        guard.check()
    finally:
        worker.gate.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for _, writer in sockets:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for _, writer in sockets), return_exceptions=True)

    deadline = perf_counter() + 2
    final = await read_state(monitor)
    while final != (0, 0) and perf_counter() < deadline:
        await asyncio.sleep(0.01)
        final = await read_state(monitor)
    recovery = await monitor.post("/chat", content=b"recovery")
    recovered = recovery.status_code == 200 and recovery.content == b"recovery"
    expected_peak = (EXECUTION_SLOTS, width - EXECUTION_SLOTS)
    return {
        "clients": width,
        "connections_opened": len(sockets),
        "peak_in_flight": sum(peak),
        "peak_running": peak[0],
        "peak_waiting": peak[1],
        "http_responses": sum(statuses.values()),
        "http_status_counts": dict(statuses),
        "verified_200": verified,
        "http_503": statuses["503"],
        "timeouts": sum(count for key, count in errors.items() if key.endswith("TimeoutError")),
        "errors": dict(errors),
        "final_running": final[0],
        "final_waiting": final[1],
        "recovery_http": recovery.status_code,
        "recovery_body_matches": recovered,
        "connection_setup_ms": round((dispatched - started) * 1000, 2),
        "time_to_release_gate_ms": round((gate_released - dispatched) * 1000, 2),
        "time_after_release_to_last_response_ms": round((finished - gate_released) * 1000, 2),
        "peak_sampled_combined_rss_mib": round(guard.peak / MIB, 2),
        "passed": verified == width and peak == expected_peak and final == (0, 0) and recovered,
    }


async def measure(levels, results):
    limiter = RequestLimiter(EXECUTION_SLOTS, capacity=0)
    worker = EchoWorker(limiter)
    app = HeavyRequestMiddleware(worker, limiter=limiter)
    config = uvicorn.Config(
        app, host=HOST, port=0, workers=1, lifespan="off", ws="none",
        access_log=False, log_level="error", timeout_keep_alive=90,
    )
    server = uvicorn.Server(config)
    with socket.socket() as listener:
        listener.bind((HOST, 0))
        port = listener.getsockname()[1]
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            deadline = perf_counter() + 10
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("Server exited during startup")
                if perf_counter() >= deadline:
                    raise TimeoutError("Server startup timeout")
                await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                base_url=f"http://{HOST}:{port}", trust_env=False, timeout=5
            ) as monitor:
                for width in levels:
                    result = await wave(width, port, worker, monitor)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    if not result["passed"]:
                        break
        finally:
            worker.gate.set()
            server.should_exit = True
            await server_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-clients", choices=LEVELS, type=int, default=LEVELS[-1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    started = datetime.now().astimezone()
    name = f"admission-loopback-{started:%Y%m%d-%H%M%S}.json"
    destination = args.out or Path(__file__).with_name(name)
    levels = [width for width in LEVELS if width <= args.max_clients]
    record = {
        "started": started.isoformat(),
        "transport": "HTTP/1.1 over independent loopback TCP connections, batch preconnect",
        "workload": "production HeavyRequestMiddleware/RequestLimiter, gated echo, no model",
        "scope": "client and single-worker server in one process; RSS sampled for both",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uvicorn": uvicorn.__version__,
        "capacity": 0,
        "execution_slots": EXECUTION_SLOTS,
        "preconnect_batch_size": OPEN_BATCH,
        "process_rss_limit_mib": MAX_PROCESS_RSS // MIB,
        "min_available_memory_mib": MIN_FREE_MEMORY // MIB,
        "levels": levels,
        "results": [],
        "passed": False,
    }
    with destination.open("x", encoding="utf-8") as output:
        try:
            asyncio.run(measure(levels, record["results"]))
            record["passed"] = len(record["results"]) == len(levels) and all(
                row["passed"] for row in record["results"]
            )
        finally:
            record["finished"] = datetime.now().astimezone().isoformat()
            json.dump(record, output, indent=2)
            print(destination, flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
