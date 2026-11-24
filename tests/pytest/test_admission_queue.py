"""Exercise admission at ASGI level with real request bodies and controlled task completion."""

import asyncio
from collections import defaultdict, deque

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import Response

from local_rag.api.limits import HeavyRequestMiddleware, RequestLimiter


class ControlledWorker:
    def __init__(self):
        self.entered = asyncio.Queue()
        self.release = defaultdict(asyncio.Event)

    async def __call__(self, scope, receive, send):
        if scope["path"] == "/health":
            await Response("ok")(scope, receive, send)
            return
        body = await Request(scope, receive).body()
        self.entered.put_nowait(body)
        await self.release[body].wait()
        await Response(body)(scope, receive, send)

    async def next_body(self):
        return await asyncio.wait_for(self.entered.get(), 3)


async def request(app, *parts, path="/chat", disconnect=None, on_read=None):
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "GET" if path == "/health" else "POST", "path": path,
        "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "scheme": "http", "http_version": "1.1", "headers": [],
        "client": ("test", 123), "server": ("test", 80),
    }
    body = deque(parts or (b"",))
    closed = disconnect if disconnect is not None else asyncio.Event()
    messages = []

    async def receive():
        if body:
            part = body.popleft()
            if on_read is not None:
                on_read(part)
            return {"type": "http.request", "body": part, "more_body": bool(body)}
        await closed.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    if messages:
        start, *content = messages
        assert start["type"] == "http.response.start"
        return httpx.Response(start["status"], headers=start["headers"],
                              content=b"".join(m["body"] for m in content))


async def wait_counts(limiter, active, waiting):
    async def ready():
        while limiter.counts() != (active, waiting):
            await asyncio.sleep(0)
    await asyncio.wait_for(ready(), 3)


def assert_busy(response):
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["code"] == "service_busy"


@pytest.mark.parametrize("capacity", [100, None], ids=["hundred-admitted", "legacy-fail-fast"])
def test_total_capacity_is_shared_and_waiters_run_in_arrival_order(capacity):
    async def scenario():
        limiter = RequestLimiter(2, capacity=capacity) if capacity is not None else RequestLimiter(2)
        worker = ControlledWorker()
        middleware = HeavyRequestMiddleware(worker, limiter=limiter)
        bodies = [str(i).encode() for i in range(capacity or 2)]
        requests = []
        for i, body in enumerate(bodies):
            task = asyncio.create_task(request(middleware, body, path="/search" if i % 2 else "/chat"))
            requests.append(task)
            if i < 2:
                assert await worker.next_body() == body
            else:
                await wait_counts(limiter, 2, i - 1)
        assert_busy(await asyncio.wait_for(request(middleware, b"overflow", path="/chat/stream"), 3))
        assert (await asyncio.wait_for(request(middleware, path="/health"), 3)).status_code == 200
        # Keep one executor occupied so every handoff to a waiter is individually observable.
        previous = bodies[0]
        for expected in bodies[2:]:
            worker.release[previous].set()
            assert await worker.next_body() == expected
            previous = expected
        worker.release[previous].set()
        worker.release[bodies[1]].set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), 5)
        assert all(r.status_code == 200 for r in responses)
        assert [r.content for r in responses] == bodies
        await wait_counts(limiter, 0, 0)
        await limiter.wait_for_idle()

    asyncio.run(scenario())


def test_uncapped_admission_grows_past_100_and_1024_then_drains():
    async def scenario():
        limiter = RequestLimiter(2, capacity=0)
        worker = ControlledWorker()
        middleware = HeavyRequestMiddleware(worker, limiter=limiter)
        bodies = [str(i).encode() for i in range(1027)]
        requests = []
        for total in (100, 101, 1024, len(bodies)):
            for body in bodies[len(requests):total]:
                requests.append(asyncio.create_task(request(middleware, body)))
            await wait_counts(limiter, 2, total - 2)
        assert await worker.next_body() == bodies[0]
        assert await worker.next_body() == bodies[1]
        assert worker.entered.empty()
        assert all(not task.done() for task in requests)
        assert (await asyncio.wait_for(request(middleware, path="/health"), 3)).status_code == 200
        for body in bodies:
            worker.release[body].set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), 15)
        assert all(r.status_code == 200 for r in responses)
        assert [r.content for r in responses] == bodies
        await wait_counts(limiter, 0, 0)
        await limiter.wait_for_idle()

    asyncio.run(scenario())


@pytest.mark.parametrize("capacity", [2, 0], ids=["finite", "uncapped"])
@pytest.mark.parametrize("peer_disconnected", [True, False], ids=["http-disconnect", "caller-cancelled"])
def test_waiter_disconnect_or_cancellation_immediately_frees_pending_capacity(peer_disconnected, capacity):
    async def scenario():
        limiter = RequestLimiter(1, capacity=capacity)
        worker = ControlledWorker()
        middleware = HeavyRequestMiddleware(worker, limiter=limiter)
        first = asyncio.create_task(request(middleware, b"first"))
        assert await worker.next_body() == b"first"
        closed = asyncio.Event()
        abandoned = asyncio.create_task(request(middleware, b"aban", b"doned", path="/chat/stream", disconnect=closed))
        await wait_counts(limiter, 1, 1)
        if peer_disconnected:
            closed.set()
            await asyncio.wait_for(abandoned, 3)
        else:
            abandoned.cancel()
            with pytest.raises(asyncio.CancelledError):
                await abandoned
        await wait_counts(limiter, 1, 0)
        replacement = asyncio.create_task(request(middleware, b"sec", b"on", b"d"))
        await wait_counts(limiter, 1, 1)
        worker.release[b"first"].set()
        assert await worker.next_body() == b"second"
        worker.release[b"second"].set()
        finished = await asyncio.wait_for(asyncio.gather(first, replacement), 3)
        assert [resp.content for resp in finished] == [b"first", b"second"]
        await wait_counts(limiter, 0, 0)
        await limiter.wait_for_idle()

    asyncio.run(scenario())


@pytest.mark.parametrize("capacity,queued", [(2, 1), (0, 8)], ids=["finite", "uncapped"])
def test_shutdown_releases_waiters_and_waits_for_detached_active_worker(capacity, queued):
    async def scenario():
        limiter = RequestLimiter(1, capacity=capacity)
        worker = ControlledWorker()
        middleware = HeavyRequestMiddleware(worker, limiter=limiter)
        active = asyncio.create_task(request(middleware, b"active"))
        assert await worker.next_body() == b"active"
        waiting = [asyncio.create_task(request(middleware, b"waiting")) for _ in range(queued)]
        await wait_counts(limiter, 1, queued)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert limiter.counts() == (1, queued)
        shutdown = asyncio.create_task(limiter.wait_for_idle())
        for response in await asyncio.wait_for(asyncio.gather(*waiting), 3):
            assert_busy(response)
        assert not shutdown.done()
        assert_busy(await asyncio.wait_for(request(middleware, b"closing"), 3))
        worker.release[b"active"].set()
        await asyncio.wait_for(shutdown, 3)
        assert limiter.counts() == (0, 0)
        assert worker.entered.empty()

    asyncio.run(scenario())


def test_large_queued_upload_applies_backpressure_and_replays_the_complete_body():
    async def scenario():
        limiter = RequestLimiter(1, capacity=2)
        worker = ControlledWorker()
        middleware = HeavyRequestMiddleware(worker, limiter=limiter)
        first = asyncio.create_task(request(middleware, b"first"))
        assert await worker.next_body() == b"first"
        chunks = [bytes([i]) * (32 * 1024) for i in range(12)]
        complete_body = b"".join(chunks)
        reads = []
        drained = asyncio.Event()

        def received(chunk):
            reads.append(chunk)
            if len(reads) == len(chunks):
                drained.set()

        upload = asyncio.create_task(request(middleware, *chunks, path="/documents", on_read=received))
        await wait_counts(limiter, 1, 1)
        # Every ASGI body frame is immediately available. A queued request must
        # leave a large body on the transport until it can be processed.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(drained.wait(), 0.1)
        assert not upload.done()
        worker.release[b"first"].set()
        assert await worker.next_body() == complete_body
        worker.release[complete_body].set()
        assert (await asyncio.wait_for(upload, 3)).content == complete_body
        assert reads == chunks
        await first
        await limiter.wait_for_idle()

    asyncio.run(scenario())


def test_cancellation_during_slot_handoff_allows_the_next_waiter_to_run():
    async def scenario():
        limiter = RequestLimiter(1, capacity=3)
        worker = ControlledWorker()
        middleware = HeavyRequestMiddleware(worker, limiter=limiter)
        assert limiter.try_acquire()
        cancelled = asyncio.create_task(request(middleware, b"cancelled"))
        await wait_counts(limiter, 1, 1)
        waiting = asyncio.create_task(request(middleware, b"survivor"))
        await wait_counts(limiter, 1, 2)
        # Cancel after the permit is granted but before the queued caller resumes.
        limiter.release()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert await worker.next_body() == b"survivor"
        worker.release[b"survivor"].set()
        assert (await asyncio.wait_for(waiting, 3)).status_code == 200
        await wait_counts(limiter, 0, 0)
        await limiter.wait_for_idle()

    asyncio.run(scenario())
