"""Bound heavy requests before reading uploads or starting model/storage work."""

import asyncio
from collections import deque
import re

from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from local_rag.api.schemas import failure


def is_heavy_request(scope: Scope) -> bool:
    if scope["type"] != "http":
        return False
    path = scope["path"].rstrip("/")
    method = scope["method"]
    if path == "/artifacts" or path.startswith("/artifacts/"):
        return method in {"POST", "PUT", "DELETE"}
    if method == "POST":
        return path in {"/documents", "/search", "/chat", "/chat/stream", "/ask", "/warmup"} or bool(
            re.fullmatch(r"/documents/[^/]+/reindex", path)
        )
    return method == "DELETE" and bool(re.fullmatch(r"/documents/[^/]+", path))


class _QueuedReceive:
    """Watch for disconnect while queued without draining a streaming upload."""

    def __init__(self, receive: Receive):
        self._receive = receive
        self._buffer: deque[Message] = deque()
        self._pending: asyncio.Future[Message] | None = None

    async def wait(self, turn: asyncio.Future[bool]) -> bool:
        buffered_bytes = 0
        while not turn.done():
            self._pending = asyncio.ensure_future(self._receive())
            await asyncio.wait((turn, self._pending), return_when=asyncio.FIRST_COMPLETED)
            if not self._pending.done():
                # Ownership of this receive passes to the admitted handler.
                break
            message = self._pending.result()
            self._pending = None
            if message["type"] == "http.disconnect":
                raise ClientDisconnect()
            self._buffer.append(message)
            buffered_bytes += len(message.get("body", b""))
            # Leave large bodies on the transport after at most one ASGI frame
            # beyond this watermark. For a completed body, the next event is
            # the disconnect signal; typical JSON clients can be removed promptly.
            if message.get("more_body") and (buffered_bytes >= 64 * 1024 or len(self._buffer) >= 16):
                break
        return await asyncio.shield(turn)

    async def __call__(self) -> Message:
        if self._buffer:
            return self._buffer.popleft()
        if self._pending is not None:
            pending, self._pending = self._pending, None
            return await pending
        return await self._receive()

    async def close(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            await asyncio.gather(self._pending, return_exceptions=True)
            self._pending = None

    async def serve(self, app: ASGIApp, scope: Scope, send: Send) -> None:
        try:
            await app(scope, self, send)
        finally:
            await self.close()


class RequestLimiter:
    """Share slots in FIFO order. Zero capacity leaves the admission queue uncapped."""

    def __init__(self, limit: int, *, capacity: int | None = None):
        self._limit = limit
        self._capacity = limit if capacity is None else capacity
        if type(limit) is not int or limit < 1:
            raise ValueError("The execution limit must be a positive integer.")
        if type(self._capacity) is not int or (self._capacity != 0 and self._capacity < limit):
            raise ValueError("Admission capacity must be zero or cover all execution slots.")
        self._running = 0
        self._waiters: deque[asyncio.Future[bool]] = deque()
        self._tasks: set[asyncio.Task] = set()
        self._closing = False

    def counts(self) -> tuple[int, int]:
        return self._running, len(self._waiters)

    def try_acquire(self) -> bool:
        if self._closing or self._running >= self._limit or self._waiters:
            return False
        self._running += 1
        return True

    async def wait_for_slot(self, body: _QueuedReceive) -> bool:
        if self._closing or (self._capacity > 0 and self._running + len(self._waiters) >= self._capacity):
            return False
        turn = asyncio.get_running_loop().create_future()
        self._waiters.append(turn)
        acquired = False
        try:
            granted = await body.wait(turn)
            acquired = granted and not self._closing
            return acquired
        finally:
            if not turn.done():
                self._waiters.remove(turn)
                turn.cancel()
            elif turn.result() and not acquired:
                # Disconnect/cancellation can coincide with a slot handoff.
                self.release()

    def release(self) -> None:
        if self._running < 1:
            raise RuntimeError("No execution slot to release.")
        if self._waiters:
            self._waiters.popleft().set_result(True)
        else:
            self._running -= 1

    async def run(self, app: ASGIApp, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            work = receive.serve(app, scope, send) if isinstance(receive, _QueuedReceive) else app(scope, receive, send)
            task = asyncio.create_task(work)
        except BaseException:
            self.release()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        # A disconnected/cancelled caller cannot free a slot while a synchronous
        # handler is still using the thread pool, Ollama or the document index.
        await asyncio.shield(task)

    def _finished(self, task: asyncio.Task) -> None:
        try:
            if not task.cancelled():
                # Observe detached task exceptions; an attached caller still
                # receives the same exception through shield and normal handlers.
                task.exception()
        finally:
            self._tasks.discard(task)
            self.release()

    async def wait_for_idle(self) -> None:
        self._closing = True
        while self._waiters:
            self._waiters.popleft().set_result(False)
        if self._tasks:
            await asyncio.shield(asyncio.gather(*self._tasks, return_exceptions=True))


class HeavyRequestMiddleware:
    def __init__(self, app: ASGIApp, limiter: RequestLimiter):
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not is_heavy_request(scope):
            await self.app(scope, receive, send)
            return
        if self.limiter.try_acquire():
            await self.limiter.run(self.app, scope, receive, send)
            return
        body = _QueuedReceive(receive)
        acquired = False
        try:
            acquired = await self.limiter.wait_for_slot(body)
        except ClientDisconnect:
            return
        finally:
            if not acquired:
                await body.close()
        if acquired:
            await self.limiter.run(self.app, scope, body, send)
            return
        response = JSONResponse(
            failure("service_busy", "系统正在处理其他文档或问答请求，请稍后重试。"),
            status_code=503,
            headers={"Retry-After": "1"},
        )
        await response(scope, receive, send)
