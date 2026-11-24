"""Bounded SSE bridge: run synchronous RAG once and retain admission until it exits."""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
import json
import logging
from threading import Event

import anyio
from filelock import Timeout
from starlette.responses import StreamingResponse

from local_rag.api.schemas import failure, success
from local_rag.core.cancellation import RequestCancelled
from local_rag.core.exceptions import DocumentError

logger = logging.getLogger(__name__)


def encode_event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def chat_events(rag, question, top_k=None, document_id=None, model=None, *, history=None, heartbeat=10):
    queue = asyncio.Queue(maxsize=32)
    stopped = Event()
    loop = asyncio.get_running_loop()

    def enqueue(name, payload):
        if stopped.is_set():
            return
        future = asyncio.run_coroutine_threadsafe(queue.put((name, payload)), loop)
        while not stopped.is_set():
            try:
                future.result(timeout=.1)
                return
            except FutureTimeout:
                continue
        future.cancel()

    def emit(name, data):
        enqueue(name, success(data))

    def work():
        try:
            result = rag.ask(question, top_k, document_id, model=model, on_event=emit, cancel=stopped,
                             **({"history": history} if history else {}))
            enqueue("result", success(result))
        except RequestCancelled:
            pass
        except DocumentError as exc:
            enqueue("error", failure(exc.code, exc.message, document_id=exc.document_id))
        except Timeout:
            enqueue("error", failure("index_busy", "索引服务繁忙，请稍后重试。"))
        except Exception:
            logger.exception("Streaming RAG failed")
            enqueue("error", failure("internal_error", "服务内部错误，请查看 API 日志后重试。"))
        finally:
            enqueue("", None)

    worker = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            try:
                name, payload = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if not name:
                break
            yield encode_event(name, payload)
            if name in {"result", "error"}:
                break
    finally:
        stopped.set()
        # Do not release the heavy-request slot while an index read or model
        # request is still alive. A stalled upstream remains bounded by timeout.
        with anyio.CancelScope(shield=True):
            await asyncio.shield(worker)


class ChatStreamingResponse(StreamingResponse):
    media_type = "text/event-stream"

    def __init__(self, content, status_code=200):
        super().__init__(content, status_code=status_code, media_type=self.media_type,
                         headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # ASGI 2.4 send errors happen outside the generator; explicitly close
            # it so disconnects also signal cancellation and wait for cleanup.
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
