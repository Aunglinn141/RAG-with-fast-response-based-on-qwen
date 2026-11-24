"""Memory admission tests use isolated storage and no live model or network."""

import asyncio
from threading import Event

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from local_rag.api.limits import HeavyRequestMiddleware, RequestLimiter
from local_rag.core.config import DocumentSettings, RagSettings, RuntimeSettings, VectorSettings
from tests.fixtures.embeddings import FakeEmbeddings
from tests.unit.test_rag import FakeChat


@pytest.fixture
def resource_app(isolated_root):
    from local_rag.api.app import create_app

    app = create_app(
        DocumentSettings(storage_dir=isolated_root / "documents"),
        VectorSettings(chroma_dir=isolated_root / "chroma", registry_path=isolated_root / "registry.sqlite3"),
        embedder=FakeEmbeddings(), chat_client=FakeChat(), rag_settings=RagSettings(),
        runtime_settings=RuntimeSettings(max_concurrent_requests=1, max_admitted_requests=1),
    )
    yield app
    app.state.index_service.close()


def assert_busy(response):
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    payload = response.json()
    assert payload["success"] is False
    assert payload["data"] is None
    assert payload["error"]["code"] == "service_busy"
    assert payload["error"]["message"]


def test_memory_defaults_and_env(monkeypatch):
    for name in ("RAG_MAX_CONCURRENT_REQUESTS", "RAG_MAX_ADMITTED_REQUESTS", "RAG_NUM_CTX", "RAG_NUM_PREDICT", "EMBED_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert RuntimeSettings.from_env().max_concurrent_requests == 2
    assert RuntimeSettings.from_env().max_admitted_requests == RuntimeSettings().max_admitted_requests == 0
    assert RagSettings.from_env().num_ctx == 8192
    assert RagSettings.from_env().num_predict == 2048
    assert RagSettings().model_name == "qwen3:1.7b"
    assert VectorSettings.from_env().batch_size == 8
    monkeypatch.setenv("RAG_MAX_CONCURRENT_REQUESTS", "2")
    monkeypatch.setenv("RAG_MAX_ADMITTED_REQUESTS", "3")
    monkeypatch.setenv("RAG_NUM_CTX", "4096")
    monkeypatch.setenv("RAG_NUM_PREDICT", "1536")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "4")
    assert RuntimeSettings.from_env().max_concurrent_requests == 2
    assert RuntimeSettings.from_env().max_admitted_requests == 3
    assert RagSettings.from_env().num_ctx == 4096
    assert RagSettings.from_env().num_predict == 1536
    assert VectorSettings.from_env().batch_size == 4


@pytest.mark.parametrize("value", [0, -1, 5, True, 1.5, "2"])
def test_invalid_runtime_settings(value):
    with pytest.raises(ValueError, match="RAG_MAX_CONCURRENT_REQUESTS"):
        RuntimeSettings(max_concurrent_requests=value)


@pytest.mark.parametrize("value", ["0", "5", "1.5", "invalid"])
def test_invalid_runtime_env(monkeypatch, value):
    monkeypatch.setenv("RAG_MAX_CONCURRENT_REQUESTS", value)
    with pytest.raises(ValueError):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("capacity", [-1, 1, True, False, 100.5])
def test_invalid_admission_capacity(capacity):
    with pytest.raises(ValueError, match="RAG_MAX_ADMITTED_REQUESTS"):
        RuntimeSettings(max_concurrent_requests=2, max_admitted_requests=capacity)
    with pytest.raises(ValueError, match="Admission capacity"):
        RequestLimiter(2, capacity=capacity)


def test_admission_capacity_env_accepts_zero_and_rejects_negative(monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONCURRENT_REQUESTS", "2")
    monkeypatch.setenv("RAG_MAX_ADMITTED_REQUESTS", "0")
    assert RuntimeSettings.from_env().max_admitted_requests == 0
    monkeypatch.setenv("RAG_MAX_ADMITTED_REQUESTS", "-1")
    with pytest.raises(ValueError, match="RAG_MAX_ADMITTED_REQUESTS"):
        RuntimeSettings.from_env()


def test_health_exposes_limits_and_validation_releases_slot(resource_app):
    with TestClient(resource_app, raise_server_exceptions=False) as client:
        health = client.get("/health").json()["data"]
        assert health["max_concurrent_requests"] == 1
        assert health["max_admitted_requests"] == 1
        assert (health["active_heavy_requests"], health["queued_heavy_requests"]) == (0, 0)
        assert health["concurrency_scope"] == "process"
        assert health["embedding_batch_size"] == 8
        assert health["rag_num_ctx"] == 8192
        assert health["rag_num_predict"] == 2048
        assert health["model_readiness"] == "not_checked"
        invalid = client.post("/chat", json={})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "validation_error"
        assert client.post("/search", json={"query": "测试"}).status_code == 200


def test_busy_api_rejects_all_heavy_routes_but_serves_health_and_docs(resource_app, monkeypatch):
    entered, release = Event(), Event()
    original_search = resource_app.state.index_service.search

    def blocking_search(*args, **kwargs):
        entered.set()
        assert release.wait(10), "Test failed to release worker"
        return original_search(*args, **kwargs)

    monkeypatch.setattr(resource_app.state.index_service, "search", blocking_search)

    async def scenario():
        transport = httpx.ASGITransport(app=resource_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/search", json={"query": "测试"}))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                routes = [("POST", path) for path in (
                    "/documents", "/documents/", "/search", "/chat", "/ask",
                    "/documents/11111111-1111-4111-8111-111111111111/reindex",
                )]
                routes.append(("DELETE", "/documents/11111111-1111-4111-8111-111111111111"))
                for method, path in routes:
                    assert_busy(await asyncio.wait_for(client.request(method, path), 2))
                for path in ("/health", "/docs", "/openapi.json"):
                    assert (await asyncio.wait_for(client.get(path), 2)).status_code == 200
            finally:
                release.set()
                result = await asyncio.wait_for(first, 5)
            assert result.status_code == 200
            assert (await client.post("/search", json={"query": "测试"})).status_code == 200

    asyncio.run(scenario())


def test_exception_releases_slot():
    app = FastAPI()
    limiter = RequestLimiter(1)
    app.add_middleware(HeavyRequestMiddleware, limiter=limiter)
    attempts = 0

    @app.post("/chat")
    def chat():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic failure")
        return {"ok": True}

    async def scenario():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.post("/chat")).status_code == 500
            assert (await client.post("/chat")).status_code == 200
        await limiter.wait_for_idle()

    asyncio.run(scenario())


def test_cancelled_caller_keeps_slot_until_worker_finishes_and_shutdown_waits():
    app = FastAPI()
    limiter = RequestLimiter(1)
    app.add_middleware(HeavyRequestMiddleware, limiter=limiter)
    entered, release, finished = Event(), Event(), Event()

    @app.post("/chat")
    def chat():
        entered.set()
        try:
            assert release.wait(10), "Test failed to release worker"
            return {"ok": True}
        finally:
            finished.set()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = asyncio.create_task(client.post("/chat"))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert not finished.is_set()
                assert_busy(await client.post("/chat"))
                shutdown = asyncio.create_task(limiter.wait_for_idle())
                await asyncio.sleep(0)
                assert not shutdown.done()
            finally:
                release.set()
                await asyncio.wait_for(limiter.wait_for_idle(), 5)
            assert finished.is_set()
            await shutdown

    asyncio.run(scenario())


def test_configurable_limit_allows_two_requests_and_rejects_third():
    app = FastAPI()
    limiter = RequestLimiter(2)
    app.add_middleware(HeavyRequestMiddleware, limiter=limiter)

    async def scenario():
        entered = asyncio.Queue()
        release = asyncio.Event()

        @app.post("/chat")
        async def chat():
            entered.put_nowait(True)
            await release.wait()
            return {"ok": True}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            requests = [asyncio.create_task(client.post("/chat")) for _ in range(2)]
            try:
                for _ in requests:
                    await asyncio.wait_for(entered.get(), 2)
                assert_busy(await client.post("/chat"))
            finally:
                release.set()
                results = await asyncio.gather(*requests)
                await limiter.wait_for_idle()
            assert all(response.status_code == 200 for response in results)

    asyncio.run(scenario())
