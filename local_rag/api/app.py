"""ASGI app: python -m uvicorn local_rag.api.app:app --host 127.0.0.1 --port 8000."""

from contextlib import asynccontextmanager
import logging
from starlette.concurrency import run_in_threadpool

from fastapi import FastAPI

from local_rag import __version__
from local_rag.api.routes import ask, artifacts, documents, search, system
from local_rag.artifacts.service import ArtifactService
from local_rag.api.errors import register_error_handlers
from local_rag.api.limits import HeavyRequestMiddleware, RequestLimiter
from local_rag.api.schemas import ERROR_RESPONSES
from local_rag.core.config import DocumentSettings, VectorSettings, RagSettings, RuntimeSettings
from local_rag.documents.service import DocumentStore
from local_rag.indexing.service import IndexService
from local_rag.rag.service import RagService
from local_rag.api.runtime import RuntimeWarmup
from local_rag.core.exceptions import DocumentError


def create_app(settings: DocumentSettings | None = None,
               vector_settings: VectorSettings | None = None, *, embedder=None,
               rag_settings: RagSettings | None = None, chat_client=None,
               runtime_settings: RuntimeSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else DocumentSettings.from_env()
    runtime = runtime_settings if runtime_settings is not None else RuntimeSettings.from_env()
    limiter = RequestLimiter(runtime.max_concurrent_requests, capacity=runtime.max_admitted_requests)
    store = DocumentStore(settings)
    index = IndexService(store, vector_settings or VectorSettings.from_env(), embedder)
    artifact_service = ArtifactService(index)
    index.artifacts = artifact_service
    rag = RagService(index, rag_settings or RagSettings.from_env(), chat_client)

    warmup = RuntimeWarmup(index, rag)

    @asynccontextmanager
    async def lifespan(api):
        try:
            # Injected providers own their initialization.
            if runtime.warmup_on_startup and embedder is None and chat_client is None:
                try:
                    await run_in_threadpool(warmup.run)
                except DocumentError as exc:
                    logging.getLogger(__name__).warning("Startup warm-up unavailable: %s", exc)
            else:
                warmup.last = {"status": "disabled"}
            yield
        finally:
            await limiter.wait_for_idle()
            index.close()

    api = FastAPI(title="本地 RAG API", version=__version__, lifespan=lifespan,
                  description="先上传文档并取得 data.document_id，再检索或使用 /chat 问答。"
                              "业务 JSON 统一返回 success、data、error；下载接口返回文件。"
                              "首次模型加载可能较慢，请等待响应。",
                  responses=ERROR_RESPONSES,
                  swagger_ui_parameters={"defaultModelsExpandDepth": -1, "displayRequestDuration": True,
                                         "validatorUrl": None},
                  openapi_tags=[{"name": "system", "description": "检查 API 服务及当前配置"},
                                {"name": "documents", "description": "上传、查询和删除文档及索引"},
                                {"name": "search", "description": "BGE-M3 检索原文"},
                                {"name": "artifacts", "description": "文物编号、结构化事实、向量检索与有来源的基础问答"},
                                {"name": "chat", "description": "可切换本地模型的知识库问答及真实来源"}])
    api.state.document_store = store
    api.state.index_service = index
    api.state.rag_service = rag
    api.state.artifact_service = artifact_service
    api.state.runtime_settings = runtime
    api.state.request_limiter = limiter
    api.state.warmup = warmup
    api.add_middleware(HeavyRequestMiddleware, limiter=limiter)

    register_error_handlers(api)

    api.include_router(system.create_router(settings, index, rag.settings, runtime, warmup, limiter=limiter))
    api.include_router(documents.create_router(store, index))
    api.include_router(search.create_router(index))
    api.include_router(ask.create_router(rag))
    api.include_router(artifacts.create_router(artifact_service))
    return api


app = create_app()
