"""Health and browser entry points."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator
from local_rag.core.config import validate_chat_model_name
from fastapi.responses import RedirectResponse, Response

from local_rag.core.config import DocumentSettings, MIME_TYPES, RuntimeSettings
from local_rag.api.schemas import ApiResponse, HealthData, success
from local_rag.documents.ocr import MODELS


class WarmupRequest(BaseModel):
    model: str | None = None

    @field_validator("model")
    @classmethod
    def valid_model(cls, value):
        return validate_chat_model_name(value) if value is not None else None


def create_router(settings: DocumentSettings, index, rag_settings=None,
                  runtime_settings: RuntimeSettings | None = None, warmup=None, *, limiter=None) -> APIRouter:
    router = APIRouter(tags=["system"])
    runtime_settings = runtime_settings if runtime_settings is not None else RuntimeSettings()

    @router.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse(url="/docs")

    @router.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    async def admission_counts():
        return limiter.counts() if limiter is not None else (0, 0)

    @router.get("/health", response_model=ApiResponse[HealthData], summary="检查 API 状态及当前配置")
    def health(counts: tuple[int, int] = Depends(admission_counts)):
        return success({"status": "ok", "warmup": warmup.last if warmup else {"status": "not_started"}, "supported_types": [ext[1:] for ext in MIME_TYPES],
                "ocr": {"mode": settings.ocr_mode, "engine": "RapidOCR/PP-OCRv6", "device": "cpu",
                        "models_present": all((settings.ocr_model_dir / name).is_file() for name, _ in MODELS.values()),
                        "max_pages": settings.ocr_max_pages, "dpi": settings.ocr_dpi},
                "max_file_bytes": settings.max_file_bytes,
                "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
                "auto_index": index.settings.auto_index,
                "embedding_model": index.settings.model_name,
                "vector_store": "chroma_persistent", "registry": "sqlite",
                "chat_model": rag_settings.model_name if rag_settings else None,
                "chat_backend": rag_settings.chat_backend if rag_settings else "ollama",
                "fast_mode": rag_settings.fast_mode if rag_settings else False,
                "rag_top_k": 4,
                "max_concurrent_requests": runtime_settings.max_concurrent_requests,
                "max_admitted_requests": runtime_settings.max_admitted_requests,
                "active_heavy_requests": counts[0], "queued_heavy_requests": counts[1],
                "concurrency_scope": "process",
                "embedding_batch_size": index.settings.batch_size,
                "rag_num_ctx": rag_settings.num_ctx if rag_settings else None,
                "rag_num_predict": rag_settings.num_predict if rag_settings else None,
                "model_readiness": "not_checked"})

    if warmup is not None:
        @router.post("/warmup", response_model=ApiResponse[dict], summary="预热本地模型和已有索引")
        def warm_resources(request: WarmupRequest | None = None):
            return success(warmup.run(request.model if request else None))

    return router
