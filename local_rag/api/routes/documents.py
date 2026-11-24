"""Document upload, retrieval and download endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from local_rag.core.config import MIME_TYPES
from local_rag.core.exceptions import DocumentError
from local_rag.documents.service import DocumentStore
from local_rag.api.schemas import ApiResponse, DocumentData, DocumentListData, ChunksData, DeleteData, success

logger = logging.getLogger(__name__)


class _RequestTooLarge(MultiPartException):
    pass


def create_router(store: DocumentStore, index) -> APIRouter:
    router = APIRouter(tags=["documents"])
    settings = store.settings

    @router.post("/documents", status_code=201, response_model=ApiResponse[DocumentData],
                 summary="上传并处理单个文档", description="支持 PDF（含 OCR）、DOCX、Excel、CSV/TSV、文本和图片；默认保存、提取、分块并建立索引。"
                 "索引失败时 error.document_id 可用于重新索引。", openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["file"],
            "properties": {"file": {"type": "string", "format": "binary"}},
        }}}}
    })
    async def upload_document(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "multipart/form-data":
            raise DocumentError("invalid_upload", "请使用 multipart/form-data 的 file 字段上传文件。", 400)
        # Limit both declared and actual request bytes, before multipart spooling.
        # The extra allowance covers multipart boundaries and part headers only.
        request_limit = settings.max_file_bytes + 64 * 1024
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
                if declared_size < 0:
                    raise ValueError
            except ValueError as exc:
                raise DocumentError("invalid_upload", "Content-Length 无效。", 400) from exc
            if declared_size > request_limit:
                raise DocumentError("file_too_large", "上传请求超过大小上限。", 413)

        async def limited_stream():
            received = 0
            async for data in request.stream():
                received += len(data)
                if received > request_limit:
                    raise _RequestTooLarge("上传请求超过大小上限。")
                yield data

        try:
            form = await MultiPartParser(
                request.headers, limited_stream(), max_files=1, max_fields=0,
            ).parse()
        except _RequestTooLarge as exc:
            raise DocumentError("file_too_large", "上传请求超过大小上限。", 413) from exc
        except MultiPartException as exc:
            raise DocumentError("invalid_upload", "上传格式无效；请仅提供一个 file 文件字段。", 400) from exc
        try:
            file = form.get("file")
            if len(form.multi_items()) != 1 or not isinstance(file, UploadFile):
                raise DocumentError("invalid_upload", "必须且只能上传一个名为 file 的文件字段。", 400)
            try:
                # PDF/DOCX parsing and file IO must not block the event loop.
                return success(await run_in_threadpool(index.ingest, file.file, file.filename, file.content_type))
            except OSError as exc:
                logger.exception("Failed to store uploaded document")
                raise DocumentError("storage_error", "文档保存失败，请检查存储空间和目录权限。", 500) from exc
        finally:
            await form.close()

    @router.get("/documents", response_model=ApiResponse[DocumentListData], summary="查询文档登记表及处理状态")
    def list_documents():
        items = index.list_documents()
        return success({"documents": items, "count": len(items)})

    @router.post("/documents/{document_id}/reindex", response_model=ApiResponse[DocumentData], summary="重新生成文档向量索引")
    def reindex_document(document_id: str):
        return success(index.reindex(document_id))

    @router.delete("/documents/{document_id}", response_model=ApiResponse[DeleteData], summary="删除文档及其全部版本的向量")
    def delete_document(document_id: str):
        return success(index.delete(document_id))

    @router.get("/documents/{document_id}", response_model=ApiResponse[DocumentData], summary="读取文档信息及文本块")
    def get_document(document_id: str):
        return success(index.get(document_id))

    @router.get("/documents/{document_id}/chunks", response_model=ApiResponse[ChunksData], summary="读取文档文本块")
    def get_chunks(document_id: str):
        record = index.get(document_id)
        return success({"document_id": record["document_id"], "chunk_count": record["chunk_count"],
                "chunks": record["chunks"]})

    @router.get("/documents/{document_id}/original", response_class=FileResponse, summary="下载原始文档")
    def download_original(document_id: str):
        record = index.get(document_id)
        extension = "." + record["file_type"]
        return FileResponse(store.directory(document_id) / f"original{extension}",
                            filename=record["filename"], media_type=sorted(MIME_TYPES[extension])[0])

    @router.get("/documents/{document_id}/text", response_class=FileResponse, summary="下载清理后的纯文本")
    def download_text(document_id: str):
        record = index.get(document_id)
        return FileResponse(store.directory(document_id) / "text.txt", media_type="text/plain",
                            filename=Path(record["filename"]).stem + ".txt")

    return router
