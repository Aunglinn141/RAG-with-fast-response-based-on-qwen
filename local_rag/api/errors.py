"""Consistent JSON failures, without exposing tracebacks or raw request bodies."""

import logging

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from filelock import Timeout
from starlette.exceptions import HTTPException

from local_rag.api.schemas import failure
from local_rag.core.exceptions import DocumentError

logger = logging.getLogger(__name__)


def register_error_handlers(api):
    @api.exception_handler(DocumentError)
    async def document_error_handler(request, exc):
        return JSONResponse(status_code=exc.status,
                            content=failure(exc.code, exc.message, document_id=exc.document_id))

    @api.exception_handler(Timeout)
    async def busy_handler(request, exc):
        return JSONResponse(status_code=503,
                            content=failure("index_busy", "索引服务繁忙，请稍后重试。"))

    @api.exception_handler(RequestValidationError)
    async def validation_handler(request, exc):
        details = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                   for item in exc.errors()]
        return JSONResponse(status_code=422,
                            content=failure("validation_error", "请求参数无效，请检查字段、类型和取值范围。", details=details))

    @api.exception_handler(HTTPException)
    async def http_error_handler(request, exc):
        code, message = {
            404: ("not_found", "请求的接口或资源不存在。"),
            405: ("method_not_allowed", "该接口不支持此请求方法。"),
        }.get(exc.status_code, ("http_error", "HTTP 请求未完成，请检查请求方法和内容。"))
        return JSONResponse(status_code=exc.status_code, headers=exc.headers, content=failure(code, message))

    @api.exception_handler(Exception)
    async def internal_error_handler(request, exc):
        logger.exception("Unhandled API error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500,
                            content=failure("internal_error", "服务内部错误，请查看 API 日志后重试。"))
