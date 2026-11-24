"""Semantic retrieval over fully committed document indexes."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator
from local_rag.api.schemas import ApiResponse, SearchData, success


class SearchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"query": "文档中有哪些操作步骤？", "top_k": 5}
    ]})
    query: str = Field(min_length=1, max_length=4000, examples=["文档中有哪些操作步骤？"])
    mode: Literal["semantic", "hybrid"] = Field(default="semantic", description="semantic 为原有向量检索；hybrid 合并全书 BM25、别名扩展和语义结果")
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = Field(default=None, description="可选；填写上传响应 data.document_id，省略则检索所有已索引文档")

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("搜索内容不能为空。")
        return value.strip()


def create_router(index):
    router = APIRouter(tags=["search"])

    @router.post("/search", response_model=ApiResponse[SearchData], summary="使用本地 BGE-M3 搜索文档文本块")
    def search(request: SearchRequest):
        search_method = index.hybrid_search if request.mode == "hybrid" else index.search
        return success(search_method(request.query, request.top_k, request.document_id))

    return router
