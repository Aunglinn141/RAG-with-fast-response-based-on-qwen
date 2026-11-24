"""Public HTTP contracts shared by response validation and Swagger UI."""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: Literal[True] = True
    data: T
    error: None = None


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ApiError(BaseModel):
    code: str = Field(description="供调用方判断的稳定错误码", examples=["generation_unavailable"])
    message: str = Field(description="可读的错误说明")
    document_id: str | None = Field(default=None, description="上传已保存但索引失败时，用于重新索引的文档 ID")
    details: list[ValidationIssue] | None = Field(default=None, description="请求校验错误的位置与原因，不包含原始请求内容")


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    data: None = None
    error: ApiError


def success(data):
    return {"success": True, "data": data, "error": None}


def failure(code, message, *, document_id=None, details=None):
    error = ApiError(code=code, message=message, document_id=document_id, details=details)
    return {"success": False, "data": None, "error": error.model_dump(exclude_none=True)}


ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "请求或上传表单无效", 404: "接口或文档不存在", 405: "请求方法不支持",
        409: "模型版本或文档状态冲突", 413: "上传超过大小上限", 415: "不支持的文件类型",
        422: "请求校验或文档解析失败", 500: "内部错误", 502: "模型返回格式无效或输出被截断",
        503: "模型、检索或存储服务暂不可用，或索引服务繁忙",
    }.items()
}


class HealthData(BaseModel):
    chat_backend: Literal["ollama", "vllm"] = "ollama"
    fast_mode: bool = False
    ocr: dict[str, Any] = Field(default_factory=dict, description="本地 OCR 配置与模型文件存在性；不执行识别")
    warmup: dict[str, Any] = Field(default_factory=dict, description="最近一次预热结果，非实时驻留保证")
    status: Literal["ok"]
    supported_types: list[str]
    max_file_bytes: int
    chunk_size: int
    chunk_overlap: int
    auto_index: bool
    embedding_model: str
    vector_store: str
    registry: str
    chat_model: str | None
    rag_top_k: int
    max_concurrent_requests: int
    max_admitted_requests: int = Field(ge=0, description="单进程可接纳的执行中与排队请求总数，0 表示应用层不设人数上限")
    active_heavy_requests: int
    queued_heavy_requests: int
    concurrency_scope: Literal["process"]
    embedding_batch_size: int
    rag_num_ctx: int | None
    rag_num_predict: int | None
    model_readiness: Literal["not_checked"] = Field(description="健康检查不执行推理；实际模型就绪以索引、检索或问答为准")


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    uploaded_at: str
    status: str
    chunk_count: int
    model_name: str | None
    model_version: str | None
    embedding_dimension: int | None
    collection_name: str | None
    generation: str | None
    indexed_at: str | None
    updated_at: str
    last_error: str | None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(description="文档 ID、文件名、页码、章节、分块编号与字符偏移等真实来源信息")


class DocumentData(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    created_at: str
    status: str
    page_count: int | None
    character_count: int
    chunk_count: int
    chunk_config: dict[str, int]
    warnings: list[str]
    original_url: str
    text_url: str
    chunks_url: str
    chunks: list[Chunk]
    index: DocumentItem


class DocumentListData(BaseModel):
    documents: list[DocumentItem]
    count: int


class ChunksData(BaseModel):
    document_id: str
    chunk_count: int
    chunks: list[Chunk]


class DeleteData(BaseModel):
    document_id: str
    status: Literal["deleted"]
    remaining_vectors: int


class EmbeddingModel(BaseModel):
    name: str
    version: str
    dimension: int


class SearchMatch(Chunk):
    distance: float
    score: float = Field(description="检索相似度，不代表答案正确的概率")
    model_version: str


class SearchData(BaseModel):
    timings: dict[str, Any] = Field(default_factory=dict, description="分阶段耗时，单位毫秒；不包含 LLM 生成")
    strategy: dict[str, Any] | None = None
    query: str
    matches: list[SearchMatch]
    model: EmbeddingModel | None
    excluded_documents: int


class AnswerSource(SearchMatch):
    source_id: str
    cited: bool = Field(description="兼容字段；当前不生成逐字引用，固定为 false，不表示答案引用了该块")
    truncated: bool


class Evidence(BaseModel):
    source_id: str
    quote: str


class RetrievalInfo(BaseModel):
    top_k: int
    candidate_limit: int = 4
    scope: Literal["all", "document"] = "all"
    selection: Literal["similarity", "diverse"] = "similarity"
    strategy: dict[str, Any] | None = None
    context_types: dict[str, int] = Field(default_factory=dict)
    used_sections: int = 0
    returned_chunks: int
    used_chunks: int
    excluded_documents: int
    model: EmbeddingModel | None


class ChatModel(BaseModel):
    name: str
    version: str | None


class LocalChatModel(BaseModel):
    name: str
    digest: str
    size_bytes: int
    parameter_size: str
    quantization: str
    capabilities: list[str]


class UnavailableModel(BaseModel):
    name: str
    reason: str


class ModelsData(BaseModel):
    backend: Literal["ollama", "vllm"] = "ollama"
    base_url: str = ""
    default_model: str
    ollama_url: str
    models: list[LocalChatModel]
    unavailable_models: list[UnavailableModel]


class ChatData(BaseModel):
    conversation: dict[str, Any] = Field(default_factory=dict, description="本轮历史使用数量、追问补全及裁剪信息；不返回原始聊天历史")
    timings: dict[str, Any] = Field(default_factory=dict, description="检索、生成耗时，单位毫秒；加载/解码细分仅 Ollama 提供")
    question: str
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    reason: Literal["no_sources", "out_of_scope", "model_abstained", "invalid_evidence", "missing_evidence"] | None = Field(
        default=None, description="当前使用 no_sources、out_of_scope 或 model_abstained；answered 时为空，旧引用失败枚举仅保留兼容")
    interaction: Literal["knowledge", "greeting"] = Field(default="knowledge", description="纯问候不检索文档；事实问答仍限制在知识库内")
    answer_policy: Literal["knowledge_base_only"] = "knowledge_base_only"
    sources: list[AnswerSource] = Field(description="实际提供给模型的检索上下文，不代表答案逐项引用或已验证语义")
    evidence: list[Evidence] = Field(default_factory=list, description="兼容旧响应；当前不要求逐字引用，固定为空列表")
    retrieval: RetrievalInfo
    model: ChatModel
    warnings: list[str]
