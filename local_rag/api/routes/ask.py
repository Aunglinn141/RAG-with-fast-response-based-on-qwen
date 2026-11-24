"""Multilingual answers restricted to knowledge-base context, with authentic retrieval metadata."""

from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator
from local_rag.api.schemas import ApiResponse, ChatData, ModelsData, success
from local_rag.core.config import validate_chat_model_name
from local_rag.api.streaming import ChatStreamingResponse, chat_events
from local_rag.rag.conversation import ConversationTurn, MAX_HISTORY_TURNS, validate_history


class AskRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"question": "文档中有哪些操作步骤？", "top_k": 4}
    ]})
    question: str = Field(min_length=1, max_length=2000, examples=["文档中有哪些操作步骤？"])
    history: list[ConversationTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS,
                                            description="当前聊天最近的完整问答，最多 12 轮、合计 24000 字符；不作为事实证据")
    top_k: int | None = Field(default=None, ge=1, le=10, strict=True,
                              description="省略时自动检索全库并选择上下文；显式指定用于管理调试")
    document_id: UUID | None = Field(default=None, description="可选；填写上传响应 data.document_id，省略则检索所有已索引文档")

    model: str | None = Field(default=None, min_length=1, max_length=256,
                              description="可选；填写 GET /models 返回的本地聊天模型名称，省略时使用默认模型")

    @field_validator("model")
    @classmethod
    def valid_model(cls, value):
        return validate_chat_model_name(value) if value is not None else None

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空。")
        return value.strip()

    @field_validator("history")
    @classmethod
    def bounded_history(cls, value):
        validate_history(value)
        return value


def create_router(rag):
    router = APIRouter(tags=["chat"])

    @router.get("/models", response_model=ApiResponse[ModelsData], summary="列出所选推理服务提供的本地聊天模型")
    def models():
        return success(rag.available_models())

    @router.post("/chat", response_model=ApiResponse[ChatData],
                 summary="中英文知识库问答与简单问候，由所选本地模型回答")
    @router.post("/ask", response_model=ApiResponse[ChatData], deprecated=True,
                 summary="兼容问答入口，推荐使用 /chat",
                 description="保留旧路径；响应与 /chat 相同，业务数据统一位于 data。")
    def ask(request: AskRequest):
        return success(rag.ask(request.question, request.top_k,
                       str(request.document_id) if request.document_id is not None else None, model=request.model,
                       **({"history": [t.model_dump() for t in request.history]} if request.history else {})))

    @router.post("/chat/stream", response_class=ChatStreamingResponse,
                 summary="流式知识库问答：逐步输出正文，完成后返回最终结果",
                 responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}},
                                  "description": "SSE 事件：stage、delta、reset、result、error；data 使用统一 JSON 外壳。"}},
                 description="纯问候直接回复，中英文问题按提问语言回答。允许归纳和改写知识库信息，不要求逐字引用；知识库以外的信息不能回答。result 是最终结果；错误或拒答时替换或清除草稿。")
    async def ask_stream(request: AskRequest):
        return ChatStreamingResponse(chat_events(
            rag, request.question, request.top_k,
            str(request.document_id) if request.document_id is not None else None, request.model,
            history=[t.model_dump() for t in request.history]))

    return router
