"""Typed LangGraph orchestration with explicit, request-local conversation history."""

from dataclasses import dataclass
from threading import Event
from typing import Callable

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


@dataclass(frozen=True)
class RagRunContext:
    """Request-local controls, kept outside graph state and model prompts."""

    on_event: Callable[[str, dict], None] | None = None
    cancel: Event | None = None


class RagState(TypedDict, total=False):
    question: str
    history: list[dict]
    prompt_history: list[dict]
    resolved_question: str
    conversation_resolution: str
    rewrite_history_turns: int
    top_k: int | None
    document_id: str | None
    model: str | None
    model_name: str
    automatic: bool
    candidate_limit: int
    greeting: bool
    language: str
    enhanced: bool
    comprehensive: bool
    translated_query: str | None
    translation_status: str
    retrieved: dict
    result: dict
    raw_answer: str | None
    model_version: str


def _after_prepare(state: RagState):
    return "build_context" if state["greeting"] else "retrieve"


def _after_context(state: RagState):
    return "generate" if state["greeting"] or state["result"]["sources"] else END


def build_rag_graph(service):
    """Compile once per service; every invocation receives fresh state/context.

    The generation node retains the existing scope gate, bounded retries and
    translation policy. Streaming callbacks stay synchronous so the API's
    bounded SSE queue continues to apply backpressure and cancellation.
    """
    builder = StateGraph(RagState, context_schema=RagRunContext)
    builder.add_node("prepare", service._prepare_request)
    builder.add_node("retrieve", service._retrieve)
    builder.add_node("build_context", service._build_context)
    builder.add_node("generate", service._generate_answer)
    builder.add_node("finalize", service._finalize_answer)
    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", _after_prepare, {
        "retrieve": "retrieve", "build_context": "build_context",
    })
    builder.add_edge("retrieve", "build_context")
    builder.add_conditional_edges("build_context", _after_context, {
        "generate": "generate", END: END,
    })
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(name="local_rag")
