"""Bounded, request-local conversation context; previous answers are not evidence."""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_rag.clients.ollama import OllamaOutputLimitError
from local_rag.core.cancellation import check_cancelled
from local_rag.core.exceptions import DocumentError
from local_rag.core.timing import model_phase

MAX_HISTORY_TURNS = 12
MAX_HISTORY_CHARACTERS = 24000


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=3000)


def validate_history(history):
    if history is None:
        return []
    if not isinstance(history, list) or len(history) > MAX_HISTORY_TURNS:
        raise ValueError(f"历史必须为最多 {MAX_HISTORY_TURNS} 轮问答的列表。")
    turns = [ConversationTurn.model_validate(t).model_dump() for t in history]
    if sum(len(t["question"]) + len(t["answer"]) for t in turns) > MAX_HISTORY_CHARACTERS:
        raise ValueError(f"聊天历史不能超过 {MAX_HISTORY_CHARACTERS} 个字符。")
    return turns


def normalize_history(history):
    try:
        return validate_history(history)
    except (ValidationError, ValueError, TypeError) as exc:
        raise DocumentError("invalid_conversation", "聊天历史格式无效或超过长度限制。", 422) from exc


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def bounded_history(history, byte_budget):
    """Keep recent complete exchanges; never splice around an omitted turn."""
    selected = []
    for turn in reversed(history):
        if encoded_size([turn, *selected]) <= byte_budget:
            selected.insert(0, turn)
            continue
        question_only = {"question": turn["question"], "answer_omitted": True}
        if encoded_size([question_only, *selected]) <= byte_budget:
            selected.insert(0, question_only)
        break
    return selected


FOLLOWUP_PROMPT = """把当前追问改写为含具体名称的完整检索问题。根据历史消息补上省略的对象，尤其将“它”替换成最近讨论的文物名称；当前明确换了对象就用新对象。只改写，不回答。返回 JSON：standalone_question。"""

REFERENCE = re.compile(r"它|他|她|这件|那件|该文物|该人物|那个|这个|刚才|上述|前面|继续|再说|详细|\b(it|its|they|their|that|those|more)\b", re.I)


@model_phase("conversation_rewrite")
def resolve_followup(client, model, question, history, *, num_ctx, num_predict, cancel=None):
    if not history:
        return question, "not_needed", 0
    budget = num_ctx - min(num_predict, 512) - 1024
    fixed = len(FOLLOWUP_PROMPT.encode("utf-8")) + encoded_size({"question": question, "history": []})
    selected = bounded_history(history, min(5000, max(0, budget - fixed)))
    if not selected:
        return question, "history_omitted", 0
    check_cancelled(cancel)
    prompt = json.dumps({"history": selected, "question": question}, ensure_ascii=False)
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"standalone_question": {"type": "string", "minLength": 1, "maxLength": 2000}},
              "required": ["standalone_question"]}
    try:
        raw = client.classify(model, prompt, system_prompt=FOLLOWUP_PROMPT, response_format=schema,
                              num_ctx=num_ctx, num_predict=min(num_predict, 512))
        check_cancelled(cancel)
        data = json.loads(raw)
        resolved = data.get("standalone_question") if isinstance(data, dict) else None
        if not isinstance(resolved, str) or not resolved.strip() or len(resolved.strip()) > 2000:
            raise ValueError("Invalid follow-up resolution")
        resolved = resolved.strip()
        if resolved == question and REFERENCE.search(question):
            raise ValueError("Follow-up reference was not resolved")
        return resolved, "resolved", len(selected)
    except (ValueError, TypeError, OllamaOutputLimitError):
        dependent = REFERENCE.search(question)
        suffix = f"\n后续问题：{question}"
        fallback = (history[-1]["question"][:2000 - len(suffix)] + suffix
                    if (dependent or len(question) < 12) and len(suffix) < 2000 else question)
        return fallback, "fallback", len(selected)
