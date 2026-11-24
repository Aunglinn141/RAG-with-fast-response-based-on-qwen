"""Retrieve knowledge-base context and generate a scoped answer with a local model."""

from copy import deepcopy
from contextlib import closing
import json
import logging
import re

from langgraph.runtime import Runtime

from local_rag.clients.chat import create_chat_client
from local_rag.clients.errors import (ModelError as OllamaError, ModelOutputLimitError as OllamaOutputLimitError,
                                      ModelUnavailableError as OllamaModelError)
from local_rag.core.config import validate_chat_model_name
from local_rag.core.exceptions import DocumentError
from local_rag.rag.prompts import answer_schema, generation_prompt, needs_artifact_guidance, TRANSLATION_PROMPT, COMPACT_TRANSLATION_PROMPT
from local_rag.rag.retrieval import select_context_candidates
from local_rag.rag.scenarios import plan_retrieval
from local_rag.rag.context import crop_metadata, fit_source
from local_rag.rag.conversation import bounded_history, normalize_history, resolve_followup
from local_rag.rag.streaming import AnswerPreview
from local_rag.rag.workflow import RagRunContext, RagState, build_rag_graph
from local_rag.rag.scope import SCOPE_PROMPT, CATALOG_SCOPE_PROMPT, within_knowledge_scope, requires_live_exhibition_evidence
from local_rag.rag.name_aliases import name_hints, translation_glossary
from local_rag.rag.language import INSUFFICIENT_ANSWERS, answer_language, is_greeting, query_language, retrieval_query
from local_rag.core.cancellation import check_cancelled
from local_rag.core.timing import timed_operation, model_phase
from local_rag.artifacts.retrieval import catalog_contexts, conversation_catalog_query
from local_rag.artifacts.presentation import public_artifact_text, PublicArtifactEvents

logger = logging.getLogger(__name__)
INSUFFICIENT_ANSWER = INSUFFICIENT_ANSWERS["zh"]


class RagService:
    def __init__(self, index, settings, client=None):
        self.index, self.settings = index, settings
        self.client = client if client is not None else create_chat_client(settings)
        self.graph = build_rag_graph(self)

    @staticmethod
    def _prompt(question, sources, *, language=None, history=None, resolved_question=None):
        # Retrieval still uses stable IDs to distinguish same-name records. The
        # generation boundary uses names, including history from older clients.
        public = lambda text: public_artifact_text(text, sources)
        question = public(question)
        resolved_question = public(resolved_question) if resolved_question else None
        # Budgeted history may deliberately retain only the question. Preserve
        # answer_omitted rather than requiring or inventing an answer field.
        history = [{key: public(value) if key in ('question', 'answer') else value
                    for key, value in turn.items()} for turn in history] if history else None
        return json.dumps({"question": resolved_question or question, "reply_language": language or answer_language(question),
                           **({"conversation": history} if history else {}),
                           **({"original_question": question} if resolved_question and resolved_question != question else {}),
                           "name_hints": [hint for hint in name_hints(resolved_question or question) if any(hint["name"] in s["text"] for s in sources)],
                           **({"people": next(s["metadata"]["query_people"] for s in sources if s["metadata"].get("query_people"))}
                              if any(s["metadata"].get("query_people") for s in sources) else {}),
                           "contexts": [
            {"source_id": source["source_id"], "text": public(source["text"]),
             **({"section": source["metadata"]["section_title"]} if source["metadata"].get("context_type") else {})}
            for source in sources
        ]}, ensure_ascii=False)

    def _prompt_budget(self, question, matches, *, balanced=False, conversation=False):
        # UTF-8 bytes conservatively bound byte-level BPE tokens; reserve output and template space.
        byte_budget = self.settings.num_ctx - self.settings.num_predict - 1024
        # Reserve the larger of the two complete prompts; retry replaces the style instructions.
        artifacts = needs_artifact_guidance(question, matches)
        catalog_only = bool(matches) and all(s.get('metadata', {}).get('context_type') == 'artifact' for s in matches)
        fixed_bytes = max(len(generation_prompt("zh", compact=compact, comprehensive=overview, artifacts=artifacts,
                                                conversation=conversation).encode("utf-8"))
                          for compact, overview in [(False, False), (True, False)] + ([(False, True)] if balanced and not catalog_only else []))
        return byte_budget, max(fixed_bytes, len((CATALOG_SCOPE_PROMPT if catalog_only else SCOPE_PROMPT).encode("utf-8")))

    def _sources(self, question, matches, *, balanced=False, history=None, resolved_question=None):
        byte_budget, fixed_bytes = self._prompt_budget(question, matches, balanced=balanced,
                                                      conversation=bool(history or resolved_question))

        def prompt_size(sources):
            return len(self._prompt(question, sources, language="zh", history=history,
                                    resolved_question=resolved_question).encode("utf-8"))

        if fixed_bytes + prompt_size([]) > byte_budget:
            raise DocumentError("question_too_long", "问题超出当前模型上下文预算，请缩短问题。", 422)
        sources, seen = [], set()
        remaining = self.settings.max_context_characters
        for position, match in enumerate(matches):
            if not match.get("text", "").strip() or match["chunk_id"] in seen or remaining <= 0:
                continue
            seen.add(match["chunk_id"])
            source = {**deepcopy(match), "source_id": f"S{len(sources) + 1}", "cited": False}
            original = match["text"]
            if source["metadata"].get("atomic"):
                size = fixed_bytes + prompt_size([*sources, source])
                if len(original) > remaining or size > byte_budget:
                    # Do not fall through to unqualified raw excerpts after losing
                    # a structured record's conflict/identity restrictions.
                    # Losing one object in a comparison or ambiguity group must
                    # not leave a seemingly unique, one-sided catalog answer.
                    if source["metadata"].get("context_type") == "artifact":
                        return []
                    break
                source["truncated"] = False
                sources.append(source)
                remaining -= len(original)
                continue
            char_share = remaining // max(1, len(matches) - position) if balanced else remaining
            lo, hi = 0, min(len(original), char_share)
            used_bytes = fixed_bytes + prompt_size(sources)
            share = (byte_budget - used_bytes) // max(1, len(matches) - position) if balanced else byte_budget
            while lo < hi:
                mid = (lo + hi + 1) // 2
                source["text"] = fit_source(match, mid)[0] if balanced else original[:mid]
                size = fixed_bytes + prompt_size([*sources, source])
                if size <= byte_budget and (not balanced or size - used_bytes <= share):
                    lo = mid
                else:
                    hi = mid - 1
            if lo < 4:
                break
            source["text"], shift = fit_source(match, lo) if balanced else (original[:lo], 0)
            crop_metadata(source, shift)
            source["truncated"] = lo < len(original)
            sources.append(source)
            remaining -= len(source["text"])
        return sources

    @staticmethod
    def _decode(raw):
        try:
            result = json.loads(raw)
            if (not isinstance(result, dict) or type(result.get("answerable")) is not bool
                    or not isinstance(result.get("answer"), str) or len(result["answer"]) > 3000
                    or result["answerable"] and not result["answer"].strip()):
                raise ValueError("Invalid answer structure")
            # Old models may emit extra citation fields; they do not gate answers
            # or become trusted source metadata under the knowledge-base policy.
            return {"answerable": result["answerable"], "answer": result["answer"]}
        except (ValueError, TypeError) as exc:
            raise DocumentError("invalid_answer", "本地聊天模型未返回完整的回答格式，请重试或切换模型。", 502) from exc

    def _generate(self, question, sources, model_name, *, comprehensive=False, greeting=False, history=None,
                  resolved_question=None, on_event=None, cancel=None):
        if on_event is not None:
            on_event = PublicArtifactEvents(on_event, sources, language=answer_language(question))
        version = self.client.model_digest(model_name)
        translate = not greeting and answer_language(question) == "en"
        prompt = self._prompt(question, sources, language="zh" if translate else None,
                              history=history, resolved_question=resolved_question)
        catalog_only = bool(sources) and all(s.get('metadata', {}).get('context_type') == 'artifact' for s in sources)
        clarification = catalog_only and any(s['metadata'].get('resolution') != '名称或系统编号匹配' for s in sources)
        if catalog_only and not clarification and requires_live_exhibition_evidence(question):
            return None, version
        if not greeting and not clarification and not within_knowledge_scope(self.client, model_name, prompt, num_ctx=self.settings.num_ctx,
                                      num_predict=self.settings.num_predict, cancel=cancel, catalog_only=catalog_only,
                                      fast=self.settings.fast_mode):
            if self.client.model_digest(model_name) != version:
                raise DocumentError("chat_model_version_mismatch", "回答范围判断期间模型版本发生变化，请重试。", 409)
            return None, version
        if on_event is not None:
            on_event("stage", {"stage": "generating"})
        for compact in (False, True):
            try:
                check_cancelled(cancel)
                options = dict(
                    system_prompt=generation_prompt("zh" if translate else answer_language(question), compact=compact, comprehensive=comprehensive and not catalog_only, greeting=greeting,
                                                    artifacts=needs_artifact_guidance(question, sources),
                                                    conversation=bool(history or resolved_question), clarification=clarification),
                    response_format=answer_schema(compact=compact, comprehensive=comprehensive and not catalog_only, language="zh" if translate else answer_language(question), greeting=greeting),
                    num_ctx=self.settings.num_ctx, num_predict=self.settings.num_predict,
                )
                if on_event is None or translate:
                    raw = self.client.chat(model_name, prompt, **options)
                else:
                    preview = AnswerPreview()
                    with closing(self.client.chat_stream(model_name, prompt, cancel=cancel, **options)) as stream:
                        for fragment in stream:
                            check_cancelled(cancel)
                            delta = preview.feed(fragment)
                            if delta:
                                on_event("delta", {"text": delta})
                    raw = preview.raw
                    check_cancelled(cancel)
            except OllamaOutputLimitError as exc:
                if self.client.model_digest(model_name) != version:
                    raise DocumentError("chat_model_version_mismatch", "回答期间 聊天模型版本发生变化，请重试。", 409) from exc
                if compact:
                    raise
                if on_event is not None:
                    on_event("reset", {"reason": "generation_retry"})
                    on_event("stage", {"stage": "retrying"})
                logger.warning("Local chat model output reached %s tokens; retrying once with a compact answer.",
                               exc.num_predict)
                continue
            if self.client.model_digest(model_name) != version:
                raise DocumentError("chat_model_version_mismatch", "回答期间 聊天模型版本发生变化，请重试。", 409)
            if translate:
                draft = self._decode(raw)
                if draft["answerable"]:
                    raw = self._translate_answer(public_artifact_text(draft["answer"], sources), model_name, version,
                                                 comprehensive=comprehensive, on_event=on_event, cancel=cancel)
            if on_event is not None:
                on_event.finish()
            return raw, version

    @model_phase("answer_translation")
    def _translate_answer(self, answer, model_name, version, *, comprehensive=False, on_event=None, cancel=None):
        # The translator receives only the completed KB answer, never the original
        # question. It cannot use a second retrieval or free-form QA prompt.
        glossary = translation_glossary(answer)
        prompt = json.dumps({"text": answer, **({"glossary": glossary} if glossary else {})}, ensure_ascii=False)
        for compact in (False, True):
            check_cancelled(cancel)
            system = COMPACT_TRANSLATION_PROMPT if compact else TRANSLATION_PROMPT
            if len(system.encode("utf-8")) + len(prompt.encode("utf-8")) + self.settings.num_predict + 1024 > self.settings.num_ctx:
                raise DocumentError("answer_translation_unavailable", "知识库答案超出翻译上下文预算，请缩小问题范围。", 502)
            schema = answer_schema(language="en", compact=compact, comprehensive=comprehensive)
            schema["properties"].pop("answerable")
            schema["required"] = ["answer"]
            schema["properties"]["answer"]["pattern"] = "^[^\u3400-\u9fff\uf900-\ufaff\U00020000-\U000323af]*$"
            options = dict(system_prompt=system, response_format=schema,
                           num_ctx=self.settings.num_ctx, num_predict=self.settings.num_predict)
            try:
                if on_event is None:
                    raw = self.client.chat(model_name, prompt, **options)
                else:
                    preview = AnswerPreview()
                    with closing(self.client.chat_stream(model_name, prompt, cancel=cancel, **options)) as stream:
                        for fragment in stream:
                            check_cancelled(cancel)
                            delta = preview.feed(fragment)
                            if delta:
                                on_event("delta", {"text": delta})
                    raw = preview.raw
                check_cancelled(cancel)
            except OllamaOutputLimitError:
                if self.client.model_digest(model_name) != version:
                    raise DocumentError("chat_model_version_mismatch", "翻译期间模型版本发生变化，请重试。", 409)
                if compact:
                    raise
                if on_event is not None:
                    on_event("reset", {"reason": "translation_retry"})
                    on_event("stage", {"stage": "retrying"})
                continue
            if self.client.model_digest(model_name) != version:
                raise DocumentError("chat_model_version_mismatch", "翻译期间模型版本发生变化，请重试。", 409)
            try:
                data = json.loads(raw)
                translated = data.get("answer") if isinstance(data, dict) else None
                if not isinstance(translated, str) or not translated.strip() or len(translated) > 3000:
                    raise ValueError("Expected a non-empty English translation")
            except (ValueError, TypeError) as exc:
                raise DocumentError("answer_translation_unavailable", "本地模型未返回有效的英文译文，请重试或切换模型。", 502) from exc
            # A mostly-English language score still accepts untranslated Chinese
            # words. Retry from the same grounded draft, never complete them from
            # model memory or silently delete the untranslated content.
            if query_language(translated) != "en" or re.search(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U000323af]", translated):
                if compact:
                    raise DocumentError("answer_translation_unavailable", "本地模型未完成英文翻译，请重试或切换模型。", 502)
                if on_event is not None:
                    on_event("reset", {"reason": "translation_retry"})
                    on_event("stage", {"stage": "retrying"})
                continue
            return json.dumps({"answerable": True, "answer": translated.strip()}, ensure_ascii=False)

    def available_models(self):
        try:
            return {"default_model": self.settings.model_name,
                    "backend": self.settings.chat_backend,
                    "base_url": self.settings.vllm_url if self.settings.chat_backend == "vllm" else self.settings.ollama_url,
                    "ollama_url": self.settings.ollama_url, **self.client.list_chat_models()}
        except OllamaError as exc:
            raise DocumentError("models_unavailable", "无法读取本地聊天模型列表，请确认所选推理服务已启动。", 503) from exc

    @timed_operation("chat")
    def ask(self, question, top_k=4, document_id=None, model=None, *, history=None, on_event=None, cancel=None):
        """Run the shared graph for JSON, SSE and CLI callers."""
        state = self.graph.invoke(
            {"question": question, "top_k": top_k, "document_id": document_id, "model": model, "history": history},
            context=RagRunContext(on_event=on_event, cancel=cancel),
        )
        return state["result"]

    def _prepare_request(self, state: RagState, runtime: Runtime[RagRunContext]):
        run = runtime.context or RagRunContext()
        check_cancelled(run.cancel)
        question, top_k = state["question"], state.get("top_k", 4)
        history = normalize_history(state.get("history"))
        document_id, model = state.get("document_id"), state.get("model")
        model_name = self.settings.model_name
        if model is not None:
            try:
                model_name = validate_chat_model_name(model)
                self.client.model_digest(model_name)
                self.client.require_local_model(model_name, "completion")
            except (ValueError, OllamaModelError) as exc:
                raise DocumentError("invalid_chat_model", str(exc), 422) from exc
            except OllamaError as exc:
                raise DocumentError("generation_unavailable",
                                    f"无法检查项目中的模型 {model_name}，请确认所选推理服务正常运行。", 503) from exc
        automatic = top_k is None and document_id is None
        top_k = self.settings.context_chunks if automatic else (4 if top_k is None else top_k)
        candidate_limit = self.settings.retrieval_candidates if automatic else top_k
        question = question.strip()
        if not question or len(question) > 2000 or type(top_k) is not int or not 1 <= top_k <= 10:
            raise DocumentError("invalid_question", "问题须为 1–2000 个字符，Top-K 须为 1–10。", 422)
        greeting = is_greeting(question)
        language = answer_language(question)
        enhanced = not greeting and automatic and self.settings.enhanced_retrieval and hasattr(self.index, "retrieve_context")
        return {
            "question": question, "top_k": top_k, "document_id": document_id,
            "history": history, "prompt_history": [], "resolved_question": question,
            "conversation_resolution": "not_needed", "rewrite_history_turns": 0,
            "model_name": model_name, "automatic": automatic, "candidate_limit": candidate_limit,
            "greeting": greeting, "language": language, "enhanced": enhanced, "comprehensive": False,
            "translated_query": None, "translation_status": "not_needed",
            "retrieved": {"matches": [], "model": None, "strategy": {"scene": "greeting", "retrieval_skipped": True}}
                         if greeting else {},
        }

    def _retrieve(self, state: RagState, runtime: Runtime[RagRunContext]):
        run = runtime.context or RagRunContext()
        check_cancelled(run.cancel)
        model_name = state["model_name"]
        if run.on_event is not None:
            run.on_event("stage", {"stage": "retrieving"})
        check_cancelled(run.cancel)
        artifacts = getattr(self.index, "artifacts", None)
        try:
            catalog_resolution = (conversation_catalog_query(artifacts, state["question"], state["history"],
                                                             document_id=state["document_id"]) if artifacts else None)
            resolved, resolution, rewrite_turns = catalog_resolution or resolve_followup(
                    self.client, model_name, state["question"], state["history"], num_ctx=self.settings.num_ctx,
                    num_predict=self.settings.num_predict, cancel=run.cancel)
            query, translated, translation_status = retrieval_query(
                self.client, model_name, resolved, num_ctx=self.settings.num_ctx,
                num_predict=self.settings.num_predict, cancel=run.cancel)
        except OllamaError as exc:
            raise DocumentError("generation_unavailable",
                                f"无法使用本地模型 {model_name} 处理问题语言，请检查所选推理服务。", 503) from exc
        comprehensive = False
        catalog_query = resolved if re.search(r"WW-[0-9a-f]{32}", resolved, re.I) else query
        artifact_matches = catalog_contexts(artifacts, catalog_query, document_id=state["document_id"]) if artifacts else []
        try:
            if state["enhanced"]:
                plan = plan_retrieval(query, self.settings)
                retrieved = self.index.retrieve_context(query, state["candidate_limit"], state["document_id"], plan, self.settings)
                comprehensive = plan.scene in {"overview", "timeline", "relationship", "cross_chapter", "comparison"}
            else:
                retrieved = self.index.search(query, state["candidate_limit"], state["document_id"])
        except DocumentError as exc:
            if not artifact_matches or exc.code not in {"embedding_unavailable", "model_version_mismatch", "search_failed"}:
                raise
            retrieved = {"matches": [], "contexts": [], "model": None,
                         "warnings": ["原文检索暂不可用，本次仅向模型提供可用的文物结构化证据。"]}
        retrieved["artifact_contexts"] = artifact_matches
        check_cancelled(run.cancel)
        return {"retrieved": retrieved, "translated_query": translated,
                "translation_status": translation_status, "comprehensive": comprehensive,
                "resolved_question": resolved, "conversation_resolution": resolution, "rewrite_history_turns": rewrite_turns}

    def _build_context(self, state: RagState, runtime: Runtime[RagRunContext]):
        run = runtime.context or RagRunContext()
        check_cancelled(run.cancel)
        question, top_k = state["question"], state["top_k"]
        document_id, model_name = state["document_id"], state["model_name"]
        automatic, candidate_limit = state["automatic"], state["candidate_limit"]
        greeting, language, enhanced = state["greeting"], state["language"], state["enhanced"]
        retrieved = state["retrieved"]
        translated, translation_status = state["translated_query"], state["translation_status"]
        candidates = retrieved["matches"][:candidate_limit]
        matches = (retrieved["contexts"] if enhanced else
                   select_context_candidates(candidates, top_k) if automatic else candidates[:top_k])
        artifact_matches = retrieved.get("artifact_contexts", [])
        if artifact_matches:
            complete_descriptions = all(s["metadata"].get("complete_catalog") and
                                        s["metadata"].get("resolution") != "仅为检索候选，身份未确定，需要用户明确名称或编号"
                                        for s in artifact_matches)
            # A reviewed description already carries this object's original
            # narrative. Unrelated retrieval hits and neighboring objects on the
            # same PDF page must not be spliced into that object's facts.
            unresolved = any(s['metadata'].get('resolution') != '名称或系统编号匹配' for s in artifact_matches)
            matches = artifact_matches if complete_descriptions or unresolved else [*artifact_matches, *matches][:top_k]
        resolved = state["resolved_question"] if state["resolved_question"] != question else None
        prompt_history = []
        if state["history"] and not greeting:
            byte_budget, fixed_bytes = self._prompt_budget(question, matches, balanced=enhanced, conversation=True)
            empty_size = len(self._prompt(question, [], language="zh", resolved_question=resolved).encode("utf-8"))
            free_bytes = max(0, byte_budget - fixed_bytes - empty_size - 32)
            # Keep evidence space first, including indivisible structured records.
            first_source = ([{**matches[0], "source_id": "S1"}] if matches and matches[0].get("metadata", {}).get("atomic") else [])
            reserve = (len(self._prompt(question, first_source, language="zh", resolved_question=resolved).encode("utf-8"))
                       - empty_size) if first_source else 512
            prompt_history = bounded_history(state["history"], min(1800, free_bytes // 3, max(0, free_bytes - reserve)))
        sources = self._sources(question, matches, balanced=enhanced, history=prompt_history, resolved_question=resolved)
        history_truncated = bool(state["history"] and not greeting and
                                 (prompt_history != state["history"] or state["rewrite_history_turns"] < len(state["history"])))
        result = {
            "question": question, "answer": INSUFFICIENT_ANSWERS[language], "status": "insufficient_evidence",
            "interaction": "greeting" if greeting else "knowledge",
            "sources": sources, "evidence": [], "warnings": list(retrieved.get("warnings", [])), "reason": "no_sources",
            "answer_policy": "knowledge_base_only",
            "conversation": {"provided_turns": len(state["history"]), "used_turns": len(prompt_history),
                             "rewrite_turns": state["rewrite_history_turns"], "resolved_question": state["resolved_question"],
                             "resolution": state["conversation_resolution"], "truncated": history_truncated},
            "retrieval": {"top_k": 0 if greeting else top_k, "candidate_limit": 0 if greeting else candidate_limit,
                          "scope": "all" if document_id is None else "document",
                          "selection": "diverse" if automatic else "similarity",
                          "returned_chunks": len(candidates) + len(artifact_matches), "used_chunks": len(sources),
                          "excluded_documents": retrieved.get("excluded_documents", 0),
                          "model": retrieved.get("model"),
                          "strategy": {**(retrieved.get("strategy") or {}), "query_language": query_language(question),
                                       "answer_language": language,
                                       "answer_pipeline": "grounded_answer_then_translation" if language == "en" and not greeting else "direct",
                                       "translated_query": translated,
                                       "query_translation": translation_status},
                          "context_types": {kind: sum(s["metadata"].get("context_type") == kind for s in sources)
                                            for kind in ("parent", "neighbor", "child", "summary", "knowledge", "artifact")} if enhanced or artifact_matches else {},
                          "used_sections": len({s["metadata"].get("section_id") for s in sources
                                                if s["metadata"].get("section_id")}) if enhanced else 0},
            "model": {"name": model_name, "version": None},
        }
        if translation_status == "fallback":
            result["warnings"].append("Query translation was unavailable; searched the original question." if language == "en"
                                      else "问题翻译未完成，已使用原问题检索。")
        if history_truncated:
            result["warnings"].append("聊天较长，本轮仅使用近期上下文理解追问。")
        if state["conversation_resolution"] in {"fallback", "history_omitted"}:
            result["warnings"].append("追问补全未完成；如回答偏离问题，请补充所指对象的名称。")
        if len(sources) < len(matches) or any(s["truncated"] for s in sources):
            result["warnings"].append("部分检索内容因上下文预算被截短或省略，回答仅依据本次实际使用的原文片段。")
        if not sources and not greeting:
            logger.info("RAG result: model=%s reason=no_sources", model_name)
        return {"result": result, "prompt_history": prompt_history}

    def _generate_answer(self, state: RagState, runtime: Runtime[RagRunContext]):
        run = runtime.context or RagRunContext()
        check_cancelled(run.cancel)
        model_name = state["model_name"]
        try:
            raw, version = self._generate(state["question"], state["result"]["sources"], model_name,
                                          comprehensive=state["comprehensive"], greeting=state["greeting"],
                                          history=state["prompt_history"],
                                          resolved_question=state["resolved_question"] if state["resolved_question"] != state["question"] else None,
                                          on_event=run.on_event, cancel=run.cancel)
        except OllamaOutputLimitError as exc:
            logger.warning("Local chat model output still reached %s tokens after one compact retry.", exc.num_predict)
            raise DocumentError(
                "generation_truncated",
                f"本地聊天模型 的回答达到输出上限（{self.settings.num_predict} token），"
                "自动尝试简洁回答后仍未完成。请将问题聚焦到具体事件或拆成几个问题；"
                "若配置的 RAG_NUM_PREDICT 小于 2048，可提高至 2048 后重启 API。",
                502,
            ) from exc
        except OllamaError as exc:
            logger.warning("Local chat model unavailable: %s", exc)
            raise DocumentError("generation_unavailable",
                                f"无法使用本地问答模型 {model_name}，请检查所选推理服务、模型安装或生成超时。", 503) from exc
        check_cancelled(run.cancel)
        return {"raw_answer": raw, "model_version": version}

    def _finalize_answer(self, state: RagState, runtime: Runtime[RagRunContext]):
        run = runtime.context or RagRunContext()
        check_cancelled(run.cancel)
        if run.on_event is not None:
            run.on_event("stage", {"stage": "verifying"})
        check_cancelled(run.cancel)
        result = deepcopy(state["result"])
        raw, model_name = state["raw_answer"], state["model_name"]
        result["reason"] = "model_abstained"
        result["model"]["version"] = state["model_version"]
        if raw is None:
            result["reason"] = "out_of_scope"
            logger.info("RAG result: model=%s reason=out_of_scope", model_name)
            return {"result": result}
        decoded = self._decode(raw)
        if not decoded["answerable"]:
            logger.info("RAG result: model=%s reason=model_abstained sources=%s", model_name, len(result["sources"]))
            return {"result": result}
        result.update(answer=public_artifact_text(decoded["answer"].strip(), result['sources'],
                                                language=state['language']), status="answered", reason=None)
        logger.info("RAG result: model=%s status=answered policy=knowledge_base_only", model_name)
        return {"result": result}
