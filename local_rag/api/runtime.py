"""Warm existing retrieval resources; never ingest, split or embed documents."""
import logging
from time import perf_counter

from local_rag.clients.ollama import OllamaError
from local_rag.core.config import validate_chat_model_name
from local_rag.core.exceptions import DocumentError
from local_rag.rag.scenarios import plan_retrieval
from local_rag.rag.prompts import generation_prompt, answer_schema
from local_rag.rag.scope import scope_request_options

logger = logging.getLogger(__name__)


class RuntimeWarmup:
    def __init__(self, index, rag):
        self.index, self.rag = index, rag
        self.last = {"status": "not_started"}

    def run(self, model=None):
        model = validate_chat_model_name(model or getattr(self.rag.client, "active_chat_model", None) or self.rag.settings.model_name)
        started = perf_counter()
        self.last = {"status": "warming", "chat_model": model, "embedding_model": self.index.settings.model_name}
        try:
            # Read the persisted collection using one query vector, not document vectors.
            query = "知识库预热"
            settings = self.rag.settings
            if settings.enhanced_retrieval:
                retrieval = self.index.retrieve_context(query, settings.retrieval_candidates, None,
                                                       plan_retrieval(query, settings), settings)
            else:
                retrieval = self.index.search(query, 1)
            if retrieval["model"] is None:
                self.index.embedder.embed_query(query)
            chat_start = perf_counter()
            if settings.chat_backend == "vllm":
                # Compile the actual JSON grammars and prefill reusable system
                # prefixes before accepting a real question. One output token
                # per template is sufficient; truncated warm-up is expected.
                self.rag.client.warm_chat(model, num_ctx=settings.num_ctx,
                                          system_prompt=generation_prompt(), response_format=answer_schema())
                scope = scope_request_options(num_ctx=settings.num_ctx, num_predict=settings.num_predict,
                                              fast=settings.fast_mode)
                scope.pop("num_predict")
                self.rag.client.warm_chat(model, **scope)
            else:
                self.rag.client.warm_chat(model, num_ctx=settings.num_ctx)
            chat_ms = (perf_counter() - chat_start) * 1000
            names = {m.get("name") or m.get("model") for m in self.rag.client.loaded_models()}
            canonical = lambda name: name if ":" in name else name + ":latest"
            if settings.chat_backend == "vllm":
                embedding_names = {m.get("name") or m.get("model")
                                   for m in self.index.embedder.client.loaded_models()}
                both_loaded = model in names and canonical(self.index.settings.model_name) in embedding_names
                names |= embedding_names
            else:
                expected = {canonical(model), canonical(self.index.settings.model_name)}
                both_loaded = expected.issubset(names)
            self.last = {**self.last, "status": "ready" if both_loaded else "partial",
                         "chat_backend": settings.chat_backend,
                         "both_models_loaded": both_loaded, "loaded_models": sorted(n for n in names if n),
                         "indexed_documents": sum(r["status"] == "indexed" for r in self.index.list_documents()),
                         "retrieval_timings": retrieval.get("timings", {}),
                         "chat_warmup_ms": round(chat_ms, 3),
                         "total_ms": round((perf_counter() - started) * 1000, 3),
                         "warnings": [] if both_loaded else [
                             "Embedding and chat are not both resident. Check the selected inference services and available memory."]}
            return self.last
        except Exception as exc:
            self.last = {**self.last, "status": "failed", "total_ms": round((perf_counter() - started) * 1000, 3),
                         "error": str(exc)}
            if isinstance(exc, DocumentError):
                raise
            if isinstance(exc, (OllamaError, ValueError)):
                raise DocumentError("warmup_unavailable", "本地模型预热失败，请检查所选聊天服务、BGE-M3 服务及可用内存。", 503) from exc
            logger.exception("Existing index warm-up failed")
            raise DocumentError("warmup_unavailable", "已有索引预热失败，请检查本地存储与日志。", 503) from exc
