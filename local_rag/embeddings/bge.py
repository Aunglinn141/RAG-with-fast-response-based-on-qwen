"""BGE-M3 dense embeddings through the existing loopback-only Ollama client."""

from dataclasses import dataclass
import math
from typing import Protocol

from local_rag.clients.ollama import OllamaClient, OllamaError
from local_rag.core.config import VectorSettings


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str
    dimension: int = 1024


class EmbeddingProvider(Protocol):
    def identity(self) -> ModelIdentity: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(vectors, count: int, dimension: int):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise OllamaError("Embedding 数量与文本块数量不一致。")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimension:
            raise OllamaError(f"BGE-M3 向量维度必须为 {dimension}。")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector) or not any(vector):
            raise OllamaError("Embedding 包含无效数值或为全零向量。")


class BGEEmbeddings:
    def __init__(self, settings: VectorSettings):
        self.model = settings.model_name
        self.batch_size = settings.batch_size
        self.keep_alive = settings.keep_alive
        self.client = OllamaClient(settings.ollama_url, timeout=settings.timeout, keep_alive=self.keep_alive)

    def identity(self):
        self.client.require_local_model(self.model, "embedding")
        return ModelIdentity(
            self.model if ":" in self.model else self.model + ":latest",
            self.client.model_digest(self.model),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode chunks in input order using bounded batches; no chat model needed."""
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise OllamaError("文档向量化需要文本列表，且每个文本块都不能为空。")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self.embed(texts[start:start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Encode one question in the same vector space as document chunks."""
        if not isinstance(text, str) or not text.strip():
            raise OllamaError("问题向量化需要非空字符串。")
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compatibility entry point for one non-empty batch."""
        # BGE-M3 uses plain input, without nomic's search_document/search_query prefixes.
        vectors = self.client.embed(self.model, texts, keep_alive=self.keep_alive)
        validate_vectors(vectors, len(texts), 1024)
        return vectors
