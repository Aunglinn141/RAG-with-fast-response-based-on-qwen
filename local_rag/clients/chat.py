"""Construct the selected chat backend independently of BGE embeddings."""

from local_rag.clients.ollama import OllamaClient
from local_rag.clients.vllm import VLLMClient


def create_chat_client(settings):
    if settings.chat_backend == "vllm":
        return VLLMClient(settings.vllm_url, settings.timeout,
                          enable_thinking=settings.vllm_enable_thinking)
    return OllamaClient(settings.ollama_url, settings.timeout, keep_alive=settings.keep_alive)
