"""Small, standard-library-only client for local Ollama inference."""

from __future__ import annotations

import json
import math
import time
from threading import RLock
from functools import wraps
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from local_rag.core.config import DEFAULT_OLLAMA_URL, validate_keep_alive
from local_rag.core.timing import record_model_call
from local_rag.core.cancellation import check_cancelled
from local_rag.clients.errors import (
    ModelError as OllamaError,
    ModelUnavailableError as OllamaModelError,
    ModelOutputLimitError as OllamaOutputLimitError,
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OllamaError("拒绝 HTTP 重定向：请求必须留在本机 Ollama。")


def duration_payload(value):
    value = validate_keep_alive(value)
    # Ollama accepts JSON numbers for seconds; a string such as "-1" lacks a unit.
    if value[-1].isdigit():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Model residency duration must be finite.")
        return number if "." in value else int(value)
    return value


def serialized_chat(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._chat_lock:
            return fn(self, *args, **kwargs)
    return wrapper


def serialized_stream(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._chat_lock:
            yield from fn(self, *args, **kwargs)
    return wrapper


class OllamaClient:
    def __init__(self, base_url=DEFAULT_OLLAMA_URL, timeout=180, *, keep_alive="-1"):
        try:
            url = urlsplit(base_url)
            port = 11435 if url.port is None else url.port
            if (
                url.scheme != "http"
                or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username is not None
                or url.password is not None
                or url.path not in {"", "/"}
                or url.query
                or url.fragment
                or not 1 <= port <= 65535
            ):
                raise ValueError
        except ValueError as exc:
            raise OllamaError("Ollama 地址必须是本机 HTTP 地址，例如 http://127.0.0.1:11435。") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise OllamaError("timeout 必须是有限的正数。")
        # Use a numeric loopback address even when configured with 'localhost'.
        host = "[::1]" if url.hostname == "::1" else "127.0.0.1"
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.keep_alive = duration_payload(keep_alive)
        self._chat_lock = RLock()
        self.active_chat_model = None
        # Ignore system proxies; do not allow a local request to redirect outside.
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError) as exc:
            raise OllamaError(
                f"无法访问 {self.base_url} 或请求超时：{exc}。请检查服务是否启动。"
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError("Ollama 返回了无效 JSON。") from exc
        if not isinstance(result, dict) or result.get("error"):
            raise OllamaError(f"Ollama 返回错误：{result}")
        if path in {"/api/embed", "/api/chat"}:
            record_model_call("embedding" if path == "/api/embed" else "chat",
                              payload["model"], result, time.perf_counter() - started)
        return result

    def loaded_models(self):
        models = self._request("/api/ps").get("models")
        if not isinstance(models, list):
            raise OllamaError("Ollama did not return its running model list.")
        return models

    def _activate_chat_model(self, model):
        canonical = model if ":" in model else model + ":latest"
        if self.active_chat_model and self.active_chat_model != canonical:
            # Release only the previous chat model owned by this client; retain BGE.
            self._request("/api/generate", {"model": self.active_chat_model, "keep_alive": 0})
        self.active_chat_model = canonical

    @serialized_chat
    def warm_chat(self, model, *, num_ctx=8192):
        self.require_local_model(model, "completion")
        self._activate_chat_model(model)
        result = self._request("/api/chat", {"model": model, "messages": [], "stream": False,
                                             "keep_alive": self.keep_alive, "options": {"num_ctx": num_ctx}})
        if result.get("done") is not True:
            raise OllamaError("Chat model warm-up did not complete.")
        return result

    def version(self):
        version = self._request("/api/version").get("version")
        if not isinstance(version, str) or not version.strip():
            raise OllamaError("Ollama 未返回有效版本号。")
        return version

    def model_digest(self, model):
        """Return the installed manifest digest, rather than a mutable tag as version."""
        canonical = model if ":" in model else model + ":latest"
        for item in self._request("/api/tags").get("models", []):
            if item.get("name") == canonical or item.get("model") == canonical:
                digest = item.get("digest")
                if isinstance(digest, str) and digest.strip():
                    return digest
        raise OllamaModelError(f"项目 Ollama 未安装 {canonical} 或缺少模型摘要；请先下载模型。")

    def list_chat_models(self):
        """Inspect installed models without loading weights or contacting another server."""
        installed = self._request("/api/tags").get("models")
        if not isinstance(installed, list):
            raise OllamaError("Ollama 未返回有效的本地模型列表。")
        models, unavailable, seen = [], [], set()
        for item in installed:
            if not isinstance(item, dict):
                raise OllamaError("Ollama 模型列表包含无效记录。")
            name = item.get("name") or item.get("model")
            if not isinstance(name, str) or not name.strip():
                raise OllamaError("Ollama 模型记录缺少名称。")
            if name in seen:
                continue
            seen.add(name)
            try:
                info = self.require_local_model(name, "completion")
            except OllamaError as exc:
                unavailable.append({"name": name, "reason": str(exc)})
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            size = item.get("size")
            models.append({
                "name": name,
                "digest": item.get("digest") if isinstance(item.get("digest"), str) else "",
                "size_bytes": size if type(size) is int and size >= 0 else 0,
                "parameter_size": str(details.get("parameter_size") or ""),
                "quantization": str(details.get("quantization_level") or ""),
                "capabilities": [v for v in info.get("capabilities", []) if isinstance(v, str)],
            })
        return {"models": sorted(models, key=lambda model: model["name"]),
                "unavailable_models": unavailable}

    def require_local_model(self, model, capability):
        """Check metadata before inference, including aliases for remote models."""
        if not isinstance(model, str) or not model.strip() or "cloud" in model.lower():
            raise OllamaModelError("必须指定本地模型，不能使用 cloud 模型。")
        info = self._request("/api/show", {"model": model})
        if info.get("remote_host") or info.get("remote_model"):
            raise OllamaModelError(f"{model} 指向远程模型，已拒绝调用。")
        if capability not in info.get("capabilities", []):
            raise OllamaModelError(f"{model} 不支持所需能力：{capability}。")
        return info

    def _chat_payload(self, model, prompt, *, system_prompt=None, response_format=None,
                      num_ctx=4096, num_predict=256, stream=False):
        if not isinstance(prompt, str) or not prompt.strip():
            raise OllamaError("问题不能为空。")
        info = self.require_local_model(model, "completion")
        self._activate_chat_model(model)
        messages = []
        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise OllamaError("系统提示词不能为空。")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx},
            "keep_alive": self.keep_alive,
        }
        if response_format is not None:
            payload["format"] = response_format
        # Thinking controls vary across local model families.
        thinking = info.get("thinking")
        values = thinking.get("values", []) if isinstance(thinking, dict) else []
        if any(value is False for value in values):
            payload["think"] = False
        elif "minimal" in values:
            payload["think"] = "minimal"
        elif "low" in values:
            payload["think"] = "low"
        return payload

    @serialized_chat
    def chat(self, model, prompt, *, system_prompt=None, response_format=None,
             num_ctx=4096, num_predict=256):
        payload = self._chat_payload(model, prompt, system_prompt=system_prompt,
                                     response_format=response_format, num_ctx=num_ctx, num_predict=num_predict)
        result = self._request("/api/chat", payload)
        message = result.get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if result.get("done") is True and result.get("done_reason") == "length":
            raise OllamaOutputLimitError(num_predict)
        if not isinstance(answer, str) or not answer.strip() or result.get("done") is not True:
            raise OllamaError("问答模型未返回完整、非空的回答。")
        return answer.strip()

    def classify(self, model, prompt, *, system_prompt, response_format, num_ctx=4096, num_predict=1024):
        """Short, non-streaming structured classification with the same local model."""
        return self.chat(model, prompt, system_prompt=system_prompt, response_format=response_format,
                         num_ctx=num_ctx, num_predict=num_predict)

    @serialized_stream
    def chat_stream(self, model, prompt, *, system_prompt=None, response_format=None,
                    num_ctx=4096, num_predict=256, cancel=None):
        """Yield actual content deltas, never thinking fields; require a terminal packet."""
        check_cancelled(cancel)
        payload = self._chat_payload(model, prompt, system_prompt=system_prompt,
                                     response_format=response_format, num_ctx=num_ctx,
                                     num_predict=num_predict, stream=True)
        check_cancelled(cancel)
        request = Request(self.base_url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"})
        started = time.monotonic()
        received, total_bytes = False, 0
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                while True:
                    check_cancelled(cancel)
                    line = response.readline(65537)
                    check_cancelled(cancel)
                    total_bytes += len(line)
                    if time.monotonic() - started > self.timeout:
                        raise OllamaError("本地模型流式回答超时，请稍后重试。")
                    if not line:
                        raise OllamaError("模型连接提前结束，未收到完整回答。")
                    if len(line) > 65536 or total_bytes > 2 * 1024 * 1024:
                        raise OllamaError("模型流式输出超过允许大小。")
                    if not line.strip():
                        continue
                    packet = json.loads(line.decode("utf-8"))
                    if not isinstance(packet, dict) or packet.get("error"):
                        raise OllamaError("Ollama 流式生成失败。")
                    message = packet.get("message", {})
                    if not isinstance(message, dict) or type(packet.get("done")) is not bool:
                        raise OllamaError("Ollama 流式响应格式无效。")
                    content = message.get("content", "")
                    if not isinstance(content, str):
                        raise OllamaError("Ollama 流式内容不是文本。")
                    if content:
                        received = received or bool(content.strip())
                        yield content
                    if packet["done"]:
                        record_model_call("chat", model, packet, time.monotonic() - started)
                        if packet.get("done_reason") == "length":
                            raise OllamaOutputLimitError(num_predict)
                        if not received:
                            raise OllamaError("问答模型未返回非空的回答。")
                        return
        except HTTPError as exc:
            raise OllamaError(f"Ollama HTTP {exc.code}：流式生成请求失败。") from exc
        except (URLError, OSError) as exc:
            raise OllamaError(f"无法访问 {self.base_url} 或流式连接中断、超时。") from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError("Ollama 返回了无效流式 JSON。") from exc

    def embed(self, model, texts, *, keep_alive=None):
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise OllamaError("Embedding 输入必须是非空文本或非空文本列表。")
        self.require_local_model(model, "embedding")
        result = self._request(
            "/api/embed",
            {"model": model, "input": texts, "truncate": False, "keep_alive": self.keep_alive if keep_alive is None else duration_payload(keep_alive)},
        )
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError("Embedding 数量与输入文本数量不一致。")
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise OllamaError("Embedding 返回了空向量或无效向量。")
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in vector
            ) or not any(vector):
                raise OllamaError("Embedding 包含无效数值或为全零向量。")
            if dimension is not None and len(vector) != dimension:
                raise OllamaError("Embedding 向量维度不一致。")
            dimension = len(vector)
        return vectors
