"""Concurrent, standard-library client for a local vLLM OpenAI-compatible server."""

from contextlib import contextmanager
import hashlib
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from local_rag.clients.errors import ModelError, ModelUnavailableError, ModelOutputLimitError
from local_rag.core.cancellation import check_cancelled
from local_rag.core.config import DEFAULT_VLLM_URL, validate_chat_model_name
from local_rag.core.timing import record_model_call


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("拒绝 HTTP 重定向：vLLM 请求必须留在本机。")


class VLLMClient:
    # No model-switch lock: vLLM owns residency and continuous batching.
    def __init__(self, base_url=DEFAULT_VLLM_URL, timeout=180, *, enable_thinking=False):
        try:
            url = urlsplit(base_url)
            port = url.port if url.port is not None else 8001
            if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                    or url.username is not None or url.password is not None
                    or url.path not in {"", "/", "/v1", "/v1/"} or url.query or url.fragment
                    or not 1 <= port <= 65535):
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise ModelError("vLLM 地址必须是本机 HTTP 地址，例如 http://127.0.0.1:8001/v1。") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelError("timeout 必须是有限的正数。")
        host = "[::1]" if url.hostname == "::1" else "127.0.0.1"
        self.server_url = f"http://{host}:{port}"
        self.base_url = self.server_url + "/v1"
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    @contextmanager
    def _open(self, path, payload=None, *, stream=False, server_root=False):
        request = Request((self.server_url if server_root else self.base_url) + path,
                          data=None if payload is None else json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Accept": "text/event-stream" if stream else "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                yield response
        except HTTPError as exc:
            # Do not expose server request bodies (which may contain document text).
            raise ModelError(f"vLLM HTTP {exc.code}：请检查模型名称、上下文长度及服务日志。") from exc
        except (URLError, OSError) as exc:
            raise ModelError(f"无法访问 {self.base_url}，或请求中断、超时；请检查 vLLM 服务。") from exc
        except (ValueError, UnicodeError) as exc:
            raise ModelError("vLLM 返回了无效 JSON。") from exc

    def _request(self, path, payload=None, *, server_root=False):
        with self._open(path, payload, server_root=server_root) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ModelError("vLLM 响应超过允许大小。")
            result = json.loads(raw)
        if not isinstance(result, dict) or result.get("error"):
            raise ModelError("vLLM 返回错误或无效响应。")
        return result

    def version(self):
        version = self._request("/version", server_root=True).get("version")
        if not isinstance(version, str) or not version.strip():
            raise ModelError("vLLM 未返回有效版本号。")
        return version

    def _models(self):
        models = self._request("/models").get("data")
        if not isinstance(models, list):
            raise ModelError("vLLM 未返回有效的模型列表。")
        for item in models:
            if not isinstance(item, dict):
                raise ModelError("vLLM 模型记录无效。")
            try:
                validate_chat_model_name(item.get("id"))
            except ValueError as exc:
                raise ModelError("vLLM 模型记录缺少有效名称。") from exc
        return models

    @staticmethod
    def _identity(item):
        # /models exposes metadata, not a weight checksum. Label this honestly.
        # Exclude permission records: some servers regenerate their IDs per read.
        data = {key: item.get(key) for key in ("id", "root", "parent", "max_model_len")}
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return "vllm-metadata:" + digest

    def require_local_model(self, model, capability):
        try:
            model = validate_chat_model_name(model)
        except ValueError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        if capability != "completion":
            raise ModelUnavailableError("此 vLLM 客户端仅用于聊天；Embedding 继续使用 BGE-M3。")
        for item in self._models():
            if item["id"] == model:
                return {**item, "capabilities": ["completion"]}
        raise ModelUnavailableError(f"vLLM 未提供模型 {model}；请检查 --served-model-name。")

    def model_digest(self, model):
        return self._identity(self.require_local_model(model, "completion"))

    def list_chat_models(self):
        models = {item["id"]: item for item in self._models()}
        return {"models": [{"name": name, "digest": self._identity(models[name]),
                            "size_bytes": 0, "parameter_size": "", "quantization": "",
                            "capabilities": ["completion"]} for name in sorted(models)],
                "unavailable_models": []}

    def loaded_models(self):
        return [{"name": item["id"]} for item in self._models()]

    def _payload(self, model, prompt, *, system_prompt=None, response_format=None,
                 num_ctx=8192, num_predict=256, stream=False):
        try:
            model = validate_chat_model_name(model)
        except ValueError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        if not isinstance(prompt, str) or not prompt.strip():
            raise ModelError("问题不能为空。")
        if (type(num_ctx) is not int or type(num_predict) is not int
                or not 0 < num_predict < num_ctx):
            raise ModelError("输出 token 预算必须大于零且小于上下文预算。")
        messages = []
        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise ModelError("系统提示词不能为空。")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "stream": stream,
                   "temperature": 0, "max_tokens": num_predict,
                   "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(response_format, dict):
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "rag_response", "schema": response_format, "strict": True}}
        elif response_format is not None:
            raise ModelError("不支持的回答格式。")
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _choice(packet):
        choices = packet.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1
                or not isinstance(choices[0], dict) or choices[0].get("index") != 0):
            raise ModelError("vLLM 未返回有效的单个回答。")
        return choices[0]

    @staticmethod
    def _record(model, packet, elapsed):
        usage = packet.get("usage") or {}
        tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        # vLLM does not expose Ollama's load/decode durations. Only wall time and
        # the actual token count are available here; do not invent decode speed.
        record_model_call("chat", model, {"eval_count": tokens}, elapsed)

    def chat(self, model, prompt, *, system_prompt=None, response_format=None,
             num_ctx=8192, num_predict=256):
        payload = self._payload(model, prompt, system_prompt=system_prompt,
                                response_format=response_format, num_ctx=num_ctx, num_predict=num_predict)
        started = time.perf_counter()
        packet = self._request("/chat/completions", payload)
        self._record(model, packet, time.perf_counter() - started)
        choice = self._choice(packet)
        if choice.get("finish_reason") == "length":
            raise ModelOutputLimitError(num_predict)
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
            raise ModelError("vLLM 未返回完整、非空的回答。")
        return content.strip()

    def classify(self, model, prompt, **options):
        return self.chat(model, prompt, **options)

    def warm_chat(self, model, *, num_ctx=8192, system_prompt=None, response_format=None):
        info = self.require_local_model(model, "completion")
        maximum = info.get("max_model_len")
        if type(maximum) is int and num_ctx > maximum:
            raise ModelError(f"RAG_NUM_CTX={num_ctx} 超过 vLLM 的 max_model_len={maximum}。")
        # Actually exercise generation; /health alone does not warm inference.
        payload = self._payload(model, "Hi", num_ctx=num_ctx, num_predict=1,
                                system_prompt=system_prompt, response_format=response_format)
        packet = self._request("/chat/completions", payload)
        if self._choice(packet).get("finish_reason") not in {"stop", "length"}:
            raise ModelError("vLLM 预热未完成。")
        return packet

    def chat_stream(self, model, prompt, *, system_prompt=None, response_format=None,
                    num_ctx=8192, num_predict=256, cancel=None):
        check_cancelled(cancel)
        payload = self._payload(model, prompt, system_prompt=system_prompt, response_format=response_format,
                                num_ctx=num_ctx, num_predict=num_predict, stream=True)
        started = time.monotonic()
        total_bytes, received, finished = 0, False, False
        usage_packet, data_lines = {}, []
        with self._open("/chat/completions", payload, stream=True) as response:
            while True:
                check_cancelled(cancel)
                line = response.readline(65537)
                check_cancelled(cancel)
                total_bytes += len(line)
                if time.monotonic() - started > self.timeout:
                    raise ModelError("vLLM 流式回答超时。")
                if not line:
                    raise ModelError("vLLM 连接提前结束，未收到 [DONE]。")
                if len(line) > 65536 or total_bytes > 2 * 1024 * 1024:
                    raise ModelError("vLLM 流式输出超过允许大小。")
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                    continue
                if line or not data_lines:
                    continue  # SSE comments, event/id fields and keepalives
                data, data_lines = "\n".join(data_lines), []
                if data == "[DONE]":
                    if not finished or not received:
                        raise ModelError("vLLM 未返回完整、非空的回答。")
                    self._record(model, usage_packet, time.monotonic() - started)
                    return
                packet = json.loads(data)
                if not isinstance(packet, dict) or packet.get("error"):
                    raise ModelError("vLLM 流式生成失败。")
                if packet.get("usage") is not None:
                    usage_packet = packet
                if packet.get("choices") == [] and packet.get("usage") is not None:
                    continue
                if finished:
                    raise ModelError("vLLM 在回答结束后发送了额外内容。")
                choice = self._choice(packet)
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise ModelError("vLLM 流式响应格式无效。")
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise ModelError("vLLM 流式内容不是文本。")
                if content:
                    received = received or bool(content.strip())
                    yield content
                reason = choice.get("finish_reason")
                if reason == "length":
                    self._record(model, usage_packet, time.monotonic() - started)
                    raise ModelOutputLimitError(num_predict)
                if reason is not None:
                    if reason != "stop":
                        raise ModelError(f"vLLM 未完成回答（{reason}）。")
                    finished = True
