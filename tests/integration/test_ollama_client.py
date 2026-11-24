"""Contract checks using a loopback HTTP server; no models or Internet needed."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
import unittest
from unittest.mock import patch

from local_rag.clients.ollama import OllamaClient, OllamaError
from local_rag.core.config import VectorSettings
from local_rag.embeddings.bge import BGEEmbeddings


class LocalServerTests(unittest.TestCase):
    def setUp(self):
        self.responses = {
            "/api/tags": {"models": [{"name": "bge-m3:latest", "digest": "local-test-digest"}]},
            "/api/version": {"version": "test"},
            "/api/show": {"capabilities": ["completion", "embedding", "thinking"], "thinking": {"values": [False, True]}},
            "/api/chat": {"message": {"content": "2"}, "done": True, "done_reason": "stop"},
            "/api/embed": {"embeddings": [[0.5, -0.2], [-0.1, 0.3]]},
        }
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length)) if length else None
                owner.requests.append((self.path, body))
                response = owner.responses[self.path]
                if isinstance(response, tuple):
                    status, response = response
                else:
                    status = 200
                data = json.dumps(response).encode()
                self.send_response(status)
                if status == 302:
                    self.send_header("Location", "/redirect-must-not-be-followed")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_POST = do_GET

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.client = OllamaClient(f"http://127.0.0.1:{self.server.server_port}", timeout=2)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def test_chat_and_batch_embedding_contract(self):
        self.assertEqual(self.client.version(), "test")
        self.assertEqual(self.client.chat("local-model", "1+1?"), "2")
        vectors = self.client.embed("local-embed", ["first", "second"])
        self.assertEqual(len(vectors), 2)
        requests = dict(self.requests)
        self.assertIs(requests["/api/chat"]["stream"], False)
        self.assertIs(requests["/api/chat"]["think"], False)
        self.assertEqual(requests["/api/embed"]["input"], ["first", "second"])

    def test_bge_model_digest_and_dense_embedding_contract(self):
        provider = BGEEmbeddings(VectorSettings(ollama_url=self.client.base_url))
        self.assertEqual(provider.identity().version, "local-test-digest")
        self.responses["/api/embed"] = {"embeddings": [[1.0] + [0.0] * 1023]}
        self.assertEqual(len(provider.embed(["plain BGE input"])[0]), 1024)
        request = dict(self.requests)["/api/embed"]
        self.assertEqual(request["input"], ["plain BGE input"])
        self.assertFalse(request["truncate"])
        self.assertEqual(request["keep_alive"], -1)
        self.responses["/api/tags"] = {"models": []}
        with self.assertRaisesRegex(OllamaError, "未安装"):
            provider.identity()

    def test_document_and_query_embeddings_use_the_same_local_model(self):
        provider = BGEEmbeddings(VectorSettings(ollama_url=self.client.base_url, batch_size=1))
        self.responses["/api/show"] = {"capabilities": ["embedding"]}
        self.responses["/api/embed"] = {"embeddings": [[1.0] + [0.0] * 1023]}
        self.assertEqual(len(provider.embed_documents(["图书馆开放时间", "停车场位置"])), 2)
        self.assertEqual(len(provider.embed_query("图书馆什么时候开放？")), 1024)
        calls = [body for path, body in self.requests if path == "/api/embed"]
        self.assertEqual([body["input"] for body in calls],
                         [["图书馆开放时间"], ["停车场位置"], ["图书馆什么时候开放？"]])
        self.assertTrue(all(body["model"] == "bge-m3" and body["truncate"] is False for body in calls))
        self.assertNotIn("/api/chat", dict(self.requests))

    def test_embedding_cloud_metadata_is_rejected_before_inference(self):
        provider = BGEEmbeddings(VectorSettings(ollama_url=self.client.base_url))
        for field in ("remote_host", "remote_model"):
            for method, value in ((provider.embed_documents, ["document"]),
                                  (provider.embed_query, "question")):
                self.responses["/api/show"] = {"capabilities": ["embedding"], field: "remote"}
                self.requests.clear()
                with self.subTest(field=field, method=method.__name__):
                    with self.assertRaisesRegex(OllamaError, "远程模型"):
                        method(value)
                    self.assertEqual([path for path, _ in self.requests], ["/api/show"])

    def test_embedding_redirect_is_rejected(self):
        self.responses["/api/embed"] = (302, {})
        provider = BGEEmbeddings(VectorSettings(ollama_url=self.client.base_url))
        with self.assertRaisesRegex(OllamaError, "重定向"):
            provider.embed_query("question")
        self.assertEqual([path for path, _ in self.requests], ["/api/show", "/api/embed"])

    def test_embedding_ignores_system_proxy(self):
        self.responses["/api/embed"] = {"embeddings": [[1.0] + [0.0] * 1023]}
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1",
                                     "HTTP_PROXY": "http://127.0.0.1:1",
                                     "no_proxy": "", "NO_PROXY": ""}):
            provider = BGEEmbeddings(VectorSettings(ollama_url=self.client.base_url))
            self.assertEqual(len(provider.embed_query("question")), 1024)

    def test_cloud_alias_is_rejected_before_inference(self):
        self.responses["/api/show"]["remote_host"] = "https://ollama.com"
        with self.assertRaisesRegex(OllamaError, "远程模型"):
            self.client.chat("innocent-looking-alias", "hello")
        self.assertEqual([path for path, _ in self.requests], ["/api/show"])

    def test_chat_model_cannot_silently_replace_embedding_model(self):
        self.responses["/api/show"] = {"capabilities": ["completion"]}
        with self.assertRaisesRegex(OllamaError, "embedding"):
            self.client.embed("chat-only", "text")
        self.assertNotIn("/api/embed", dict(self.requests))

    def test_invalid_embedding_responses_fail(self):
        for vectors in ([], [[1]], [[], []], [[1], [1, 2]], [[0], [1]],
                        [[float("nan")], [1]], [[float("inf")], [1]], [[True], [1]]):
            with self.subTest(vectors=vectors):
                self.responses["/api/embed"] = {"embeddings": vectors}
                with self.assertRaises(OllamaError):
                    self.client.embed("local-embed", ["first", "second"])

    def test_incomplete_or_truncated_chat_fails(self):
        for response in ({"message": {"content": ""}, "done": True},
                         {"message": {"content": "2"}, "done": False},
                         {"message": {"content": "2"}, "done": True, "done_reason": "length"}):
            with self.subTest(response=response):
                self.responses["/api/chat"] = response
                with self.assertRaises(OllamaError):
                    self.client.chat("local-model", "hello")

    def test_missing_model_is_actionable_error(self):
        self.responses["/api/show"] = (404, {"error": "model not found"})
        with self.assertRaisesRegex(OllamaError, "HTTP 404.*model not found"):
            self.client.embed("absent-model", "text")

    def test_redirect_is_never_followed(self):
        self.responses["/api/version"] = (302, {})
        with self.assertRaisesRegex(OllamaError, "重定向"):
            self.client.version()
        self.assertEqual([path for path, _ in self.requests], ["/api/version"])

    def test_system_proxy_is_ignored(self):
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1",
                                     "HTTP_PROXY": "http://127.0.0.1:1",
                                     "no_proxy": "", "NO_PROXY": ""}):
            client = OllamaClient(self.client.base_url, timeout=2)
            self.assertEqual(client.version(), "test")


class ConfigurationTests(unittest.TestCase):
    def test_external_or_ambiguous_urls_are_rejected(self):
        for url in ("https://ollama.com", "http://192.168.1.2:11434",
                    "http://localhost.evil.test", "http://user@localhost",
                    "http://localhost/api", "http://localhost?target=external",
                    "http://localhost:0", "http://localhost:99999"):
            with self.subTest(url=url), self.assertRaises(OllamaError):
                OllamaClient(url)

    def test_localhost_is_normalized_without_dns(self):
        self.assertEqual(OllamaClient("http://localhost:11434").base_url,
                         "http://127.0.0.1:11434")

    def test_invalid_timeout_is_rejected(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(OllamaError):
                OllamaClient(timeout=timeout)


if __name__ == "__main__":
    unittest.main()
