"""HTTP RAG flow with real isolated storage and explicitly synthetic models."""

import importlib
import os
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from local_rag.clients.ollama import OllamaError, OllamaOutputLimitError
from local_rag.core.config import DocumentSettings, RagSettings, RUNTIME_DIR, VectorSettings
from tests.fixtures.embeddings import FakeEmbeddings
from tests.unit.test_rag import FakeChat, QUOTE


class RagAPITests(unittest.TestCase):
    def setUp(self):
        self.root = RUNTIME_DIR / "tests" / ("rag-api-" + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.clean_storage)
        # Importing app.py creates its compatibility ASGI app. Keep that bootstrap
        # isolated as well when this test is the first importer in the process.
        first_import = "local_rag.api.app" not in sys.modules
        with patch.dict(os.environ, {
            "DOCUMENT_STORAGE_DIR": str(self.root / "bootstrap/documents"),
            "CHROMA_STORAGE_DIR": str(self.root / "bootstrap/chroma"),
            "DOCUMENT_REGISTRY_PATH": str(self.root / "bootstrap/registry.sqlite3"),
        }):
            module = importlib.import_module("local_rag.api.app")
        if first_import:
            module.app.state.index_service.close()
        self.chat = FakeChat()
        self.embedder = FakeEmbeddings()
        self.app = module.create_app(
            DocumentSettings(storage_dir=self.root / "documents"),
            VectorSettings(chroma_dir=self.root / "chroma", registry_path=self.root / "registry.sqlite3"),
            embedder=self.embedder, rag_settings=RagSettings(), chat_client=self.chat,
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def clean_storage(self):
        if self.root.resolve().parent == (RUNTIME_DIR / "tests").resolve():
            shutil.rmtree(self.root)

    def upload(self, filename="桃园记录.txt", text=QUOTE):
        response = self.client.post("/documents", files={"file": (filename, text.encode("utf-8"), "text/plain")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["data"]

    def test_question_retrieves_uploaded_text_and_returns_authentic_sources(self):
        record = self.upload()
        self.upload("其他资料.txt", "这一份文档记载了其他无关内容。")
        response = self.client.post("/ask", json={"question": "  谁在桃园结义？  ",
                                                 "document_id": record["document_id"]})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["data"]
        self.assertEqual(result["question"], "谁在桃园结义？")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["retrieval"]["top_k"], 4)
        originals = {chunk["chunk_id"]: chunk for chunk in record["chunks"]}
        self.assertTrue(result["sources"])
        for source in result["sources"]:
            self.assertEqual(source["metadata"], originals[source["chunk_id"]]["metadata"])
            self.assertEqual(source["metadata"]["document_id"], record["document_id"])
            self.assertIn(source["text"], originals[source["chunk_id"]]["text"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["answer_policy"], "knowledge_base_only")
        self.assertFalse(any(s["cited"] for s in result["sources"]))
        self.assertEqual(self.chat.calls[-1]["model"], "qwen3:1.7b")

    def test_empty_and_deleted_document_scope_never_generates_from_other_docs(self):
        empty = self.client.post("/ask", json={"question": "谁在桃园结义？"})
        self.assertEqual(empty.status_code, 200, empty.text)
        self.assertEqual(empty.json()["data"]["status"], "insufficient_evidence")
        self.assertFalse(self.chat.calls)
        record = self.upload()
        self.upload("保留文档.txt", QUOTE)
        self.assertEqual(self.client.delete(f"/documents/{record['document_id']}").status_code, 200)
        response = self.client.post("/ask", json={"question": "谁在桃园结义？",
                                                 "document_id": record["document_id"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"]["status"], "insufficient_evidence")
        self.assertEqual(response.json()["data"]["sources"], [])
        self.assertFalse(self.chat.calls)

    def test_invalid_requests_do_not_invoke_generation(self):
        invalid = [{}, {"question": ""}, {"question": " \n "}, {"question": "问" * 2001},
                   {"question": "问题", "top_k": 0}, {"question": "问题", "top_k": 11},
                   {"question": "问题", "top_k": True}, {"question": "问题", "top_k": 4.5},
                   {"question": "问题", "top_k": "4"},
                   {"question": "问题", "document_id": "not-a-document-id"}]
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/ask", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse(self.chat.calls)

    def test_qwen_unavailable_has_distinct_error_response(self):
        record = self.upload()
        self.chat.error = OllamaError("本机未安装 qwen3:1.7b")
        response = self.client.post("/ask", json={"question": "谁在桃园结义？",
                                                 "document_id": record["document_id"]})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["error"]["code"], "generation_unavailable")
        self.assertTrue(response.json()["error"]["message"])

    def test_truncated_generation_has_distinct_json_error(self):
        self.upload()
        with patch.object(self.chat, "chat", side_effect=OllamaOutputLimitError(2048)) as chat:
            response = self.client.post("/chat", json={"question": "关公做了些什么？", "top_k": 4})
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(response.status_code, 502)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertIsNone(body["data"])
        self.assertEqual(body["error"]["code"], "generation_truncated")
        self.assertEqual(self.client.get("/health").json()["data"]["rag_num_predict"], 2048)

    def test_project_models_and_explicit_selection(self):
        self.upload()
        catalog = self.client.get("/models")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json()["data"]["default_model"], "qwen3:1.7b")
        self.assertEqual({model["name"] for model in catalog.json()["data"]["models"]},
                         {"qwen3:1.7b", "gemma3:1b"})
        self.assertFalse(self.chat.calls)
        for name in ("gemma3:1b", "qwen3:1.7b"):
            response = self.client.post("/chat", json={"question": "谁在桃园结义？", "model": name})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["model"]["name"], name)
            self.assertEqual(self.chat.calls[-1]["model"], name)
        self.assertEqual(self.app.state.rag_service.settings.model_name, "qwen3:1.7b")

    def test_invalid_selected_model_is_rejected(self):
        self.upload()
        for value in ("missing-model:latest", "bge-m3"):
            response = self.client.post("/chat", json={"question": "谁在桃园结义？", "model": value})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "invalid_chat_model")
        for value in ("", "model with spaces", "remote-cloud", 123):
            response = self.client.post("/chat", json={"question": "谁在桃园结义？", "model": value})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "validation_error")
        self.assertFalse(self.chat.calls)

    def test_model_list_unavailable_is_explicit(self):
        self.chat.error = OllamaError("offline")
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "models_unavailable")

    def test_invalid_model_json_is_not_a_successful_answer(self):
        self.upload()
        self.chat.reply = "我不遵守结构化回答格式。"
        response = self.client.post("/ask", json={"question": "谁在桃园结义？"})
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["error"]["code"], "invalid_answer")


if __name__ == "__main__":
    unittest.main()
