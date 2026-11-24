"""Public JSON/Swagger contracts using isolated storage and synthetic local models."""

import importlib
import os
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from filelock import Timeout

from local_rag.clients.ollama import OllamaError
from local_rag.core.config import DocumentSettings, RagSettings, RUNTIME_DIR, VectorSettings
from tests.fixtures.embeddings import FakeEmbeddings
from tests.unit.test_rag import FakeChat, QUOTE


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.root = RUNTIME_DIR / "tests" / ("api-contract-" + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.clean_storage)
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
        self.index = self.app.state.index_service
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def clean_storage(self):
        if self.root.resolve().parent == (RUNTIME_DIR / "tests").resolve():
            shutil.rmtree(self.root)

    def assert_success(self, response, status=200):
        self.assertEqual(response.status_code, status, response.text)
        self.assertIn("application/json", response.headers["content-type"])
        result = response.json()
        self.assertEqual(set(result), {"success", "data", "error"})
        self.assertIs(result["success"], True)
        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["data"])
        return result["data"]

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertIn("application/json", response.headers["content-type"])
        result = response.json()
        self.assertEqual(set(result), {"success", "data", "error"})
        self.assertIs(result["success"], False)
        self.assertIsNone(result["data"])
        self.assertEqual(result["error"]["code"], code)
        self.assertIsInstance(result["error"]["message"], str)
        self.assertTrue(result["error"]["message"])
        return result["error"]

    def upload(self):
        return self.assert_success(self.client.post("/documents", files={
            "file": ("桃园记录.txt", QUOTE.encode("utf-8"), "text/plain"),
        }), 201)

    def test_six_public_operations_complete_document_lifecycle(self):
        health = self.assert_success(self.client.get("/health"))
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["embedding_model"], "bge-m3")
        self.assertEqual(self.assert_success(self.client.get("/documents"))["count"], 0)
        record = self.upload()
        document_id = record["document_id"]
        self.assertEqual(record["status"], "indexed")
        listed = self.assert_success(self.client.get("/documents"))
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["documents"][0]["document_id"], document_id)
        found = self.assert_success(self.client.post("/search", json={
            "query": "桃园结义", "document_id": document_id,
        }))
        self.assertEqual(found["matches"][0]["metadata"]["document_id"], document_id)
        answer = self.assert_success(self.client.post("/chat", json={
            "question": "谁在桃园结义？", "document_id": document_id,
        }))
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["retrieval"]["top_k"], 4)
        self.assertEqual(answer["sources"][0]["metadata"], record["chunks"][0]["metadata"])
        self.assertEqual(answer["evidence"], [])
        self.assertEqual(answer["answer_policy"], "knowledge_base_only")
        removed = self.assert_success(self.client.delete(f"/documents/{document_id}"))
        self.assertEqual(removed["status"], "deleted")
        self.assertEqual(removed["remaining_vectors"], 0)
        self.assertEqual(self.assert_success(self.client.get("/documents"))["count"], 0)
        empty = self.assert_success(self.client.post("/chat", json={
            "question": "谁在桃园结义？", "document_id": document_id,
        }))
        self.assertEqual(empty["status"], "insufficient_evidence")
        self.assertEqual(empty["sources"], [])


    def test_hybrid_search_mode_preserves_envelope_and_delete_visibility(self):
        record = self.upload()
        body = {"query": "桃园结义", "mode": "hybrid"}
        found = self.assert_success(self.client.post("/search", json=body))
        self.assertTrue(found["matches"])
        self.assertEqual(found["strategy"]["fusion"], "rrf_k60")
        self.assertIn("keyword", found["matches"][0]["metadata"]["retrieval_channels"])
        self.assertEqual(found["matches"][0]["metadata"]["document_id"], record["document_id"])
        self.assert_error(self.client.post("/search", json={"query": "桃园", "mode": "invalid"}), 422, "validation_error")
        self.assert_success(self.client.delete("/documents/" + record["document_id"]))
        self.assertEqual(self.assert_success(self.client.post("/search", json=body))["matches"], [])

    def test_ask_alias_has_same_data_contract_as_chat(self):
        record = self.upload()
        body = {"question": "谁在桃园结义？", "document_id": record["document_id"]}
        primary = self.assert_success(self.client.post("/chat", json=body))
        compatibility = self.assert_success(self.client.post("/ask", json=body))
        self.assertEqual(set(primary["timings"]), set(compatibility["timings"]))
        self.assertGreaterEqual(primary.pop("timings")["total_ms"], 0)
        self.assertGreaterEqual(compatibility.pop("timings")["total_ms"], 0)
        self.assertEqual(primary, compatibility)

    def test_auxiliary_json_routes_and_file_downloads_remain_usable(self):
        record = self.upload()
        document_id = record["document_id"]
        detail = self.assert_success(self.client.get(f"/documents/{document_id}"))
        self.assertEqual(detail, record)
        chunks = self.assert_success(self.client.get(f"/documents/{document_id}/chunks"))
        self.assertEqual(chunks["chunks"], record["chunks"])
        rebuilt = self.assert_success(self.client.post(f"/documents/{document_id}/reindex"))
        self.assertEqual(rebuilt["status"], "indexed")
        for kind in ("original", "text"):
            with self.subTest(kind=kind):
                response = self.client.get(f"/documents/{document_id}/{kind}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, QUOTE.encode("utf-8"))
                self.assertIn("attachment", response.headers["content-disposition"])
        self.assert_error(self.client.get(f"/documents/{uuid4()}/original"), 404, "document_not_found")

    def test_parse_and_upload_errors_share_the_json_envelope(self):
        for name, content, status, code in [
            ("unsupported.exe", b"binary", 415, "unsupported_file_type"),
            ("broken.pdf", b"%PDF-1.7\nnot a valid PDF", 422, "invalid_document"),
            ("blank.txt", b" \n\t", 422, "no_extractable_text"),
        ]:
            with self.subTest(name=name):
                self.assert_error(self.client.post("/documents", files={"file": (name, content)}), status, code)
        self.assert_error(self.client.post("/documents", json={}), 400, "invalid_upload")
        self.assertEqual(self.assert_success(self.client.get("/documents"))["count"], 0)

    def test_validation_and_invalid_json_are_structured_without_raw_input(self):
        secret = "PRIVATE-INPUT-MUST-NOT-BE-ECHOED"
        invalid = [
            ("/chat", {"question": "谁在桃园结义？", "top_k": secret}),
            ("/chat", {"question": " "}),
            ("/chat", {"question": "问题", "document_id": secret}),
            ("/search", {"query": " "}),
        ]
        for route, payload in invalid:
            with self.subTest(route=route, payload=payload):
                response = self.client.post(route, json=payload)
                error = self.assert_error(response, 422, "validation_error")
                self.assertTrue(error["details"])
                for detail in error["details"]:
                    self.assertEqual(set(detail), {"loc", "msg", "type"})
                self.assertNotIn(secret, response.text)
        malformed = self.client.post("/chat", content=b'{"question":', headers={"content-type": "application/json"})
        self.assert_error(malformed, 422, "validation_error")
        self.assertFalse(self.chat.calls)

    def test_missing_models_have_distinct_actionable_errors(self):
        self.upload()
        self.chat.error = OllamaError("local Qwen not installed")
        self.assert_error(self.client.post("/chat", json={"question": "谁在桃园结义？"}), 503, "generation_unavailable")
        with patch.object(self.embedder, "identity", side_effect=OllamaError("embedding model unavailable")):
            self.assert_error(self.client.post("/search", json={"query": "桃园结义"}), 503, "embedding_unavailable")

    def test_retrieval_failure_does_not_expose_storage_internals(self):
        self.upload()
        secret = "PRIVATE-STORAGE-PATH-AND-STACK"
        with patch.object(self.index.vectors, "search", side_effect=OSError(secret)):
            response = self.client.post("/search", json={"query": "桃园结义"})
        self.assert_error(response, 503, "search_failed")
        self.assertNotIn(secret, response.text)

    def test_unknown_route_and_unsupported_method_are_structured(self):
        self.assert_error(self.client.get("/not-an-api-route"), 404, "not_found")
        response = self.client.put("/health")
        self.assert_error(response, 405, "method_not_allowed")
        self.assertIn("GET", response.headers.get("allow", ""))

    def test_unexpected_exception_and_busy_lock_do_not_leak_details(self):
        secret = "PRIVATE-EXCEPTION-DETAILS"
        with patch.object(self.index, "list_documents", side_effect=RuntimeError(secret)):
            response = self.client.get("/documents")
        self.assert_error(response, 500, "internal_error")
        self.assertNotIn(secret, response.text)
        with patch.object(self.index, "list_documents", side_effect=Timeout(secret)):
            response = self.client.get("/documents")
        self.assert_error(response, 503, "index_busy")
        self.assertNotIn(secret, response.text)

    def test_swagger_and_openapi_describe_envelopes_and_upload(self):
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("text/html", docs.headers["content-type"])
        self.assertIn("SwaggerUIBundle", docs.text)
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertIn("openapi", schema)
        self.assertNotIn("success", schema)

        def resolve(value):
            while "$ref" in value:
                value = schema["components"]["schemas"][value["$ref"].split("/")[-1]]
            return value

        for path, method, status in [
            ("/health", "get", "200"), ("/documents", "post", "201"),
            ("/documents", "get", "200"), ("/documents/{document_id}", "delete", "200"),
            ("/search", "post", "200"), ("/chat", "post", "200"),
        ]:
            with self.subTest(path=path, method=method):
                operation = schema["paths"][path][method]
                success = resolve(operation["responses"][status]["content"]["application/json"]["schema"])
                self.assertEqual(set(success["properties"]), {"success", "data", "error"})
                self.assertIn("500", operation["responses"])
        for path in ("/chat", "/search"):
            validation = resolve(schema["paths"][path]["post"]["responses"]["422"]["content"]["application/json"]["schema"])
            self.assertEqual(set(validation["properties"]), {"success", "data", "error"})
        form = schema["paths"]["/documents"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(resolve(form)["properties"]["file"]["format"], "binary")
        self.assertTrue(schema["paths"]["/ask"]["post"]["deprecated"])


if __name__ == "__main__":
    unittest.main()
