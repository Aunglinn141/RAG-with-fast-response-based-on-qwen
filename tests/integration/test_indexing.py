"""Real Chroma + SQLite + HTTP tests; local deterministic embeddings isolate storage behavior."""

import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from local_rag.api.app import create_app
from local_rag.clients.ollama import OllamaError
from local_rag.core.config import DocumentSettings, VectorSettings, PROJECT_ROOT, RUNTIME_DIR
from local_rag.documents.service import DocumentStore
from tests.fixtures.documents import docx_bytes, pdf_bytes
from tests.fixtures.embeddings import FakeEmbeddings


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.root = RUNTIME_DIR / "tests" / str(uuid4())
        self.root.mkdir(parents=True)
        self.docs = DocumentSettings(storage_dir=self.root / "documents", chunk_size=100, chunk_overlap=20)
        self.config = VectorSettings(chroma_dir=self.root / "chroma",
                                     registry_path=self.root / "registry/documents.sqlite3", batch_size=2)
        self.embedder = FakeEmbeddings()
        self.open_app()

    def open_app(self):
        self.app = create_app(self.docs, self.config, embedder=self.embedder)
        self.service = self.app.state.index_service
        self.client = TestClient(self.app)
        self.client.__enter__()

    def restart(self):
        self.client.__exit__(None, None, None)
        self.open_app()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        if self.root.resolve().parent == (RUNTIME_DIR / "tests").resolve():
            shutil.rmtree(self.root)

    def upload(self, text="apples orchard harvest " * 15, name="fruit.txt"):
        response = self.client.post("/documents", files={"file": (name, text.encode())})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["data"]

    def search(self, query="apples orchard", **fields):
        response = self.client.post("/search", json={"query": query, **fields})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["data"]["matches"]

    def test_upload_register_search_and_full_metadata(self):
        fruit = self.upload()
        self.upload("rockets space moon astronomy " * 10, "space.txt")
        self.assertEqual(fruit["status"], "indexed")
        row = fruit["index"]
        self.assertEqual(row["file_hash"], fruit["sha256"])
        self.assertEqual(row["uploaded_at"], fruit["created_at"])
        self.assertEqual(row["embedding_dimension"], 1024)
        self.assertEqual(row["model_version"], "synthetic-test-v1")
        matches = self.search(top_k=2)
        self.assertEqual(matches[0]["metadata"]["document_id"], fruit["document_id"])
        originals = {chunk["chunk_id"]: chunk for chunk in fruit["chunks"]}
        self.assertEqual(matches[0]["metadata"], originals[matches[0]["chunk_id"]]["metadata"])
        self.assertEqual(self.service.vectors.count_document(fruit["document_id"]), fruit["chunk_count"])
        self.assertTrue((self.root / "chroma/chroma.sqlite3").exists())
        self.assertEqual(self.client.get("/documents").json()["data"]["count"], 2)

    def test_pdf_and_word_metadata_survive_vector_storage(self):
        for name, data, query in [("pages.pdf", pdf_bytes(["apples orchard", "rockets space"]), "rockets space"),
                                  ("guide.docx", docx_bytes(), "Second section")]:
            with self.subTest(name=name):
                response = self.client.post("/documents", files={"file": (name, data)})
                self.assertEqual(response.status_code, 201, response.text)
                record = response.json()["data"]
                matches = self.search(query, document_id=record["document_id"], top_k=1)
                original = next(c for c in record["chunks"] if c["chunk_id"] == matches[0]["chunk_id"])
                self.assertEqual(matches[0]["metadata"], original["metadata"])
                if name.endswith("pdf"):
                    self.assertEqual(matches[0]["metadata"]["page_number"], 2)
                else:
                    self.assertTrue(matches[0]["metadata"]["paragraphs"])

    def test_fresh_process_search_and_delete_persist(self):
        record = self.upload()
        self.client.__exit__(None, None, None)
        script = '''
import json, sys
from pathlib import Path
from fastapi.testclient import TestClient
from local_rag.api.app import create_app
from local_rag.core.config import DocumentSettings, VectorSettings
from tests.fixtures.embeddings import FakeEmbeddings
root = Path(sys.argv[1])
config = VectorSettings(chroma_dir=root/'chroma', registry_path=root/'registry/documents.sqlite3')
with TestClient(create_app(DocumentSettings(storage_dir=root/'documents'), config, embedder=FakeEmbeddings())) as client:
    found = client.post('/search', json={'query':'apples orchard'}).json()["data"]['matches']
    assert found and found[0]['metadata']['document_id'] == sys.argv[2], found
    assert client.delete('/documents/' + sys.argv[2]).status_code == 200
    print(json.dumps({'fresh_process_search': True, 'deleted': True}))
'''
        try:
            result = subprocess.run([sys.executable, "-c", script, str(self.root), record["document_id"]],
                                    cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue(json.loads(result.stdout)["fresh_process_search"])
        finally:
            self.open_app()
        self.assertEqual(self.search(), [])
        self.assertEqual(self.service.vectors.count_document(record["document_id"]), 0)
        self.assertEqual(self.client.get(f"/documents/{record['document_id']}").status_code, 404)

    def test_delete_preserves_other_documents_and_is_idempotent(self):
        removed = self.upload()
        kept = self.upload("rockets space moon", "space.txt")
        url = f"/documents/{removed['document_id']}"
        for _ in range(2):
            response = self.client.delete(url)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["remaining_vectors"], 0)
        self.assertFalse((self.docs.storage_dir / removed["document_id"]).exists())
        self.assertEqual(self.client.get(url + "/original").status_code, 404)
        self.assertEqual({m["metadata"]["document_id"] for m in self.search()}, {kept["document_id"]})
        self.restart()
        self.assertEqual(self.service.registry.get(removed["document_id"])["status"], "deleted")
        self.assertEqual(self.client.get("/documents").json()["data"]["count"], 1)

    def test_reindex_replaces_generations_and_all_model_versions(self):
        record = self.upload()
        document_id = record["document_id"]
        old_name = record["index"]["collection_name"]
        old_ids = self.service.vectors.collection(old_name).get(include=[])["ids"]
        self.embedder.version = "synthetic-test-v2"
        mismatch = self.client.post("/search", json={"query": "apples"})
        self.assertEqual(mismatch.status_code, 409)
        response = self.client.post(f"/documents/{document_id}/reindex")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"]["index"]["model_version"], "synthetic-test-v2")
        self.assertEqual(self.service.vectors.collection(old_name).count(), 0)
        self.assertEqual(self.service.vectors.count_document(document_id), record["chunk_count"])
        response = self.client.post(f"/documents/{document_id}/reindex")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.service.vectors.count_document(document_id), record["chunk_count"])
        new_ids = self.service.vectors.collection(response.json()["data"]["index"]["collection_name"]).get(include=[])["ids"]
        self.assertFalse(set(old_ids) & set(new_ids))
        self.assertTrue(self.search())
        self.client.delete(f"/documents/{document_id}")
        self.assertEqual(self.service.vectors.count_document(document_id), 0)

    def test_partial_batch_failure_is_cleaned_and_can_retry(self):
        real_embed = self.embedder.embed
        def fail_second(texts):
            if self.embedder.calls == 1:
                raise OllamaError("simulated model outage")
            return real_embed(texts)
        with patch.object(self.embedder, "embed", side_effect=fail_second):
            response = self.client.post("/documents", files={"file": ("retry.txt", b"apples orchard " * 100)})
        self.assertEqual(response.status_code, 503, response.text)
        document_id = response.json()["error"]["document_id"]
        self.assertTrue((self.docs.storage_dir / document_id / "original.txt").exists())
        self.assertEqual(self.service.registry.get(document_id)["status"], "failed")
        self.assertEqual(self.service.vectors.count_document(document_id), 0)
        self.assertEqual(self.search(), [])
        self.assertEqual(self.client.post(f"/documents/{document_id}/reindex").status_code, 200)
        self.assertTrue(self.search())

    def test_failed_reindex_hides_and_removes_old_vectors(self):
        record = self.upload()
        document_id = record["document_id"]
        with patch.object(self.embedder, "identity", side_effect=OllamaError("offline")):
            response = self.client.post(f"/documents/{document_id}/reindex")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.service.vectors.count_document(document_id), 0)
        self.assertEqual(self.search(), [])

    def test_wrong_dimension_and_model_change_during_batch_are_rejected(self):
        with patch.object(self.embedder, "embed", return_value=[[1.0, 0.0]]):
            response = self.client.post("/documents", files={"file": ("bad.txt", b"apples")})
        self.assertEqual(response.status_code, 503)
        document_id = response.json()["error"]["document_id"]
        real_embed = self.embedder.embed
        def change_model(texts):
            result = real_embed(texts)
            self.embedder.version = "changed-mid-batch"
            return result
        with patch.object(self.embedder, "embed", side_effect=change_model):
            response = self.client.post(f"/documents/{document_id}/reindex")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.service.vectors.count_document(document_id), 0)

    def test_delete_failure_is_invisible_and_recovered_on_restart(self):
        record = self.upload()
        kept = self.upload("rockets space moon", "kept.txt")
        document_id = record["document_id"]
        with patch.object(self.service.vectors, "delete_document", side_effect=OSError("disk unavailable")):
            response = self.client.delete(f"/documents/{document_id}")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.service.registry.get(document_id)["status"], "delete_failed")
        self.assertGreater(self.service.vectors.count_document(document_id), 0)
        self.assertEqual({m["metadata"]["document_id"] for m in self.search()}, {kept["document_id"]})
        self.restart()
        self.assertEqual(self.service.registry.get(document_id)["status"], "deleted")
        self.assertEqual(self.service.vectors.count_document(document_id), 0)
        self.assertFalse((self.docs.storage_dir / document_id).exists())

    def test_index_cleanup_failure_stays_hidden_until_recovery(self):
        record = self.upload()
        document_id = record["document_id"]
        with patch.object(self.service.vectors, "delete_document", side_effect=OSError("cleanup unavailable")):
            response = self.client.post(f"/documents/{document_id}/reindex")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.service.registry.get(document_id)["status"], "cleanup_required")
        self.assertEqual(self.search(), [])
        self.restart()
        self.assertEqual(self.service.registry.get(document_id)["status"], "failed")
        self.assertEqual(self.service.vectors.count_document(document_id), 0)

    def test_obsolete_generation_cannot_crowd_out_live_results(self):
        record = self.upload("rockets space moon", "space.txt")
        stale = {"chunk_id": record["document_id"] + ":999", "text": "apples orchard",
                 "metadata": {"document_id": record["document_id"], "chunk_index": 999}}
        self.service.vectors.upsert(record["index"]["collection_name"], record["document_id"],
                                    "obsolete-generation", [stale], self.embedder.embed([stale["text"]]))
        matches = self.search(top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertNotEqual(matches[0]["chunk_id"], stale["chunk_id"])
        self.assertEqual(matches[0]["text"], "rockets space moon")
        self.client.delete(f"/documents/{record['document_id']}")
        self.assertEqual(self.service.vectors.count_document(record["document_id"]), 0)

    def test_interrupted_index_recovery_and_step2_migration(self):
        record = self.upload()
        document_id = record["document_id"]
        self.service.registry.update(document_id, status="indexing")
        legacy = DocumentStore(self.docs).ingest(io.BytesIO(b"existing local document"), "legacy.txt", "text/plain")
        self.restart()
        self.assertEqual(self.service.registry.get(document_id)["status"], "failed")
        self.assertEqual(self.service.vectors.count_document(document_id), 0)
        self.assertEqual(self.service.registry.get(legacy["document_id"])["status"], "ready")
        self.assertEqual(self.client.post(f"/documents/{legacy['document_id']}/reindex").status_code, 200)

    def test_search_validation_and_document_filter(self):
        record = self.upload()
        for body in ({"query": " "}, {"query": "apples", "top_k": 0}, {"query": "x" * 4001}):
            self.assertEqual(self.client.post("/search", json=body).status_code, 422)
        self.assertEqual(self.search(document_id=str(uuid4())), [])
        self.assertTrue(self.search(document_id=record["document_id"]))
        self.assertEqual(self.client.delete("/documents/not-a-uuid").status_code, 404)
        self.assertEqual(self.client.post(f"/documents/{uuid4()}/reindex").status_code, 404)


if __name__ == "__main__":
    unittest.main()
