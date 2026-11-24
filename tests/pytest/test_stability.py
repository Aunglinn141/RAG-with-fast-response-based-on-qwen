"""Storage and outage contracts with real HTTP/SQLite/Chroma and synthetic vectors.

FakeEmbeddings validates integrity, not Chinese retrieval quality. The separate
fixed-corpus benchmark must measure actual BGE-M3 and Qwen answer quality.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
from urllib.error import URLError

import pytest
from fastapi.testclient import TestClient

from local_rag.api.app import create_app
from local_rag.clients.ollama import OllamaClient, OllamaError
from local_rag.core.config import DocumentSettings, PROJECT_ROOT, RagSettings, VectorSettings
from local_rag.embeddings.bge import BGEEmbeddings
from tests.fixtures.embeddings import FakeEmbeddings
from tests.fixtures.restart_probe import storage_snapshot
from tests.unit.test_rag import FakeChat, QUOTE


pytestmark = pytest.mark.stability


@pytest.fixture
def indexed_app(isolated_root):
    settings = DocumentSettings(storage_dir=isolated_root / "documents", chunk_size=100, chunk_overlap=20)
    vectors = VectorSettings(chroma_dir=isolated_root / "chroma",
                             registry_path=isolated_root / "registry" / "documents.sqlite3", batch_size=2)
    api = create_app(settings, vectors, embedder=FakeEmbeddings(), chat_client=FakeChat())
    with TestClient(api) as client:
        yield api, client


def upload(client, body, name="桃园记录.txt"):
    response = client.post("/documents", files={"file": (name, body, "text/plain")})
    assert response.status_code == 201, response.text
    return response.json()["data"]


def test_duplicate_uploads_keep_independent_ids_and_delete_only_one(indexed_app):
    api, client = indexed_app
    body = (QUOTE + " apples orchard\n").encode("utf-8")
    first, second = upload(client, body), upload(client, body)
    first_id, second_id = first["document_id"], second["document_id"]
    assert first_id != second_id
    expected_hash = hashlib.sha256(body).hexdigest()
    assert first["sha256"] == second["sha256"] == expected_hash
    assert first["index"]["file_hash"] == second["index"]["file_hash"] == expected_hash
    assert client.get("/documents").json()["data"]["count"] == 2
    assert client.get(first["original_url"]).content == client.get(second["original_url"]).content == body
    assert {chunk["chunk_id"] for chunk in first["chunks"]}.isdisjoint(
        {chunk["chunk_id"] for chunk in second["chunks"]})

    removed = client.delete(f"/documents/{first_id}")
    assert removed.status_code == 200, removed.text
    assert removed.json()["data"]["remaining_vectors"] == 0
    assert api.state.index_service.registry.get(first_id)["status"] == "deleted"
    assert api.state.index_service.vectors.count_document(first_id) == 0
    assert api.state.index_service.vectors.count_document(second_id) == second["chunk_count"]
    found = client.post("/search", json={"query": "apples orchard", "top_k": 4})
    assert found.status_code == 200, found.text
    assert {match["metadata"]["document_id"] for match in found.json()["data"]["matches"]} == {second_id}
    assert client.get(second["original_url"]).content == body


@pytest.mark.parametrize("body,error_code", [
    pytest.param(b"", "empty_file", id="zero-byte-file"),
    pytest.param(b" \r\n\t\n  ", "no_extractable_text", id="only-ascii-whitespace"),
    pytest.param("\u3000\u3000\n\t".encode("utf-8"), "no_extractable_text", id="only-chinese-whitespace"),
])
def test_empty_upload_leaves_no_original_chunks_registry_or_vectors(indexed_app, body, error_code):
    api, client = indexed_app
    response = client.post("/documents", files={"file": ("empty.txt", body, "text/plain")})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == error_code
    index = api.state.index_service
    assert list(index.documents.root.iterdir()) == []  # Includes any abandoned .pending directory.
    assert index.registry.rows(include_deleted=True) == []
    assert index.vectors.managed_collections() == []
    assert client.get("/documents").json()["data"]["count"] == 0


@pytest.fixture
def refused_ollama_url():
    # Reserve an unused loopback port, then release it without ever starting a
    # listener. No running Ollama process is stopped or reconfigured by this test.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    yield f"http://127.0.0.1:{port}"


@pytest.mark.parametrize("stage,expected_code", [
    ("embedding", "embedding_unavailable"),
    ("generation", "generation_unavailable"),
])
def test_real_connection_refusal_returns_503_and_preserves_index(
        indexed_app, refused_ollama_url, stage, expected_code):
    api, client = indexed_app
    record = upload(client, QUOTE.encode("utf-8"))
    index, rag = api.state.index_service, api.state.rag_service
    before = storage_snapshot(index, record["document_id"])
    disconnected = OllamaClient(refused_ollama_url, timeout=3)
    # Verify the operating system refused a real connection, rather than mocking
    # a client exception or allowing a timeout to stand in for a disconnected service.
    with pytest.raises(OllamaError) as error:
        disconnected.version()
    assert isinstance(error.value.__cause__, URLError)
    assert isinstance(error.value.__cause__.reason, ConnectionRefusedError)

    if stage == "embedding":
        index.embedder = BGEEmbeddings(VectorSettings(ollama_url=refused_ollama_url, timeout=3))
    else:
        rag.client = disconnected
    response = client.post("/chat", json={"question": "谁在桃园结义？"})
    assert response.status_code == 503, response.text
    assert response.json()["success"] is False
    assert response.json()["data"] is None
    assert response.json()["error"]["code"] == expected_code
    assert client.get("/health").status_code == 200

    index.embedder, rag.client = FakeEmbeddings(), FakeChat()
    assert storage_snapshot(index, record["document_id"]) == before
    recovered = client.post("/chat", json={"question": "谁在桃园结义？"})
    assert recovered.status_code == 200, recovered.text
    result = recovered.json()["data"]
    assert result["status"] == "answered"
    assert result["evidence"] == []
    assert result["answer_policy"] == "knowledge_base_only"
    assert result["sources"][0]["metadata"]["document_id"] == record["document_id"]


def test_three_fresh_process_restarts_preserve_bytes_vectors_and_deleted_tombstone(isolated_root):
    docs = DocumentSettings(storage_dir=isolated_root / "documents", chunk_size=100, chunk_overlap=20)
    vectors = VectorSettings(chroma_dir=isolated_root / "chroma",
                             registry_path=isolated_root / "registry" / "documents.sqlite3", batch_size=2)
    api = create_app(docs, vectors, embedder=FakeEmbeddings(), chat_client=FakeChat(), rag_settings=RagSettings())
    body = ((QUOTE + " apples orchard\n") * 12).encode("utf-8")
    with TestClient(api) as client:
        record = upload(client, body)
        expected = storage_snapshot(api.state.index_service, record["document_id"])
    assert expected["original_sha256"] == hashlib.sha256(body).hexdigest()
    assert expected["vector_count"] == expected["chunk_count"] > 1
    assert expected["search_chunk_ids"]

    def probe(*options):
        process = subprocess.run(
            [sys.executable, "-m", "tests.fixtures.restart_probe", str(isolated_root), record["document_id"], *options],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert process.returncode == 0, process.stdout + process.stderr
        return json.loads(process.stdout)

    for restart_number in range(1, 4):
        assert probe() == expected, f"Persistent state changed after fresh process restart {restart_number}"

    deleted = probe("--delete")
    assert deleted["registry"]["status"] == "deleted"
    assert deleted["registry"]["file_hash"] == expected["registry"]["file_hash"]
    assert deleted["vector_count"] == 0
    assert deleted["search_chunk_ids"] == deleted["visible_ids"] == []
    assert deleted["document_directory_exists"] is False
    for _ in range(3):
        assert probe() == deleted, "A restart changed or resurrected the deleted document"
