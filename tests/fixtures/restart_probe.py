"""Fresh-process storage probe using FakeEmbeddings, never a live model.

Only accepts a direct child of .runtime/tests, so subprocess checks cannot target
the user's real knowledge base. Run via python -m tests.fixtures.restart_probe.
"""

import argparse
import hashlib
import json
from pathlib import Path

from local_rag.core.config import DocumentSettings, PROJECT_ROOT, VectorSettings
from local_rag.documents.service import DocumentStore
from local_rag.indexing.service import IndexService
from tests.fixtures.embeddings import FakeEmbeddings


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def storage_snapshot(index, document_id):
    """Compare actual on-disk bytes, registry rows and stored vectors on restart."""
    row = index.registry.get(document_id)
    assert row is not None, document_id
    result = {
        "registry": row,
        "vector_count": index.vectors.count_document(document_id),
        "visible_ids": sorted(item["document_id"] for item in index.list_documents()),
        "search_chunk_ids": sorted(item["chunk_id"] for item in index.search(
            "桃园结义 apples orchard", limit=max(row["chunk_count"], 1))["matches"]),
        "document_directory_exists": (index.documents.root / document_id).exists(),
        "embedding_kind": "FakeEmbeddings: synthetic-test-v1 (no live model)",
    }
    if row["status"] == "deleted":
        return result

    record = index.documents.get(document_id)
    directory = index.documents.directory(document_id)
    original = directory / f"original.{record['file_type']}"
    stored = index.vectors.collection(row["collection_name"]).get(
        where={"document_id": document_id}, include=["embeddings", "documents", "metadatas"])
    vectors = stored["embeddings"].tolist()
    ordered_vectors = sorted(
        zip(stored["ids"], stored["documents"], stored["metadatas"], vectors), key=lambda entry: entry[0])
    result.update(
        original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        document_json_sha256=hashlib.sha256((directory / "document.json").read_bytes()).hexdigest(),
        text_sha256=hashlib.sha256((directory / "text.txt").read_bytes()).hexdigest(),
        chunks_sha256=canonical_hash(record["chunks"]),
        vectors_sha256=canonical_hash(ordered_vectors),
        chunk_count=record["chunk_count"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("document_id")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    allowed_parent = (PROJECT_ROOT / ".runtime" / "tests").resolve()
    if root.parent != allowed_parent or args.root.is_symlink():
        parser.error("probe root must be an isolated direct child of .runtime/tests")
    index = IndexService(
        DocumentStore(DocumentSettings(storage_dir=root / "documents", chunk_size=100, chunk_overlap=20)),
        VectorSettings(chroma_dir=root / "chroma", registry_path=root / "registry" / "documents.sqlite3"),
        FakeEmbeddings(),
    )
    try:
        if args.delete:
            index.delete(args.document_id)
        print(json.dumps(storage_snapshot(index, args.document_id), ensure_ascii=False))
    finally:
        index.close()


if __name__ == "__main__":
    main()
