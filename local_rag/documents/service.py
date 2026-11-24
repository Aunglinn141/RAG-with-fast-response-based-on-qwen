"""Persist originals and processed documents under generated UUID directories."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from local_rag.core.config import DocumentSettings, MIME_TYPES, SUPPORTED_FILE_MESSAGE
from local_rag.core.exceptions import DocumentError
from local_rag.documents.reader import DocumentReader, normalize_filename
from local_rag.documents.splitters import DocumentChunker


class DocumentStore:
    def __init__(self, settings: DocumentSettings):
        self.settings = settings
        self.reader = DocumentReader(settings)
        self.root = Path(settings.storage_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def ingest(self, stream, filename: str | None, content_type: str | None) -> dict:
        # Never use an uploaded filename as a filesystem path.
        filename = normalize_filename(filename)
        extension = Path(filename).suffix.lower()
        if extension not in MIME_TYPES:
            raise DocumentError("unsupported_file_type", SUPPORTED_FILE_MESSAGE, 415)
        content_type = (content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
        if content_type not in MIME_TYPES[extension] | {"application/octet-stream"}:
            raise DocumentError("file_type_mismatch", "文件 MIME 类型与扩展名不匹配。", 415)

        document_id = str(uuid4())
        staging = self.root / f"{document_id}.pending"
        destination = self.root / document_id
        staging.mkdir()
        try:
            original = staging / f"original{extension}"
            size, digest = 0, hashlib.sha256()
            with original.open("xb") as output:
                while data := stream.read(64 * 1024):
                    size += len(data)
                    if size > self.settings.max_file_bytes:
                        raise DocumentError("file_too_large", f"文件大小超过 {self.settings.max_file_bytes} 字节上限。", 413)
                    digest.update(data)
                    output.write(data)
            if size == 0:
                raise DocumentError("empty_file", "上传文件为空。", 422)

            reading = self.reader.read(original, document_id=document_id, filename=filename)
            extraction = reading.extraction
            chunk_size, chunk_overlap = self.settings.chunk_overrides.get(
                extension[1:], (self.settings.chunk_size, self.settings.chunk_overlap)
            )
            chunked = DocumentChunker(chunk_size, chunk_overlap).split(reading)
            text, chunks = chunked.text, chunked.chunks
            record = {
                "document_id": document_id, "filename": filename, "file_type": extension[1:],
                "size_bytes": size, "sha256": digest.hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "ready",
                "page_count": extraction.page_count, "character_count": len(text),
                "chunk_count": len(chunks),
                "chunk_config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                "warnings": extraction.warnings,
                "original_url": f"/documents/{document_id}/original",
                "text_url": f"/documents/{document_id}/text",
                "chunks_url": f"/documents/{document_id}/chunks",
                "chunks": chunks,
            }
            (staging / "text.txt").write_text(text, encoding="utf-8", newline="\n")
            (staging / "document.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Readers see the document only after all files are written successfully.
            staging.rename(destination)
            return record
        finally:
            if staging.exists() and staging.resolve().parent == self.root:
                shutil.rmtree(staging)

    def directory(self, document_id: str) -> Path:
        try:
            canonical = str(UUID(document_id))
        except (ValueError, AttributeError) as exc:
            raise DocumentError("document_not_found", "文档不存在。", 404) from exc
        directory = self.root / canonical
        if not (directory / "document.json").is_file():
            raise DocumentError("document_not_found", "文档不存在。", 404)
        return directory

    def get(self, document_id: str) -> dict:
        return json.loads((self.directory(document_id) / "document.json").read_text(encoding="utf-8"))

    def hierarchy(self, document_id, parent_characters=2100):
        from local_rag.storage.hierarchy import load_hierarchy
        record = self.get(document_id)
        text, structure = load_hierarchy(self.directory(document_id), record, parent_characters)
        return record, text, structure

    def delete(self, document_id: str):
        directory = self.root / str(UUID(document_id))
        # Resolve the absolute target before any recursive deletion; reject linked directories.
        if directory.resolve().parent != self.root or directory.is_symlink() or directory.resolve() != directory:
            raise ValueError("Document deletion target escapes storage directory")
        if directory.exists():
            shutil.rmtree(directory)
