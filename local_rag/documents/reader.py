"""Read a local document with provenance, independently of API/storage/models."""

from pathlib import Path
import stat
from uuid import UUID, uuid4

from local_rag.core.config import DocumentSettings, MIME_TYPES, SUPPORTED_FILE_MESSAGE
from local_rag.core.exceptions import DocumentError
from local_rag.documents.extractors import extract_document
from local_rag.documents.models import ReadDocument


def normalize_filename(filename: str | None) -> str:
    """Retain the display basename of local or browser-supplied filenames."""
    filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename or len(filename) > 255 or any(ord(char) < 32 for char in filename):
        raise DocumentError("invalid_filename", "请提供有效文件名（最多 255 字符）。", 400)
    return filename


class DocumentReader:
    """Extract text and physical PDF pages; never index or persist a document.

    Word and Markdown use logical sections and paragraph positions. They do not
    have stable physical page numbers. Scans use local OCR; tables retain sheet/row positions.
    """

    def __init__(self, settings: DocumentSettings | None = None):
        self.settings = settings if settings is not None else DocumentSettings.from_env()

    def read(self, path: str | Path, *, document_id: str | None = None,
             filename: str | None = None) -> ReadDocument:
        source = Path(path)
        name = normalize_filename(source.name if filename is None else filename)
        extension = Path(name).suffix.lower()
        try:
            identifier = str(uuid4()) if document_id is None else str(UUID(document_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise DocumentError("invalid_document_id", "document_id 必须是有效的 UUID。", 400) from exc

        try:
            info = source.stat()
            if not stat.S_ISREG(info.st_mode):
                raise DocumentError("invalid_document_path", "文档路径必须指向普通文件。", 400)
            if extension not in MIME_TYPES:
                raise DocumentError("unsupported_file_type", SUPPORTED_FILE_MESSAGE, 415)
            if info.st_size == 0:
                raise DocumentError("empty_file", "文档文件为空。", 422)
            if info.st_size > self.settings.max_file_bytes:
                raise DocumentError("file_too_large", f"文件大小超过 {self.settings.max_file_bytes} 字节上限。", 413)
            extraction = extract_document(source, extension, self.settings.max_text_characters, settings=self.settings)
        except FileNotFoundError as exc:
            raise DocumentError("document_file_not_found", "找不到要读取的文档文件。", 404) from exc
        except OSError as exc:
            raise DocumentError("document_read_failed", "无法读取文档文件，请检查文件权限和磁盘状态。", 500) from exc

        return ReadDocument(
            document_id=identifier, filename=name, file_type=extension[1:],
            size_bytes=info.st_size, extraction=extraction,
        )
