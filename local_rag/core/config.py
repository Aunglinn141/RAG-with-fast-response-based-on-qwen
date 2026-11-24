"""Configuration shared by the API, document pipeline and command line tools."""

from dataclasses import dataclass, field
import math
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435"
DEFAULT_CHAT_MODEL = "qwen3:1.7b"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_VLLM_URL = "http://127.0.0.1:8001/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-1.7B"


def validate_keep_alive(value):
    value = str(value).strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:ms|s|m|h)?", value):
        raise ValueError("OLLAMA_KEEP_ALIVE must be seconds or a duration; -1 keeps models resident.")
    return value

MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/x-markdown", "text/plain"},
    ".text": {"text/plain"}, ".log": {"text/plain"},
    ".json": {"application/json", "text/json", "text/plain"},
    ".html": {"text/html"}, ".htm": {"text/html"},
    ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
    ".tsv": {"text/tab-separated-values", "text/plain"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".xlsm": {"application/vnd.ms-excel.sheet.macroenabled.12"},
    ".xls": {"application/vnd.ms-excel"},
    ".png": {"image/png"}, ".jpg": {"image/jpeg"}, ".jpeg": {"image/jpeg"},
    ".bmp": {"image/bmp", "image/x-ms-bmp"}, ".webp": {"image/webp"},
    ".tif": {"image/tiff"}, ".tiff": {"image/tiff"},
}
SUPPORTED_FILE_MESSAGE = "支持 PDF、DOCX、TXT、MD、Excel（XLSX/XLS/XLSM）、CSV/TSV、HTML、JSON、LOG 及 PNG/JPEG/BMP/WebP/TIFF 图片。"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TABLE_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv"}


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Share the same integer/character limits across uploads and local chunking."""
    if (type(chunk_size) is not int or type(chunk_overlap) is not int
            or not 50 <= chunk_size <= 10000 or not 0 <= chunk_overlap < chunk_size):
        raise ValueError("Require integer 50 <= chunk_size <= 10000 and 0 <= chunk_overlap < chunk_size.")


@dataclass
class DocumentSettings:
    storage_dir: Path = field(default_factory=lambda: RUNTIME_DIR / "documents")
    max_file_bytes: int = 20 * 1024 * 1024
    max_text_characters: int = 2_000_000
    chunk_size: int = 700
    chunk_overlap: int = 100
    # Keys are extensions without dots, e.g. {"pdf": (900, 120)}.
    chunk_overrides: dict[str, tuple[int, int]] = field(default_factory=dict)
    ocr_mode: str = "auto"
    ocr_model_dir: Path = field(default_factory=lambda: RUNTIME_DIR / "ocr" / "models")
    ocr_dpi: int = 150
    ocr_max_pages: int = 100
    ocr_max_pixels: int = 20_000_000
    ocr_threads: int = 2
    spreadsheet_max_cells: int = 500_000
    spreadsheet_max_sheets: int = 100

    def __post_init__(self):
        self.storage_dir = Path(self.storage_dir)
        if not self.storage_dir.is_absolute():
            self.storage_dir = PROJECT_ROOT / self.storage_dir
        self.storage_dir = self.storage_dir.resolve()
        if self.max_file_bytes <= 0 or self.max_text_characters <= 0:
            raise ValueError("File and text size limits must be positive.")
        for size, overlap in [(self.chunk_size, self.chunk_overlap), *self.chunk_overrides.values()]:
            validate_chunk_settings(size, overlap)
        if any(f".{extension}" not in MIME_TYPES for extension in self.chunk_overrides):
            raise ValueError("Chunk overrides must use supported file extensions.")
        self.ocr_model_dir = Path(self.ocr_model_dir)
        if not self.ocr_model_dir.is_absolute():
            self.ocr_model_dir = PROJECT_ROOT / self.ocr_model_dir
        self.ocr_model_dir = self.ocr_model_dir.resolve()
        if self.ocr_mode not in {"auto", "always", "off"}:
            raise ValueError("DOCUMENT_OCR_MODE must be auto, always or off.")
        if not 72 <= self.ocr_dpi <= 300 or not 1 <= self.ocr_threads <= 8:
            raise ValueError("OCR requires 72..300 DPI and 1..8 CPU threads.")
        if any(type(v) is not int or v <= 0 for v in
               [self.ocr_dpi, self.ocr_threads, self.ocr_max_pages, self.ocr_max_pixels,
                self.spreadsheet_max_cells, self.spreadsheet_max_sheets]):
            raise ValueError("OCR and spreadsheet limits must be positive integers.")

    @classmethod
    def from_env(cls, *, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """Explicit CLI chunk values override the corresponding environment value."""
        return cls(
            storage_dir=Path(os.getenv("DOCUMENT_STORAGE_DIR", str(RUNTIME_DIR / "documents"))),
            max_file_bytes=int(os.getenv("DOCUMENT_MAX_FILE_BYTES", str(20 * 1024 * 1024))),
            max_text_characters=int(os.getenv("DOCUMENT_MAX_TEXT_CHARACTERS", "2000000")),
            chunk_size=chunk_size if chunk_size is not None else int(os.getenv("DOCUMENT_CHUNK_SIZE", "700")),
            chunk_overlap=chunk_overlap if chunk_overlap is not None else int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "100")),
            ocr_mode=os.getenv("DOCUMENT_OCR_MODE", "auto").lower(),
            ocr_model_dir=Path(os.getenv("DOCUMENT_OCR_MODEL_DIR", str(RUNTIME_DIR / "ocr" / "models"))),
            ocr_dpi=int(os.getenv("DOCUMENT_OCR_DPI", "150")),
            ocr_max_pages=int(os.getenv("DOCUMENT_OCR_MAX_PAGES", "100")),
            ocr_max_pixels=int(os.getenv("DOCUMENT_OCR_MAX_PIXELS", "20000000")),
            ocr_threads=int(os.getenv("DOCUMENT_OCR_THREADS", "2")),
            spreadsheet_max_cells=int(os.getenv("DOCUMENT_SPREADSHEET_MAX_CELLS", "500000")),
            spreadsheet_max_sheets=int(os.getenv("DOCUMENT_SPREADSHEET_MAX_SHEETS", "100")),
        )


@dataclass
class VectorSettings:
    chroma_dir: Path = field(default_factory=lambda: RUNTIME_DIR / "chroma")
    registry_path: Path = field(default_factory=lambda: RUNTIME_DIR / "registry" / "documents.sqlite3")
    ollama_url: str = DEFAULT_OLLAMA_URL
    model_name: str = DEFAULT_EMBED_MODEL
    batch_size: int = 8
    auto_index: bool = True
    timeout: float = 180
    lock_timeout: float = 30
    keep_alive: str = "-1"

    def __post_init__(self):
        self.keep_alive = validate_keep_alive(self.keep_alive)
        for name in ("chroma_dir", "registry_path"):
            path = Path(getattr(self, name))
            setattr(self, name, (path if path.is_absolute() else PROJECT_ROOT / path).resolve())
        if self.model_name.split(":", 1)[0] != "bge-m3":
            raise ValueError("The vector module requires a local bge-m3 model (e.g. bge-m3:latest).")
        if (type(self.batch_size) is not int or not 1 <= self.batch_size <= 256
                or not math.isfinite(self.timeout) or self.timeout <= 0
                or not math.isfinite(self.lock_timeout) or self.lock_timeout <= 0):
            raise ValueError("Require integer 1 <= batch_size <= 256 and finite positive timeouts.")

    @classmethod
    def from_env(cls):
        enabled = os.getenv("RAG_AUTO_INDEX", "true").lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("RAG_AUTO_INDEX must be true, false, 1 or 0.")
        return cls(
            chroma_dir=Path(os.getenv("CHROMA_STORAGE_DIR", str(RUNTIME_DIR / "chroma"))),
            registry_path=Path(os.getenv("DOCUMENT_REGISTRY_PATH", str(RUNTIME_DIR / "registry/documents.sqlite3"))),
            ollama_url=os.getenv("LOCAL_OLLAMA_URL", DEFAULT_OLLAMA_URL),
            model_name=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "8")),
            auto_index=enabled in {"true", "1"},
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "180")),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "-1"),
        )


def validate_chat_model_name(value: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value.strip()) > 256
            or any(c.isspace() or ord(c) < 32 for c in value.strip()) or "cloud" in value.lower()):
        raise ValueError("Specify an installed local chat model name without whitespace or cloud models.")
    return value.strip()


def env_flag(name, default="true"):
    value = os.getenv(name, default).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true, false, 1 or 0.")
    return value in {"true", "1"}


@dataclass
class RagSettings:
    ollama_url: str = DEFAULT_OLLAMA_URL
    model_name: str = DEFAULT_CHAT_MODEL
    timeout: float = 180
    max_context_characters: int = 6000
    num_ctx: int = 8192
    num_predict: int = 2048
    retrieval_candidates: int = 24
    context_chunks: int = 8
    enhanced_retrieval: bool = True
    parent_child: bool = True
    parent_characters: int = 2100
    neighbor_radius: int = 1
    section_summaries: bool = True
    hybrid_search: bool = True
    entity_retrieval: bool = True
    cross_chapter: bool = True
    keep_alive: str = "-1"
    chat_backend: str = "ollama"
    vllm_url: str = DEFAULT_VLLM_URL
    vllm_enable_thinking: bool = False
    fast_mode: bool = False

    def __post_init__(self):
        if self.chat_backend not in {"ollama", "vllm"}:
            raise ValueError("RAG_CHAT_BACKEND must be ollama or vllm.")
        if type(self.vllm_enable_thinking) is not bool:
            raise ValueError("VLLM_ENABLE_THINKING must be boolean.")
        if type(self.fast_mode) is not bool:
            raise ValueError("RAG_FAST_MODE must be boolean.")
        if any(type(value) is not bool for value in (self.enhanced_retrieval, self.parent_child, self.section_summaries,
                                                                    self.hybrid_search, self.entity_retrieval, self.cross_chapter)):
            raise ValueError("Retrieval feature switches must be boolean.")
        if (type(self.parent_characters) is not int or not 1000 <= self.parent_characters <= 6000
                or type(self.neighbor_radius) is not int or not 0 <= self.neighbor_radius <= 2):
            raise ValueError("Require parent characters 1000..6000 and neighbor radius 0..2.")
        self.keep_alive = validate_keep_alive(self.keep_alive)
        self.model_name = validate_chat_model_name(self.model_name)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("RAG_CHAT_TIMEOUT must be finite and positive.")
        if not 100 <= self.max_context_characters <= 12000:
            raise ValueError("Require 100 <= RAG_MAX_CONTEXT_CHARACTERS <= 12000.")
        if (type(self.num_ctx) is not int or not 4096 <= self.num_ctx <= 32768
                or type(self.num_predict) is not int or not 128 <= self.num_predict <= 2048):
            raise ValueError("Invalid RAG context or output token budget.")
        if (type(self.retrieval_candidates) is not int or type(self.context_chunks) is not int
                or not 1 <= self.context_chunks <= 10
                or not self.context_chunks <= self.retrieval_candidates <= 50):
            raise ValueError("Require 1 <= RAG_CONTEXT_CHUNKS <= 10 and context chunks <= RAG_RETRIEVAL_CANDIDATES <= 50.")

    @classmethod
    def from_env(cls):
        backend = os.getenv("RAG_CHAT_BACKEND", "ollama").strip().lower()
        return cls(
            ollama_url=os.getenv("LOCAL_OLLAMA_URL", DEFAULT_OLLAMA_URL),
            chat_backend=backend,
            vllm_url=os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_URL),
            vllm_enable_thinking=env_flag("VLLM_ENABLE_THINKING", "false"),
            fast_mode=env_flag("RAG_FAST_MODE", "false"),
            model_name=(os.getenv("VLLM_CHAT_MODEL", DEFAULT_VLLM_MODEL) if backend == "vllm"
                        else os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL)),
            timeout=float(os.getenv("RAG_CHAT_TIMEOUT", "180")),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "-1"),
            max_context_characters=int(os.getenv("RAG_MAX_CONTEXT_CHARACTERS", "6000")),
            num_ctx=int(os.getenv("RAG_NUM_CTX", "8192")),
            num_predict=int(os.getenv("RAG_NUM_PREDICT", "2048")),
            retrieval_candidates=int(os.getenv("RAG_RETRIEVAL_CANDIDATES", "24")),
            context_chunks=int(os.getenv("RAG_CONTEXT_CHUNKS", "8")),
            enhanced_retrieval=env_flag("RAG_ENHANCED_RETRIEVAL"),
            parent_child=env_flag("RAG_PARENT_CHILD"),
            parent_characters=int(os.getenv("RAG_PARENT_CHARACTERS", "2100")),
            neighbor_radius=int(os.getenv("RAG_NEIGHBOR_RADIUS", "1")),
            section_summaries=env_flag("RAG_SECTION_SUMMARIES"),
            hybrid_search=env_flag("RAG_HYBRID_SEARCH"),
            entity_retrieval=env_flag("RAG_ENTITY_RETRIEVAL"),
            cross_chapter=env_flag("RAG_CROSS_CHAPTER"),
        )


@dataclass
class RuntimeSettings:
    # Per API process. The first-stage deployment uses one Uvicorn worker.
    max_concurrent_requests: int = 2
    warmup_on_startup: bool = True
    max_admitted_requests: int = 0

    def __post_init__(self):
        if type(self.max_concurrent_requests) is not int or not 1 <= self.max_concurrent_requests <= 4:
            raise ValueError("Require 1 <= RAG_MAX_CONCURRENT_REQUESTS <= 4 (integer).")
        capacity = self.max_admitted_requests
        if type(capacity) is not int or (capacity != 0 and capacity < self.max_concurrent_requests):
            raise ValueError("RAG_MAX_ADMITTED_REQUESTS must be 0 or an integer >= RAG_MAX_CONCURRENT_REQUESTS.")

    @classmethod
    def from_env(cls):
        return cls(max_concurrent_requests=int(os.getenv("RAG_MAX_CONCURRENT_REQUESTS", "2")),
                   max_admitted_requests=int(os.getenv("RAG_MAX_ADMITTED_REQUESTS", "0")),
                   warmup_on_startup=env_flag("RAG_WARMUP_ON_STARTUP"))
