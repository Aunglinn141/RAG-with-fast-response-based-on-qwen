# Load into the current PowerShell session before starting the API:
#   . .\configs\local.example.ps1
# Optional personal copy: configs/local.ps1 (ignored by Git).

$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$env:DOCUMENT_STORAGE_DIR = Join-Path $taskProjectRoot '.runtime/documents'
$env:DOCUMENT_MAX_FILE_BYTES = '20971520'
$env:DOCUMENT_MAX_TEXT_CHARACTERS = '2000000'
$env:DOCUMENT_CHUNK_SIZE = '700'
$env:DOCUMENT_CHUNK_OVERLAP = '100'
$env:LOCAL_OLLAMA_URL = 'http://127.0.0.1:11435'
# Default for requests without model; GUI can select any project-local completion model.
$env:OLLAMA_CHAT_MODEL = 'qwen3:1.7b'
$env:RAG_CHAT_BACKEND = 'ollama' # ollama / vllm; see vllm.example.ps1
$env:RAG_FAST_MODE = 'false' # boolean-only scope check; complete answers unchanged
$env:OLLAMA_EMBED_MODEL = 'bge-m3'
# Keep BGE-M3 and the selected LLM resident; restart Ollama with the project script.
$env:OLLAMA_KEEP_ALIVE = '-1'
$env:RAG_WARMUP_ON_STARTUP = 'true'
$env:CHROMA_STORAGE_DIR = Join-Path $taskProjectRoot '.runtime/chroma'
$env:DOCUMENT_REGISTRY_PATH = Join-Path $taskProjectRoot '.runtime/registry/documents.sqlite3'
$env:EMBED_BATCH_SIZE = '8'
$env:RAG_AUTO_INDEX = 'true'
$env:OLLAMA_TIMEOUT = '180'
$env:RAG_CHAT_TIMEOUT = '180'
$env:RAG_MAX_CONTEXT_CHARACTERS = '6000'
# Automatic all-document Q&A; the GUI does not expose retrieval scope or counts.
$env:RAG_RETRIEVAL_CANDIDATES = '24'
$env:RAG_CONTEXT_CHUNKS = '8'
$env:RAG_NUM_CTX = '8192'
# Output tokens per attempt, including the answer and evidence JSON.
$env:RAG_NUM_PREDICT = '2048'
# Per API process; keep one Uvicorn worker for this local memory profile.
$env:RAG_MAX_CONCURRENT_REQUESTS = '2'
# 0 leaves the FIFO queue uncapped. A positive limit caps running and waiting requests combined.
$env:RAG_MAX_ADMITTED_REQUESTS = '0'

# Scenario-aware retrieval; local extractive summaries need no extra LLM.
$env:RAG_ENHANCED_RETRIEVAL = 'true'
$env:RAG_PARENT_CHILD = 'true'
$env:RAG_PARENT_CHARACTERS = '2100'
$env:RAG_NEIGHBOR_RADIUS = '1'
$env:RAG_SECTION_SUMMARIES = 'true'

# Local OCR: run python -m local_rag.cli.setup_ocr once after installing dependencies.
$env:DOCUMENT_OCR_MODE = 'auto' # auto / always / off
$env:DOCUMENT_OCR_MODEL_DIR = '.runtime/ocr/models'
$env:DOCUMENT_OCR_DPI = '150'
$env:DOCUMENT_OCR_MAX_PAGES = '100'
$env:DOCUMENT_OCR_MAX_PIXELS = '20000000'
$env:DOCUMENT_OCR_THREADS = '2'
$env:DOCUMENT_SPREADSHEET_MAX_CELLS = '500000'
$env:DOCUMENT_SPREADSHEET_MAX_SHEETS = '100'
