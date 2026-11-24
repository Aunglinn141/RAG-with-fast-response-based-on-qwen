param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [ValidateSet('Api', 'Frontend')][string]$Component = 'Api',
    [int]$ApiPort = 8001,
    [int]$FrontendPort = 5174
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskManifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$taskSavedEnvironment = @{}
$taskExitCode = 1
Push-Location -LiteralPath $taskProjectRoot
try {
    # Validate the frozen corpus and all storage paths before starting an API that can recover databases.
    & python -c 'import sys; from local_rag.evaluation.museum_demo import load_prepared; load_prepared(sys.argv[1])' $taskManifestPath
    if ($LASTEXITCODE -ne 0) { throw 'Demo validation failed; no service was started.' }
    $taskManifest = Get-Content -LiteralPath $taskManifestPath -Encoding UTF8 | ConvertFrom-Json
    $taskConfig = $taskManifest.configuration
    $taskEnvironment = @{ LOCAL_RAG_API_URL = "http://127.0.0.1:$ApiPort" }
    if ($Component -eq 'Api') {
        $taskMappings = @{
            documents = @{
                storage_dir = 'DOCUMENT_STORAGE_DIR'; max_file_bytes = 'DOCUMENT_MAX_FILE_BYTES'
                max_text_characters = 'DOCUMENT_MAX_TEXT_CHARACTERS'; chunk_size = 'DOCUMENT_CHUNK_SIZE'
                chunk_overlap = 'DOCUMENT_CHUNK_OVERLAP'; ocr_mode = 'DOCUMENT_OCR_MODE'
                ocr_model_dir = 'DOCUMENT_OCR_MODEL_DIR'; ocr_dpi = 'DOCUMENT_OCR_DPI'
                ocr_max_pages = 'DOCUMENT_OCR_MAX_PAGES'; ocr_max_pixels = 'DOCUMENT_OCR_MAX_PIXELS'
                ocr_threads = 'DOCUMENT_OCR_THREADS'; spreadsheet_max_cells = 'DOCUMENT_SPREADSHEET_MAX_CELLS'
                spreadsheet_max_sheets = 'DOCUMENT_SPREADSHEET_MAX_SHEETS'
            }
            vectors = @{
                chroma_dir = 'CHROMA_STORAGE_DIR'; registry_path = 'DOCUMENT_REGISTRY_PATH'
                ollama_url = 'LOCAL_OLLAMA_URL'; model_name = 'OLLAMA_EMBED_MODEL'
                batch_size = 'EMBED_BATCH_SIZE'; auto_index = 'RAG_AUTO_INDEX'; timeout = 'OLLAMA_TIMEOUT'
                keep_alive = 'OLLAMA_KEEP_ALIVE'
            }
            rag = @{
                model_name = 'OLLAMA_CHAT_MODEL'; timeout = 'RAG_CHAT_TIMEOUT'
                max_context_characters = 'RAG_MAX_CONTEXT_CHARACTERS'; num_ctx = 'RAG_NUM_CTX'
                num_predict = 'RAG_NUM_PREDICT'; retrieval_candidates = 'RAG_RETRIEVAL_CANDIDATES'
                context_chunks = 'RAG_CONTEXT_CHUNKS'; enhanced_retrieval = 'RAG_ENHANCED_RETRIEVAL'
                parent_child = 'RAG_PARENT_CHILD'; parent_characters = 'RAG_PARENT_CHARACTERS'
                neighbor_radius = 'RAG_NEIGHBOR_RADIUS'; section_summaries = 'RAG_SECTION_SUMMARIES'
                hybrid_search = 'RAG_HYBRID_SEARCH'; entity_retrieval = 'RAG_ENTITY_RETRIEVAL'
                cross_chapter = 'RAG_CROSS_CHAPTER'
            }
        }
        foreach ($taskGroup in $taskMappings.Keys) {
            foreach ($taskField in $taskMappings[$taskGroup].Keys) {
                $taskValue = $taskConfig.$taskGroup.$taskField
                $taskEnvironment[$taskMappings[$taskGroup][$taskField]] = if ($taskValue -is [bool]) {
                    $taskValue.ToString().ToLowerInvariant()
                } else { [string]$taskValue }
            }
        }
        $taskEnvironment['RAG_MAX_CONCURRENT_REQUESTS'] = '1'
        $taskEnvironment['RAG_WARMUP_ON_STARTUP'] = 'true'
    }
    foreach ($taskKey in $taskEnvironment.Keys) {
        $taskSavedEnvironment[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process')
        [Environment]::SetEnvironmentVariable($taskKey, $taskEnvironment[$taskKey], 'Process')
    }
    if ($taskManifest.synthetic) {
        Write-Host 'FICTIONAL MUSEUM DEMO - all collection records are synthetic.'
    } else {
        Write-Host 'MUSEUM DOCUMENT PILOT - answers are based on the supplied secondary compilations.'
    }
    Write-Host "Storage: $($taskManifest.root)"
    if ($Component -eq 'Api') {
        & powershell -NoProfile -File (Join-Path $PSScriptRoot 'start_api.ps1') -Port $ApiPort
    } else {
        & powershell -NoProfile -File (Join-Path $PSScriptRoot 'start_frontend.ps1') -Port $FrontendPort
    }
    $taskExitCode = $LASTEXITCODE
} finally {
    foreach ($taskKey in $taskSavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskKey, $taskSavedEnvironment[$taskKey], 'Process')
    }
    Pop-Location
}
exit $taskExitCode
