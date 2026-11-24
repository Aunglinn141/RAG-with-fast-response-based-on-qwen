param([string]$Model = 'bge-m3')

$ErrorActionPreference = 'Stop'
$taskOllamaUrl = 'http://127.0.0.1:11435'

# Check the exact project endpoint before the CLI tries to start the desktop app.
try {
    $taskVersion = Invoke-RestMethod -Uri "$taskOllamaUrl/api/version" -TimeoutSec 5
} catch {
    Write-Host 'Project Ollama is not running on port 11435.'
    Write-Host 'Start it in another terminal and keep that terminal open:'
    Write-Host "  powershell -NoProfile -File `"$PSScriptRoot\start_ollama.ps1`""
    Write-Host 'Then run this script again.'
    exit 1
}

Write-Host "Connected to project Ollama $($taskVersion.version) at $taskOllamaUrl"
$taskPreviousHost = $env:OLLAMA_HOST
try {
    $env:OLLAMA_HOST = '127.0.0.1:11435'
    & ollama pull $Model
    $taskPullExitCode = $LASTEXITCODE
    if ($taskPullExitCode -ne 0) {
        Write-Host 'Model download failed. Check the terminal running start_ollama.ps1 for the server error.'
        Write-Host 'Keep the model files; a later pull can reuse completed download parts.'
        exit $taskPullExitCode
    }
    Write-Host 'Embedding model downloaded. Run the acceptance check:'
    Write-Host "  cd `"$(Split-Path -Parent $PSScriptRoot)`""
    Write-Host '  python -m local_rag.cli.check_ollama --output .runtime/reports/verification.json'
} finally {
    if ($null -eq $taskPreviousHost) {
        Remove-Item Env:OLLAMA_HOST -ErrorAction SilentlyContinue
    } else {
        $env:OLLAMA_HOST = $taskPreviousHost
    }
}
