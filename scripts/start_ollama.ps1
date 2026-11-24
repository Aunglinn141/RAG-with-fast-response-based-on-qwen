param(
    [string]$HttpsProxy
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot

# Opt in from the terminal starting this server; leave inherited settings alone otherwise.
if ($PSBoundParameters.ContainsKey('HttpsProxy')) {
    $taskProxyUri = $null
    if (-not [Uri]::TryCreate($HttpsProxy, [UriKind]::Absolute, [ref]$taskProxyUri) -or
        $taskProxyUri.Scheme -notin @('http', 'https') -or
        -not $taskProxyUri.Host) {
        throw 'HttpsProxy must be an HTTP proxy URL, for example http://127.0.0.1:7897'
    }
    $env:HTTPS_PROXY = $taskProxyUri.AbsoluteUri
    $env:NO_PROXY = 'localhost,127.0.0.1,::1'
    Write-Host "Model download proxy: $($taskProxyUri.Scheme)://$($taskProxyUri.Host):$($taskProxyUri.Port)"
}

# Keep this project's model files and server configuration local to this session.
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $taskProjectRoot '.runtime/ollama/models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
# Retain BGE-M3 plus ONE selected chat model; inference remains serial.
$env:OLLAMA_MAX_LOADED_MODELS = '2'
if (-not $env:OLLAMA_KEEP_ALIVE) { $env:OLLAMA_KEEP_ALIVE = '-1' }
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_QUEUE = '1'
New-Item -ItemType Directory -Force -Path $env:OLLAMA_MODELS | Out-Null

Write-Host 'Ollama: http://127.0.0.1:11435 (cloud disabled)'
Write-Host "Models: $env:OLLAMA_MODELS"
Write-Host 'Memory profile: BGE-M3 + 1 chat model (2 loaded), 1 parallel request, 1 queued request'
& ollama serve
exit $LASTEXITCODE
