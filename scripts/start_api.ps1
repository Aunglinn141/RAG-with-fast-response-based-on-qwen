param(
    [int]$Port = 8000,
    [switch]$Reload,
    [ValidateSet('ollama', 'vllm')]
    [string]$Backend,
    [ValidateNotNullOrEmpty()]
    [string]$ChatModel
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }

# Fail before application startup (and database recovery) if another API owns the port.
$taskPortProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
try {
    $taskPortProbe.Server.ExclusiveAddressUse = $true
    $taskPortProbe.Start()
} catch [System.Net.Sockets.SocketException] {
    $taskSocketError = $_.Exception.GetBaseException().SocketErrorCode
    if ($taskSocketError -eq [System.Net.Sockets.SocketError]::AddressAlreadyInUse) {
        Write-Host "Cannot start API: 127.0.0.1:$Port is already in use."
        Write-Host 'In the terminal running the previous API, press Ctrl+C, then run this script again.'
        Write-Host "The /docs page at port $Port may still belong to the old API until it is restarted."
    } else {
        Write-Host "Cannot start API on 127.0.0.1:$Port (socket error: $taskSocketError)."
        Write-Host 'Check whether this port is occupied or reserved before starting the API.'
    }
    exit 1
} finally {
    $taskPortProbe.Stop()
}

# Explicit selection changes only the chat backend/model, preserving inherited
# storage, embedding and retrieval settings (including isolated demo libraries).
# Omitting these arguments retains the existing environment-based startup flow.
if ($PSBoundParameters.ContainsKey('Backend')) {
    $env:RAG_CHAT_BACKEND = $Backend.ToLowerInvariant()
}
$taskBackend = if ($env:RAG_CHAT_BACKEND) { $env:RAG_CHAT_BACKEND.Trim().ToLowerInvariant() } else { 'ollama' }
if ($taskBackend -notin @('ollama', 'vllm')) { throw 'RAG_CHAT_BACKEND must be ollama or vllm.' }
if ($PSBoundParameters.ContainsKey('ChatModel')) {
    if ($taskBackend -eq 'vllm') { $env:VLLM_CHAT_MODEL = $ChatModel }
    else { $env:OLLAMA_CHAT_MODEL = $ChatModel }
}
$taskChatModel = if ($taskBackend -eq 'vllm') {
    if ($env:VLLM_CHAT_MODEL) { $env:VLLM_CHAT_MODEL } else { 'Qwen/Qwen3-1.7B' }
} else {
    if ($env:OLLAMA_CHAT_MODEL) { $env:OLLAMA_CHAT_MODEL } else { 'qwen3:1.7b' }
}
Write-Host "Chat backend: $taskBackend | Default model: $taskChatModel"
Write-Host 'BGE-M3 still requires project Ollama. The selected chat service must already be running.'

$taskArguments = @('-m', 'uvicorn', 'local_rag.api.app:app', '--host', '127.0.0.1', '--port', "$Port", '--workers', '1')
if ($Reload) { $taskArguments += @('--reload', '--reload-dir', (Join-Path $taskProjectRoot 'local_rag')) }

Push-Location -LiteralPath $taskProjectRoot
try {
    & python @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
