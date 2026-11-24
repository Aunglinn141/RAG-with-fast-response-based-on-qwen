param(
    [int]$Port = 5173,
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskFrontendDirectory = Join-Path $taskProjectRoot 'frontend'
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
$taskNodeCommand = Get-Command node -ErrorAction SilentlyContinue
$taskLocalNode = Join-Path $taskProjectRoot '.runtime/node/node.exe'
if ($taskNodeCommand) {
    $taskNodePath = $taskNodeCommand.Source
    $taskNodeVersion = [Version]((& $taskNodePath --version).TrimStart('v'))
    $taskNodeSupported = ($taskNodeVersion.Major -eq 20 -and $taskNodeVersion.Minor -ge 19) -or ($taskNodeVersion.Major -eq 22 -and $taskNodeVersion.Minor -ge 12) -or $taskNodeVersion.Major -gt 22
    if (-not $taskNodeSupported -and (Test-Path -LiteralPath $taskLocalNode)) { $taskNodePath = $taskLocalNode }
} elseif (Test-Path -LiteralPath $taskLocalNode) {
    $taskNodePath = $taskLocalNode
} else {
    throw 'Node.js is required. Install Node 20.19+ or 22.12+.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskFrontendDirectory 'node_modules/vite/bin/vite.js'))) {
    throw 'Frontend dependencies are missing. From the project root, run powershell -NoProfile -File .\scripts\install_frontend.ps1 first.'
}
if ($Preview -and -not (Test-Path -LiteralPath (Join-Path $taskFrontendDirectory 'dist/index.html'))) {
    throw 'The frontend build is missing. Run npm run build from the frontend directory first.'
}

$taskPortProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
try {
    $taskPortProbe.Server.ExclusiveAddressUse = $true
    $taskPortProbe.Start()
} catch [System.Net.Sockets.SocketException] {
    Write-Host "Cannot start GUI: 127.0.0.1:$Port is occupied or unavailable."
    Write-Host 'Use the running GUI, or stop its terminal with Ctrl+C before starting another instance.'
    exit 1
} finally {
    $taskPortProbe.Stop()
}

Write-Host "GUI: http://127.0.0.1:$Port"
Write-Host 'The API and Ollama must remain running for document operations.'
$taskArguments = @((Join-Path $taskFrontendDirectory 'scripts/run-vite.mjs'))
if ($Preview) { $taskArguments += 'preview' }
$taskArguments += @('--host', '127.0.0.1', '--port', "$Port", '--strictPort')
Push-Location -LiteralPath $taskFrontendDirectory
try {
    & $taskNodePath @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
