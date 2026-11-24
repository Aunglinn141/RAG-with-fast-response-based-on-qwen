param(
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$taskFrontendDirectory = Join-Path $taskProjectRoot 'frontend'
$taskModulesDirectory = Join-Path $taskFrontendDirectory 'node_modules'
$taskCacheDirectory = Join-Path $taskProjectRoot '.runtime/npm'

# npm ci replaces node_modules. Verify its target before running any install.
if (-not (Test-Path -LiteralPath (Join-Path $taskFrontendDirectory 'package-lock.json'))) {
    throw 'frontend/package-lock.json is missing. Run this script from a complete project checkout.'
}
foreach ($taskDirectory in @($taskFrontendDirectory, $taskModulesDirectory)) {
    if (Test-Path -LiteralPath $taskDirectory) {
        $taskItem = Get-Item -LiteralPath $taskDirectory -Force
        if (($taskItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing npm ci through a linked directory: $taskDirectory"
        }
        if (-not $taskItem.FullName.StartsWith($taskProjectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Install target is outside this project: $taskDirectory"
        }
    }
}

function Test-CompatibleNode([string]$Executable) {
    $taskVersionText = & $Executable --version
    if ($LASTEXITCODE -ne 0) { return $false }
    $taskVersion = [Version]($taskVersionText.TrimStart('v'))
    return ($taskVersion.Major -eq 20 -and $taskVersion.Minor -ge 19) -or
        ($taskVersion.Major -eq 22 -and $taskVersion.Minor -ge 12) -or
        $taskVersion.Major -gt 22
}

$taskNodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
$taskNodePath = $null
if ($taskNodeCommand -and (Test-CompatibleNode $taskNodeCommand.Source)) {
    $taskNodePath = $taskNodeCommand.Source
}
if (-not $taskNodePath) {
    $taskLocalNode = Join-Path $taskProjectRoot '.runtime/node/node.exe'
    if ((Test-Path -LiteralPath $taskLocalNode) -and (Test-CompatibleNode $taskLocalNode)) {
        $taskNodePath = $taskLocalNode
    }
}
if (-not $taskNodePath) {
    throw 'A compatible Node runtime is required (20.19+ or 22.12+). Install it before installing frontend dependencies.'
}

# Invoke npm's JS entry with the selected Node. npm.cmd may pin the old system Node.
$taskNpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $taskNpmCommand) { $taskNpmCommand = Get-Command npm -ErrorAction SilentlyContinue }
$taskNpmCandidates = @((Join-Path (Split-Path -Parent $taskNodePath) 'node_modules/npm/bin/npm-cli.js'))
if ($taskNpmCommand) {
    $taskNpmCandidates += Join-Path (Split-Path -Parent $taskNpmCommand.Source) 'node_modules/npm/bin/npm-cli.js'
}
if ($taskNodeCommand) {
    $taskNpmCandidates += Join-Path (Split-Path -Parent $taskNodeCommand.Source) 'node_modules/npm/bin/npm-cli.js'
}
$taskNpmCli = $taskNpmCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $taskNpmCli) { throw 'npm is missing. Install Node.js with npm and retry.' }

# A loaded native module cannot be replaced on Windows. Fail before npm removes files.
if (Test-Path -LiteralPath $taskModulesDirectory) {
    foreach ($taskNativeFile in Get-ChildItem -LiteralPath $taskModulesDirectory -Filter '*.node' -File -Recurse) {
        try {
            $taskHandle = [System.IO.File]::Open($taskNativeFile.FullName, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            $taskHandle.Dispose()
        } catch {
            Write-Host "Cannot install: a native dependency is in use or is not writable: $($taskNativeFile.FullName)"
            Write-Host 'In this project GUI terminal, press Ctrl+C to stop Vite/dev/preview. Wait for any frontend build to finish, then retry.'
            Write-Host 'Dependency installation has not started. Do not run npm ci while the frontend is running.'
            exit 1
        }
    }
}

Write-Host "Node: $(& $taskNodePath --version) ($taskNodePath)"
Write-Host "npm: $(& $taskNodePath $taskNpmCli --version)"
Write-Host "Frontend: $taskFrontendDirectory"
Write-Host "npm cache: $taskCacheDirectory"
$taskArguments = @($taskNpmCli, 'ci', '--cache', $taskCacheDirectory, '--no-audit', '--no-fund')
if ($Offline) { $taskArguments += '--offline' }
$taskPreviousPath = $env:PATH
Push-Location -LiteralPath $taskFrontendDirectory
try {
    # Lifecycle commands must use the same runtime as npm itself.
    $env:PATH = (Split-Path -Parent $taskNodePath) + [System.IO.Path]::PathSeparator + $taskPreviousPath
    & $taskNodePath @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $taskPreviousPath
    Pop-Location
}
if ($taskExitCode -eq 0) {
    Write-Host 'Frontend dependencies installed. Start the GUI with scripts/start_frontend.ps1.'
} elseif ($Offline) {
    Write-Host 'Offline installation failed. If a package is missing from the project npm cache, retry without -Offline when the download network is available.'
}
exit $taskExitCode
