param(
    [string]$Python = 'python',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @()
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskOriginalPythonPath = $env:PYTHONPATH
$taskOriginalPluginAutoload = $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
$taskExitCode = 1

Push-Location -LiteralPath $taskProjectRoot
try {
    # Prefer pytest installed in the selected Python environment. An optional,
    # local fallback can support offline machines without altering that environment.
    & $Python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('pytest') is None)"
    if ($LASTEXITCODE -ne 0) {
        $taskFallback = Join-Path $taskProjectRoot '.runtime\python-tools'
        if (-not (Test-Path -LiteralPath (Join-Path $taskFallback 'pytest\__init__.py'))) {
            throw 'pytest is not installed. Run: python -m pip install -r requirements-dev.txt'
        }
        $env:PYTHONPATH = if ($taskOriginalPythonPath) {
            $taskFallback + [IO.Path]::PathSeparator + $taskOriginalPythonPath
        } else { $taskFallback }
        & $Python -c 'import pytest'
        if ($LASTEXITCODE -ne 0) {
            throw 'The local pytest fallback is incomplete. Install requirements-dev.txt in your Python environment.'
        }
        Write-Host 'Using workspace pytest fallback from .runtime/python-tools.'
    }
    # Tests need no third-party pytest plugins; avoid accidental plugins from a
    # developer's unrelated ML environment and keep the suite serial by default.
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
    $taskReportDir = Join-Path $taskProjectRoot '.runtime\reports'
    New-Item -ItemType Directory -Path $taskReportDir -Force | Out-Null
    $taskReport = Join-Path $taskReportDir 'pytest.xml'
    & $Python -m pytest --junitxml=$taskReport @PytestArgs
    $taskExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $taskOriginalPythonPath
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = $taskOriginalPluginAutoload
    Pop-Location
}
exit $taskExitCode
