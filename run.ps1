[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Cli = Join-Path $Root ".venv\Scripts\deepbedmap.exe"
$env:UV_CACHE_DIR = Join-Path $Root ".cache\uv"
$env:DEEPBEDMAP_CACHE_DIR = Join-Path $Root ".cache\deepbedmap"

if (-not (Test-Path -LiteralPath $Cli)) {
    Write-Host "No project runtime found; running the one-time setup ..."
    & (Join-Path $Root "setup.ps1") -Accelerator auto
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($CommandArgs.Count -eq 0) {
    $CommandArgs = @("--help")
}

if ($CommandArgs[0] -eq "test") {
    & $Python -c "import pytest" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Adding the project test tools ..."
        $Accelerator = if (Test-Path (Join-Path $Root ".deepbedmap-accelerator")) {
            Get-Content -LiteralPath (Join-Path $Root ".deepbedmap-accelerator") -Raw
        } else { "auto" }
        & (Join-Path $Root "setup.ps1") -Accelerator $Accelerator -Dev
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    [string[]]$Remaining = if ($CommandArgs.Count -gt 1) {
        @($CommandArgs[1..($CommandArgs.Count - 1)])
    } else { @() }
    # Windows virus scanners and crashed test runs can retain handles to a reused
    # pytest base directory. A unique project-local directory makes each run
    # independent without falling back to a user-global Temp folder.
    $TestTemp = Join-Path $Root (".cache\pytest-" + [Guid]::NewGuid().ToString("N"))
    & $Python -m pytest --basetemp $TestTemp @Remaining
    $TestExitCode = $LASTEXITCODE
    if (Test-Path -LiteralPath $TestTemp) {
        $CacheRoot = [IO.Path]::GetFullPath((Join-Path $Root ".cache")).TrimEnd('\')
        $ResolvedTestTemp = [IO.Path]::GetFullPath($TestTemp).TrimEnd('\')
        if ($ResolvedTestTemp.StartsWith("$CacheRoot\pytest-", [StringComparison]::OrdinalIgnoreCase)) {
            try {
                Remove-Item -LiteralPath $ResolvedTestTemp -Recurse -Force
            }
            catch {
                Write-Warning "Could not remove test scratch directory: $ResolvedTestTemp"
            }
        }
    }
    exit $TestExitCode
}

if ($CommandArgs[0] -eq "lab") {
    & $Python -c "import jupyterlab" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Adding JupyterLab ..."
        $Accelerator = if (Test-Path (Join-Path $Root ".deepbedmap-accelerator")) {
            Get-Content -LiteralPath (Join-Path $Root ".deepbedmap-accelerator") -Raw
        } else { "auto" }
        & (Join-Path $Root "setup.ps1") -Accelerator $Accelerator -Lab
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    [string[]]$Remaining = if ($CommandArgs.Count -gt 1) {
        @($CommandArgs[1..($CommandArgs.Count - 1)])
    } else { @() }
    & $Python -m jupyterlab @Remaining
    exit $LASTEXITCODE
}

& $Cli @CommandArgs
exit $LASTEXITCODE
