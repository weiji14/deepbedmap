[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Accelerator = "auto",
    [switch]$Dev,
    [switch]$Lab,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$Tools = Join-Path $Root ".tools"
$Uv = Join-Path $Tools "uv.exe"
$Venv = Join-Path $Root ".venv"
$UvVersion = "0.11.29"
$UvArchiveName = "uv-x86_64-pc-windows-msvc.zip"
$UvArchiveSha256 = "a047d55651bc3e0ca24595b25ec4cfcb10f9dca9fb56514e661269b37d4fae68"
$UvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvArchiveName"

function Invoke-PythonDownload {
    param([string]$Url, [string]$Destination)

    $Script = @'
import pathlib, sys, urllib.request
request = urllib.request.Request(sys.argv[1], headers={"User-Agent": "DeepBedMap bootstrap"})
with urllib.request.urlopen(request, timeout=90) as response:
    pathlib.Path(sys.argv[2]).write_bytes(response.read())
'@
    $Candidates = @(
        @{ Command = "py"; Prefix = @("-3") },
        @{ Command = "python"; Prefix = @() }
    )
    foreach ($Candidate in $Candidates) {
        if (-not (Get-Command $Candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }
        try {
            & $Candidate.Command @($Candidate.Prefix) -c $Script $Url $Destination
            if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $Destination)) {
                return $true
            }
        }
        catch {
            continue
        }
    }
    return $false
}

function Install-ProjectUv {
    if (Test-Path -LiteralPath $Uv) {
        return
    }
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -notin @("AMD64", "x86_64")) {
        throw "This bootstrap currently supports x86-64 Windows. Install uv manually for $env:PROCESSOR_ARCHITECTURE."
    }
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    $Archive = Join-Path $Tools $UvArchiveName
    Write-Host "Downloading project-local uv $UvVersion ..."
    $Downloaded = Invoke-PythonDownload -Url $UvUrl -Destination $Archive
    if (-not $Downloaded) {
        try {
            Start-BitsTransfer -Source $UvUrl -Destination $Archive -ErrorAction Stop
            $Downloaded = $true
        }
        catch {
            try {
                Invoke-WebRequest -Uri $UvUrl -OutFile $Archive -UseBasicParsing
                $Downloaded = $true
            }
            catch {
                throw "Unable to download uv. Configure HTTPS access to github.com and retry."
            }
        }
    }
    $ActualHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $UvArchiveSha256) {
        Remove-Item -LiteralPath $Archive -Force
        throw "uv archive checksum mismatch: $ActualHash"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Tools -Force
    Remove-Item -LiteralPath $Archive -Force
    if (-not (Test-Path -LiteralPath $Uv)) {
        throw "uv archive did not contain uv.exe"
    }
}

Install-ProjectUv

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_CACHE_DIR = Join-Path $Root ".cache\uv"
$env:UV_PROJECT_ENVIRONMENT = $Venv
$env:DEEPBEDMAP_CACHE_DIR = Join-Path $Root ".cache\deepbedmap"

$RequestedAccelerator = $Accelerator
if ($Accelerator -eq "auto") {
    $Accelerator = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $Accelerator = "cuda"
        }
    }
}

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    $ResolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $ResolvedVenv = [IO.Path]::GetFullPath($Venv).TrimEnd('\')
    if (-not $ResolvedVenv.StartsWith("$ResolvedRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a virtual environment outside the repository: $ResolvedVenv"
    }
    Remove-Item -LiteralPath $ResolvedVenv -Recurse -Force
}

Write-Host "Preparing project-local Python 3.12 ..."
& $Uv python install 3.12 --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.12" }

function Sync-ProjectRuntime {
    param([ValidateSet("cpu", "cuda")][string]$SelectedAccelerator)

    $SyncArguments = @("sync", "--frozen", "--extra", $SelectedAccelerator)
    if ($Lab) {
        $SyncArguments += @("--extra", "lab")
    }
    if (-not $Dev) {
        $SyncArguments += "--no-dev"
    }

    Write-Host "Installing the $SelectedAccelerator runtime into .venv ..."
    Push-Location $Root
    try {
        & $Uv @SyncArguments
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
    }
    finally {
        Pop-Location
    }
}

$Cli = Join-Path $Venv "Scripts\deepbedmap.exe"
function Test-InstalledRuntime {
    param([ValidateSet("cpu", "cuda")][string]$SelectedAccelerator)

    & $Cli doctor | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    Write-Host "Validating a real $SelectedAccelerator model forward pass ..."
    & $Cli smoke --device $SelectedAccelerator --random | Out-Host
    return ($LASTEXITCODE -eq 0)
}

$Marker = Join-Path $Root ".deepbedmap-accelerator"
$PreviousAccelerator = $null
if (-not $Recreate -and (Test-Path -LiteralPath $Marker)) {
    $MarkerValue = (Get-Content -LiteralPath $Marker -Raw).Trim()
    if ($MarkerValue -in @("cpu", "cuda")) {
        $PreviousAccelerator = $MarkerValue
    }
}

try {
    Sync-ProjectRuntime -SelectedAccelerator $Accelerator
    if (-not (Test-Path -LiteralPath $Cli)) {
        throw "Installation completed without creating deepbedmap.exe"
    }

    if (-not (Test-InstalledRuntime -SelectedAccelerator $Accelerator)) {
        if ($RequestedAccelerator -eq "auto" -and $Accelerator -eq "cuda") {
            Write-Warning "The NVIDIA driver was visible, but the CUDA runtime probe failed; falling back to CPU."
            $Accelerator = "cpu"
            Sync-ProjectRuntime -SelectedAccelerator $Accelerator
            if (-not (Test-InstalledRuntime -SelectedAccelerator $Accelerator)) {
                throw "The fallback CPU runtime failed its model forward-pass check"
            }
        }
        else {
            throw "The installed $Accelerator runtime failed its model forward-pass check"
        }
    }
}
catch {
    $InstallFailure = $_
    if ($null -ne $PreviousAccelerator -and $PreviousAccelerator -ne $Accelerator) {
        Write-Warning "Setup failed; restoring the previous $PreviousAccelerator runtime."
        try {
            Sync-ProjectRuntime -SelectedAccelerator $PreviousAccelerator
            if (Test-InstalledRuntime -SelectedAccelerator $PreviousAccelerator) {
                Set-Content -LiteralPath $Marker -Value $PreviousAccelerator -NoNewline
                Write-Warning "The previous $PreviousAccelerator runtime was restored successfully."
            }
        }
        catch {
            Write-Warning "Automatic runtime restoration also failed; rerun setup.ps1."
        }
    }
    throw $InstallFailure
}

Set-Content -LiteralPath $Marker -Value $Accelerator -NoNewline
Write-Host ""
Write-Host "DeepBedMap is ready ($Accelerator)."
Write-Host ""
Write-Host "Next: .\run.ps1 smoke"
