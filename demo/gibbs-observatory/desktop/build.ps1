# Build the Thermodynamic Workbench into a distributable folder.
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File demo\gibbs-observatory\desktop\build.ps1
#
# This script exists because PyInstaller cannot be trusted to report its own
# failure. A build was observed to fail outright -- it could not delete the
# previous dist/ because a running copy of the application held
# _internal\clr_loader\ffi\dlls\amd64\ClrLoader.dll open -- and STILL EXIT 0,
# leaving the previous binary in place. The stale binary then self-tested
# clean, because it was a working build of older source. A failed build that
# reports success and leaves a plausible artifact is the worst outcome
# available here, so every gate below is checked explicitly.
$ErrorActionPreference = "Stop"

$Desktop = $PSScriptRoot
$Root = Split-Path -Parent $Desktop
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Dist = Join-Path $Root "dist"
$Work = Join-Path $Root "build\workbench-work"
$AppDir = Join-Path $Dist "ThermodynamicWorkbench"
$Exe = Join-Path $AppDir "ThermodynamicWorkbench.exe"

if (-not (Test-Path $Py)) { throw "no interpreter at $Py" }

# --- Gate 1: nothing may hold the output open -------------------------------
$running = Get-Process -Name "ThermodynamicWorkbench" -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "==> stopping $($running.Count) running instance(s)"
    $running | Stop-Process -Force
    Start-Sleep -Seconds 2
    if (Get-Process -Name "ThermodynamicWorkbench" -ErrorAction SilentlyContinue) {
        throw "a copy of the application is still running and will make this build fail silently. Close it."
    }
}

# --- Frontend ---------------------------------------------------------------
Write-Host "==> building frontend"
Push-Location (Join-Path $Root "frontend")
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
} finally { Pop-Location }

$index = Join-Path $Root "frontend\dist\index.html"
if (-not (Test-Path $index)) { throw "frontend build produced no index.html at $index" }

# --- Freeze -----------------------------------------------------------------
# Record the newest source timestamp BEFORE building, so a build that silently
# does nothing cannot pass as fresh.
$sources = Get-ChildItem -Path $Desktop, (Join-Path $Root "backend") -Recurse -Include *.py, *.spec -ErrorAction SilentlyContinue
$newestSource = ($sources | Measure-Object -Property LastWriteTime -Maximum).Maximum
Write-Host "==> newest source: $newestSource"

Write-Host "==> freezing application"
& $Py -m PyInstaller --noconfirm --clean `
    --distpath $Dist --workpath $Work (Join-Path $Desktop "workbench.spec")
$pyiExit = $LASTEXITCODE

# --- Gate 2: the binary must exist AND be newer than the source -------------
if (-not (Test-Path $Exe)) { throw "no executable at $Exe (PyInstaller exit $pyiExit)" }

$built = (Get-Item $Exe).LastWriteTime
if ($built -lt $newestSource) {
    throw ("STALE BUILD. $Exe was written $built but source changed $newestSource. " +
           "PyInstaller reported exit $pyiExit without rebuilding -- typically because " +
           "something held a file in dist/ open. The binary on disk is NOT this source.")
}
Write-Host "==> binary written $built (newer than source: ok)"

# --- Gate 3: the frozen binary must pass its own self-test ------------------
Write-Host "==> self-testing the frozen build"
& $Exe --selftest
if ($LASTEXITCODE -ne 0) { throw "the frozen build failed its own self-test (exit $LASTEXITCODE)" }

# --- Report -----------------------------------------------------------------
$size = (Get-ChildItem $AppDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host ("==> built: {0}" -f $AppDir)
Write-Host ("==> size:  {0:N0} MB" -f $size)
Write-Host "==> ship the FOLDER, not the exe alone. The exe needs _internal\ beside it."
