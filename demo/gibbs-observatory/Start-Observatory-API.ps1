# Gibbs Observatory — start API with tsu on PYTHONPATH (src layout)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$TsuSrc = (Resolve-Path (Join-Path $Root "..\..\src")).Path
$env:TSU_ROOT = $TsuSrc
$env:PYTHONPATH = "$Root;$TsuSrc"
Write-Host "TSU_ROOT=$env:TSU_ROOT"
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv — run GibbsObservatory.exe once, or python -m venv .venv && pip install -r requirements.txt" }
& $py -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8088 --reload
