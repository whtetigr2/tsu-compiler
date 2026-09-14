# Build GibbsObservatory.exe (run on Windows from repo)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "backend"))) { $Root = $PSScriptRoot }
Set-Location $Root

$py = (Get-Command python).Source
& $py -m pip install --upgrade pip pyinstaller
$launcher = Join-Path $PSScriptRoot "gibbs_observatory_launcher.py"
$dist = Join-Path $Root "dist"
$work = Join-Path $Root "build\launcher-work"

& $py -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name "GibbsObservatory" `
  --distpath $dist `
  --workpath $work `
  --specpath (Join-Path $Root "build") `
  --onefile `
  $launcher

$exe = Join-Path $dist "GibbsObservatory.exe"
# Also copy to repo root for clone-and-double-click
Copy-Item $exe (Join-Path $Root "GibbsObservatory.exe") -Force
Write-Host "Built: $exe"
Write-Host "Copied to repo root: $(Join-Path $Root 'GibbsObservatory.exe')"
Get-Item $exe | Format-List FullName, Length
