param(
  [int]$Jobs = 24
)

# P6: in-project implementation wrapper. Invokes scripts/run_implementation.tcl
# via vivado.bat in batch mode, streams the log to
# build/reports/p6_impl_vivado.log and fails unless the run prints IMPL_OK.

$ErrorActionPreference = 'Stop'

if (-not $env:HEATVIT_VIVADO_BIN) {
  # Machine-local default documented in README (adjust per machine).
  $env:HEATVIT_VIVADO_BIN = 'D:\vivado\vivado2023.2\Vivado\2023.2\bin'
}
$Vivado = Join-Path $env:HEATVIT_VIVADO_BIN.TrimEnd('\') 'vivado.bat'
if (-not (Test-Path -LiteralPath $Vivado)) {
  Write-Error "missing tool: $Vivado"
  exit 1
}

$RepoRoot = Split-Path $PSScriptRoot -Parent
$ReportDir = Join-Path $RepoRoot 'build\reports'
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$LogPath = Join-Path $ReportDir 'p6_impl_vivado.log'
$TclPath = Join-Path $RepoRoot 'scripts\run_implementation.tcl'
$XprPath = Join-Path $RepoRoot 'HeatViT.xpr'

Push-Location $RepoRoot
try {
  & $Vivado -mode batch -log $LogPath -source $TclPath -tclargs $XprPath $Jobs
  $Code = $LASTEXITCODE

  $LogText = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
  $Danger = @()
  if ($LogText) {
    $Danger = $LogText -split "`r?`n" | Where-Object {
      $_ -match 'route_design ERROR|place_design ERROR|opt_design ERROR|CRITICAL WARNING'
    }
    $Danger | ForEach-Object { Write-Host "DANGER: $_" }
  }

  if ($Code -ne 0 -or $LogText -notmatch 'IMPL_OK') {
    if (-not $Danger) { Write-Host 'IMPL_FAILED (see log tail below)' }
    Write-Host '--- log tail ---'
    Get-Content -LiteralPath $LogPath -Tail 40 | Write-Host
    exit 1
  }

  Get-Content -LiteralPath $LogPath |
    Where-Object { $_ -match 'IMPL_OK|IMPL_FAILED' } | Write-Host
  exit 0
}
finally {
  Pop-Location
}
