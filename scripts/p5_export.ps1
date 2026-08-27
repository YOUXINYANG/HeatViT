# P5 export: QAT checkpoint -> RTL weight vectors + per-tensor descriptors.
#
# Mirrors the P2-D flow with the P5 checkpoint entry: the QAT checkpoint's
# 'floats' key (HeatViT layout) replaces the official DeiT-T weights, the
# frozen PTQ scale table plus the selector s{i}_* entries drive the
# per-tensor descriptors (--write-rom regenerates rtl/generated/
# heatvit_descriptors.mem; only run when no other XSim run is pending,
# since the ROM is read at simulation time 0).
#
# Runs the golden-model vs deployment-simulator cross-check first, then
# exports one per-image vector directory (each carrying its own
# e2e_tb_config.sv to be staged before its XSim round).
#
# Usage (torch venv):
#   powershell -File scripts\p5_export.ps1 `
#     -Checkpoint p2_out\qat\p4a_rate5_16k\best.pt
param(
  [Parameter(Mandatory = $true)][string]$Checkpoint,
  [string]$Selectors = 'p2_out/selectors_sup4.pt',
  [string]$Table = 'p2_out/scale_table.json',
  [string]$Output = 'build/vectors/e2e_p5',
  [int]$Images = 3,
  [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path $PSScriptRoot -Parent
if (-not $Python) {
  $Python = Join-Path $RepoRoot '.venv-torch\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python)) {
  Write-Error "torch python missing: $Python"
  exit 1
}
Push-Location $RepoRoot
try {
  Write-Host "=== p5_crosscheck ==="
  & $Python tools/p2/p5_crosscheck.py --checkpoint $Checkpoint `
    --selectors $Selectors --table $Table --images $Images
  if ($LASTEXITCODE -ne 0) { throw 'p5_crosscheck failed' }

  Write-Host "=== p2_export_weights (--write-rom) ==="
  & $Python tools/p2/p2_export_weights.py --checkpoint $Checkpoint `
    --selectors $Selectors --table $Table --write-rom `
    --images $Images --output $Output
  if ($LASTEXITCODE -ne 0) { throw 'p2_export_weights failed' }
  Write-Host "P5 export done -> $Output"
}
finally {
  Pop-Location
}
