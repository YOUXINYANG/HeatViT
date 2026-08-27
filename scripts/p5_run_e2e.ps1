# P5 e2e XSim regression: all exported images x both backpressure masks.
#
# Stages each image's e2e_tb_config.sv before compiling its rounds (the
# shared sim/generated/e2e_tb_config.sv is overwritten per image, so each
# image directory carries its own copy). Rounds with a .pass marker are
# skipped, so a killed run resumes where it stopped. Cycle counts are
# appended to build/reports/p5_e2e_runs.txt.
#
# Requires HEATVIT_VIVADO_BIN. Run after scripts/p5_export.ps1.
param(
  [string]$VectorDir = 'build/vectors/e2e_p5'
)

$ErrorActionPreference = 'Stop'
if (-not $env:HEATVIT_VIVADO_BIN) {
  Write-Error 'HEATVIT_VIVADO_BIN is not set'
  exit 1
}
$RepoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $RepoRoot
try {
  $Report = Join-Path $RepoRoot 'build\reports\p5_e2e_runs.txt'
  New-Item -ItemType Directory -Force -Path (Split-Path $Report) | Out-Null

  foreach ($Img in @('img0', 'img1', 'img2')) {
    $Cfg = Join-Path $RepoRoot "$VectorDir\$Img\e2e_tb_config.sv"
    if (-not (Test-Path -LiteralPath $Cfg)) {
      throw "missing per-image config: $Cfg"
    }
    Copy-Item -LiteralPath $Cfg `
      -Destination (Join-Path $RepoRoot 'sim\generated\e2e_tb_config.sv') `
      -Force

    foreach ($Mask in @('0', '3')) {
      $Marker = Join-Path $RepoRoot "$VectorDir\$Img`_stall$Mask.pass"
      if (Test-Path -LiteralPath $Marker) {
        Write-Host "skip $Img STALL_MASK=$Mask (already passed)"
        continue
      }
      Write-Host "=== P5 e2e $Img STALL_MASK=$Mask ==="
      & powershell -NoProfile -ExecutionPolicy Bypass -File `
        scripts/run_xsim.ps1 -Top tb_heatvit_e2e `
        -PlusArgs "+VECTOR_DIR=$VectorDir/$Img +STALL_MASK=$Mask"
      if ($LASTEXITCODE -ne 0) {
        throw "P5 e2e failed: $Img STALL_MASK=$Mask"
      }
      $Cycles = 'unknown'
      $Log = Get-Content -LiteralPath `
        (Join-Path $RepoRoot 'build\xsim\tb_heatvit_e2e\xsim.log') -Raw
      if ($Log -match 'e2e_cycles=(\d+)') { $Cycles = $Matches[1] }
      Add-Content -LiteralPath $Report `
        -Value "$(Get-Date -Format s)`t$Img`tstall=$Mask`tcycles=$Cycles`tPASS" `
        -Encoding ASCII
      Set-Content -LiteralPath $Marker -Value 'PASS' -Encoding Ascii
      Write-Host "P5 e2e PASS $Img STALL_MASK=$Mask (cycles=$Cycles)"
    }
  }
  Write-Host 'P5_E2E_ALL_PASS'
}
finally {
  Pop-Location
}
