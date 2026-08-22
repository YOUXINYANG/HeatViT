param()

# Phase 5 Task 8: manual-Vivado-IP audit.
#
# Recursively scans every .sv under rtl/ plus the project .xpr for the
# forbidden instantiation patterns and checks for any .xci/.xcix IP files.
# Any hit exits 1; a clean run writes NO_MANUAL_VIVADO_IP_REQUIRED into
# build/reports/ip_audit.txt.

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path $PSScriptRoot -Parent
$ReportDir = Join-Path $RepoRoot 'build\reports'
$ReportPath = Join-Path $ReportDir 'ip_audit.txt'
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

$Patterns = @('xpm_', 'blk_mem_gen', 'floating_point', 'div_gen', 'axi_',
              'IPSources')
$Hits = @()

$RtlDir = Join-Path $RepoRoot 'rtl'
if (Test-Path -LiteralPath $RtlDir) {
  Get-ChildItem -LiteralPath $RtlDir -Recurse -Filter '*.sv' -File |
    ForEach-Object {
      $File = $_.FullName
      $Content = Get-Content -LiteralPath $File -Raw
      foreach ($Pattern in $Patterns) {
        if ($Content -match $Pattern) {
          $Hits += "$Pattern in $($File.Substring($RepoRoot.Length + 1))"
        }
      }
    }
}

$XprPath = Join-Path $RepoRoot 'HeatViT.xpr'
if (Test-Path -LiteralPath $XprPath) {
  $Xpr = Get-Content -LiteralPath $XprPath -Raw
  foreach ($Pattern in $Patterns) {
    if ($Xpr -match $Pattern) {
      $Hits += "$Pattern in HeatViT.xpr"
    }
  }
}

$XciFiles = Get-ChildItem -LiteralPath $RepoRoot -Recurse -File `
  -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -eq '.xci' -or $_.Extension -eq '.xcix' }
foreach ($Xci in $XciFiles) {
  $Hits += "IP container $($Xci.Name) at $($Xci.FullName.Substring($RepoRoot.Length + 1))"
}

if ($Hits.Count -gt 0) {
  Set-Content -LiteralPath $ReportPath -Value $Hits -Encoding Ascii
  Write-Host 'IP_AUDIT_FAILED'
  $Hits | ForEach-Object { Write-Host "  $_" }
  exit 1
}

Set-Content -LiteralPath $ReportPath -Value 'NO_MANUAL_VIVADO_IP_REQUIRED' `
  -Encoding Ascii
Write-Host 'NO_MANUAL_VIVADO_IP_REQUIRED'
exit 0
