param(
  [Parameter(Mandatory = $true)][string]$Top,
  [string]$PlusArgs = ''
)

$ErrorActionPreference = 'Stop'

if (-not $env:HEATVIT_VIVADO_BIN) {
  Write-Error 'HEATVIT_VIVADO_BIN is not set'
  exit 1
}

$VivadoBin = $env:HEATVIT_VIVADO_BIN.TrimEnd('\')
$Xvlog = Join-Path $VivadoBin 'xvlog.bat'
$Xelab = Join-Path $VivadoBin 'xelab.bat'
$Xsim  = Join-Path $VivadoBin 'xsim.bat'

foreach ($Tool in @($Xvlog, $Xelab, $Xsim)) {
  if (-not (Test-Path -LiteralPath $Tool)) {
    Write-Error "missing tool: $Tool"
    exit 1
  }
}

if (-not $env:XILINX_VIVADO) {
  $env:XILINX_VIVADO = Split-Path $VivadoBin -Parent
}

$RepoRoot = Split-Path $PSScriptRoot -Parent
$LogDir = Join-Path $RepoRoot ('build\xsim\' + $Top)
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Sources = New-Object System.Collections.Generic.List[string]

$IncludeDir = Join-Path $RepoRoot 'rtl\include'
if (Test-Path -LiteralPath $IncludeDir) {
  Get-ChildItem -LiteralPath $IncludeDir -Filter '*.sv' -File |
    Sort-Object Name | ForEach-Object { $Sources.Add($_.FullName) }
}

$RtlDir = Join-Path $RepoRoot 'rtl'
if (Test-Path -LiteralPath $RtlDir) {
  Get-ChildItem -LiteralPath $RtlDir -Recurse -Filter '*.sv' -File |
    Where-Object { $_.FullName -notlike ($IncludeDir + '*') } |
    Sort-Object FullName | ForEach-Object { $Sources.Add($_.FullName) }
}

$ProjectSrcDir = Join-Path $RepoRoot 'HeatViT.srcs\sources_1\new'
if (Test-Path -LiteralPath $ProjectSrcDir) {
  Get-ChildItem -LiteralPath $ProjectSrcDir -Filter '*.sv' -File |
    Sort-Object Name | ForEach-Object { $Sources.Add($_.FullName) }
}

$CommonDir = Join-Path $RepoRoot 'sim\common'
if (Test-Path -LiteralPath $CommonDir) {
  Get-ChildItem -LiteralPath $CommonDir -Filter '*.sv' -File |
    Sort-Object Name | ForEach-Object { $Sources.Add($_.FullName) }
}

$GeneratedDir = Join-Path $RepoRoot 'sim\generated'
if (Test-Path -LiteralPath $GeneratedDir) {
  Get-ChildItem -LiteralPath $GeneratedDir -Filter '*.sv' -File |
    Sort-Object Name | ForEach-Object { $Sources.Add($_.FullName) }
}

$TbFile = Join-Path $RepoRoot ('sim\tb\' + $Top + '.sv')
if (-not (Test-Path -LiteralPath $TbFile)) {
  Write-Error "testbench missing: $TbFile"
  exit 1
}
$Sources.Add($TbFile)

$Resp = Join-Path $LogDir 'xvlog_sources.f'
Set-Content -LiteralPath $Resp -Value $Sources -Encoding Ascii

Push-Location $RepoRoot
try {
  & $Xvlog -sv -f $Resp *> (Join-Path $LogDir 'xvlog.log')
  if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath (Join-Path $LogDir 'xvlog.log') | Write-Host
    exit $LASTEXITCODE
  }

  & $Xelab $Top -s ($Top + '_snapshot') -timescale 1ns/1ps *> (Join-Path $LogDir 'xelab.log')
  if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath (Join-Path $LogDir 'xelab.log') | Write-Host
    exit $LASTEXITCODE
  }

  $RunArgs = @(($Top + '_snapshot'), '-runall', '-onerror', 'quit', '-onfinish', 'quit')
  foreach ($Arg in ($PlusArgs -split ' ')) {
    if ($Arg) {
      $RunArgs += '-testplusarg'
      # Quote the value so the Vivado .bat launcher chain does not split it
      # at '=' (value-form plusargs such as +CASE=ordinary).
      $RunArgs += ('"' + $Arg.TrimStart('+') + '"')
    }
  }
  & $Xsim @RunArgs *> (Join-Path $LogDir 'xsim.log')
  $Code = $LASTEXITCODE
  $LogText = Get-Content -LiteralPath (Join-Path $LogDir 'xsim.log') -Raw
  if ($Code -eq 0 -and $LogText -match 'Fatal:|ERROR:|Error:') {
    $Code = 1
  }
  Get-Content -LiteralPath (Join-Path $LogDir 'xsim.log') |
    Where-Object { $_ -match 'TEST_PASS|Fatal|Error' } | Write-Host
  exit $Code
}
finally {
  Pop-Location
}
