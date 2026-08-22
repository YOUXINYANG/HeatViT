param(
  [Parameter(Mandatory = $true)][string]$Pattern
)

$ErrorActionPreference = 'Stop'

if (-not $env:HEATVIT_PYTHON) {
  Write-Error 'HEATVIT_PYTHON is not set'
  exit 1
}

$Py = $env:HEATVIT_PYTHON
if (-not (Test-Path -LiteralPath $Py)) {
  Write-Error "python missing: $Py"
  exit 1
}

& $Py -c "import sys; assert (3,12) <= sys.version_info[:2] <= (3,14), 'python version out of range'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Py -c "import numpy; assert numpy.__version__ == '2.5.2', numpy.__version__"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RepoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $RepoRoot
try {
  & $Py -m unittest discover -s verification/tests -p $Pattern -v
  exit $LASTEXITCODE
}
finally {
  Pop-Location
}
