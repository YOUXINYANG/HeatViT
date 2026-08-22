param(
  [Parameter(Mandatory = $true)][string]$Message,
  [Parameter(Mandatory = $true)][string[]]$Paths,
  [Parameter(Mandatory = $true)][string]$TestCommand
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $RepoRoot
try {
  if (Test-Path -LiteralPath (Join-Path $RepoRoot '.git')) {
    & git add -- @Paths
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & git commit -m $Message
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  else {
    $BuildDir = Join-Path $RepoRoot 'build'
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    $Log = Join-Path $BuildDir 'task-checkpoints.log'
    $Stamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    $Line = "$Stamp`t$Message`t$($Paths -join ',')`t$TestCommand"
    Add-Content -LiteralPath $Log -Value $Line -Encoding UTF8
  }
}
finally {
  Pop-Location
}
