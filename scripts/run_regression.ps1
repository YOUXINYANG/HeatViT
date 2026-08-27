param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('foundation', 'gemm', 'transformer', 'selector', 'e2e',
               'all')]
  [string]$Suite,
  [switch]$RegenerateVectors
)

$ErrorActionPreference = 'Stop'

if (-not $env:HEATVIT_PYTHON) {
  Write-Error 'HEATVIT_PYTHON is not set'
  exit 1
}
if (-not $env:HEATVIT_VIVADO_BIN) {
  Write-Error 'HEATVIT_VIVADO_BIN is not set'
  exit 1
}

$Py = $env:HEATVIT_PYTHON
if (-not (Test-Path -LiteralPath $Py)) {
  Write-Error "python missing: $Py"
  exit 1
}

$RepoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $RepoRoot
try {
  $Config = Get-Content -LiteralPath (Join-Path $RepoRoot 'config\heatvit_t.json') -Raw |
    ConvertFrom-Json
  $Seed = [int]$Config.seed

  function Invoke-FoundationSuite {
    # Regenerate every deterministic vector suite from the locked seed so two
    # consecutive runs must reproduce byte-identical manifests.
    $VectorSuites = @('fixed', 'requant', 'divsqrt', 'nonlinear', 'softmax', 'layernorm')
    foreach ($VectorSuite in $VectorSuites) {
      & $Py tools/generate_unit_vectors.py --suite $VectorSuite --seed $Seed `
        --output "sim/vectors/$VectorSuite"
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    # Python golden-model tests, including the config/package contract.
    & $Py -m unittest verification.tests.test_fixed verification.tests.test_nonlinear `
      verification.tests.test_config_contract -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # RTL self-checking testbenches in dependency order.
    $FoundationTops = @(
      'tb_pkg_smoke',
      'tb_requant_residual',
      'tb_udiv_isqrt',
      'tb_gelu_plan',
      'tb_gelu_pipe',
      'tb_softmax',
      'tb_layernorm',
      'tb_ln_p5_stale'
    )
    foreach ($Top in $FoundationTops) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top $Top
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
  }

  function Invoke-GemmSuite {
    # Python fixed-point and GEMM references first.
    & $Py -m unittest verification.tests.test_fixed verification.tests.test_gemm -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Regenerate the deterministic GEMM vector suite and configuration package.
    & $Py tools/generate_gemm_vectors.py --seed $Seed --output sim/vectors/gemm
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $GemmTops = @('tb_memory_path', 'tb_mem_master', 'tb_mac_bank')
    foreach ($Top in $GemmTops) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 -Top $Top
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $GemmCases = @('ordinary', 'tail', 'transpose', 'head')
    foreach ($Case in $GemmCases) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
        -Top tb_gemm_engine -PlusArgs "+CASE=$Case"
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    # Full-path random backpressure rerun of every GEMM mode.
    foreach ($Case in $GemmCases) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
        -Top tb_gemm_engine -PlusArgs "+CASE=$Case +STALL_MASK=3"
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
  }

  function Invoke-TransformerSuite {
    # Python layout and Transformer golden-model tests.
    & $Py -m unittest verification.tests.test_layout verification.tests.test_transformer -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Tensor Executor with full random backpressure, then Patch Embedding.
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_tensor_executor -PlusArgs '+STALL_MASK=3'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Py tools/generate_transformer_vectors.py --case patch --seed $Seed `
      --output build/vectors/patch
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_patch_embedding
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # MHSA9 and FFN13: regenerate the shared config right before each run.
    & $Py tools/generate_transformer_vectors.py --case mhsa --tokens 9 --seed $Seed `
      --output build/vectors/mhsa9
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_mhsa -PlusArgs '+VECTOR_DIR=build/vectors/mhsa9'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Py tools/generate_transformer_vectors.py --case ffn --tokens 13 --seed $Seed `
      --output build/vectors/ffn13
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_ffn -PlusArgs '+VECTOR_DIR=build/vectors/ffn13'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Two complete blocks: full-size no-stall plus tail/backpressure.
    & $Py tools/generate_transformer_vectors.py --case block --tokens 197 `
      --seed $Seed --output build/vectors/block197
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_transformer_block `
      -PlusArgs '+VECTOR_DIR=build/vectors/block197 +STALL_MASK=0'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Py tools/generate_transformer_vectors.py --case block --tokens 13 `
      --seed 20260816 --output build/vectors/block13
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_transformer_block `
      -PlusArgs '+VECTOR_DIR=build/vectors/block13 +STALL_MASK=3'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }

  function Invoke-SelectorSuite {
    # Python selector golden-model tests.
    & $Py -m unittest verification.tests.test_selector -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Regenerate the deterministic selector vectors (six unit cases plus
    # the full N=197 mixed-pruning case).
    & $Py tools/generate_selector_vectors.py --suite unit --seed $Seed `
      --output sim/vectors/selector
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Py tools/generate_selector_vectors.py --suite full --case mixed `
      --tokens 197 --seed $Seed --output build/vectors/selector_mixed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # The selector stage builds on the complete transformer stage.
    Invoke-FoundationSuite
    Invoke-GemmSuite
    Invoke-TransformerSuite

    $SelectorTops = @(
      'tb_selector_features',
      'tb_head_fuse',
      'tb_selector_finalize',
      'tb_token_selector'
    )
    foreach ($Top in $SelectorTops) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
        -Top $Top -PlusArgs '+STALL_MASK=3'
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
  }

  function Write-E2eRunRecord([string]$StallMask) {
    # The shared xsim.log only keeps the latest e2e round; persist each
    # round's cycle count as soon as it passes so the summary tool can merge
    # both stall masks.
    $LogPath = Join-Path $RepoRoot 'build\xsim\tb_heatvit_e2e\xsim.log'
    $Log = Get-Content -LiteralPath $LogPath -Raw
    $Cycles = 'unknown'
    if ($Log -match 'e2e_cycles=(\d+)') { $Cycles = $Matches[1] }
    $ReportDir = Join-Path $RepoRoot 'build\reports'
    New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
    Set-Content -LiteralPath (Join-Path $ReportDir "e2e_run_stall$StallMask.txt") `
      -Value "cycles=$Cycles`nstatus=PASS" -Encoding Ascii
  }

  function Invoke-E2eSuite {
    # Verify the frozen manifest before running; regenerate only when the
    # explicit switch is given so failed reruns cannot silently swap the
    # expected values.
    & $Py -m unittest verification.tests.test_e2e_manifest -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($RegenerateVectors) {
      & $Py tools/generate_e2e_vectors.py --seed 20260815 `
        --output build/vectors/e2e
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
      & $Py -m unittest verification.tests.test_e2e_manifest -v
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_heatvit_e2e -PlusArgs '+VECTOR_DIR=build/vectors/e2e +STALL_MASK=0'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-E2eRunRecord '0'
    & $Py tools/write_e2e_summary.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }

  function Invoke-AllSuite {
    Invoke-SelectorSuite
    Invoke-E2eSuite
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_heatvit_errors
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
      -Top tb_heatvit_e2e -PlusArgs '+VECTOR_DIR=build/vectors/e2e +STALL_MASK=3'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-E2eRunRecord '3'
    & $Py tools/write_e2e_summary.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }

  if ($Suite -eq 'foundation') {
    Invoke-FoundationSuite
  } elseif ($Suite -eq 'gemm') {
    Invoke-GemmSuite
  } elseif ($Suite -eq 'transformer') {
    Invoke-FoundationSuite
    Invoke-GemmSuite
    Invoke-TransformerSuite
  } elseif ($Suite -eq 'selector') {
    Invoke-SelectorSuite
  } elseif ($Suite -eq 'e2e') {
    Invoke-E2eSuite
  } else {
    Invoke-AllSuite
  }

  # Rolling regression summary: reaching this line means every step of the
  # requested suite passed (any failure above exits nonzero).
  $Stamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
  $ReportDir = Join-Path $RepoRoot 'build\reports'
  New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
  Add-Content -LiteralPath (Join-Path $ReportDir 'regression_summary.txt') `
    -Value "$Stamp`tsuite=$Suite`tresult=PASS" -Encoding UTF8
}
finally {
  Pop-Location
}
