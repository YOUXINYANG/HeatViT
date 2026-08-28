# P7-2 fast gate: run only the four selector-side self-checking TBs that
# exercise the rewritten modules (no foundation/gemm/transformer re-run).
param()
foreach ($Top in @('tb_selector_features', 'tb_head_fuse',
                   'tb_selector_finalize', 'tb_token_selector')) {
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_xsim.ps1 `
    -Top $Top -PlusArgs '+STALL_MASK=3'
  if ($LASTEXITCODE -ne 0) { Write-Error "FAST_GATE_FAIL $Top"; exit 1 }
}
Write-Output 'P7B_FAST_GATE_PASS'
