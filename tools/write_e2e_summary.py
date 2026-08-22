#!/usr/bin/env python3
"""Phase 5 Task 8: write/merge build/reports/e2e_summary.json.

Reads the frozen e2e manifest plus the current XSim logs. Each invocation
updates the entry for the STALL_MASK found in the latest tb_heatvit_e2e log
header (so ``-Suite e2e`` records the no-stall run and ``-Suite all`` later
adds the backpressure run), then re-serializes the report.
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUSIONS = (
    "合成权重无分类意义；未验证 ImageNet 准确率。",
    "未验证时序、功耗、FPS 或上板功能。",
    "交付结论严格限定为仿真逐位通过。",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    # XSim logs are UTF-16 LE (NUL-interleaved); decode accordingly.
    if raw[:2] == b"\xff\xfe" or b"\x00" in raw[:64]:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _parse_e2e_log(log_text: str):
    """Return (stall_mask, cycles, passed) for the latest e2e XSim log."""
    stall = None
    match = re.search(r"STALL_MASK=([0-9a-fA-F]+)", log_text)
    if match:
        stall = int(match.group(1), 16)
    cycles = None
    match = re.search(r"e2e_cycles=(\d+)", log_text)
    if match:
        cycles = int(match.group(1))
    return stall, cycles, "TEST_PASS tb_heatvit_e2e" in log_text


def _xsim_version(log_text: str) -> str:
    match = re.search(r"xsim\s+v([0-9.]+)", log_text)
    return f"xsim v{match.group(1)}" if match else "unknown"


def _read_run_files(report_dir: Path):
    """Per-mask run records written by run_regression.ps1 right after each
    e2e round (the shared xsim.log is overwritten by the next round)."""
    runs = {}
    for path in sorted(report_dir.glob("e2e_run_stall*.txt")):
        match = re.match(r"e2e_run_stall([0-9a-fA-F]+)\.txt", path.name)
        if not match:
            continue
        mask = int(match.group(1), 16)
        text = _read_text(path)
        cycles = None
        m = re.search(r"cycles=(\d+)", text)
        if m:
            cycles = int(m.group(1))
        runs[str(mask)] = {
            "stall_mask": mask,
            "cycles": cycles,
            "status": "PASS" if "status=PASS" in text else "UNKNOWN",
        }
    return runs


def main() -> int:
    import numpy  # noqa: F401  (record the locked version)

    vector_dir = REPO_ROOT / "build" / "vectors" / "e2e"
    manifest_path = vector_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    e2e_log = _read_text(REPO_ROOT / "build" / "xsim" / "tb_heatvit_e2e" / "xsim.log")
    errors_log = _read_text(
        REPO_ROOT / "build" / "xsim" / "tb_heatvit_errors" / "xsim.log"
    )
    stall_mask, cycles, passed = _parse_e2e_log(e2e_log)

    report_path = REPO_ROOT / "build" / "reports" / "e2e_summary.json"
    report_dir = report_path.parent
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"runs": {}, "errors_tb": {}}

    # Every section derived from the manifest is recomputed on each run so
    # a regenerated manifest (watchdog, hashes) always wins over stale data.
    report["vectors"] = {
        "seed": manifest.get("seed"),
        "token_counts": manifest.get("token_counts"),
        "watchdog_cycles": manifest.get("watchdog_cycles"),
        "manifest_sha256": _sha256(manifest_path),
        "files": manifest.get("files", {}),
    }
    report["selectors"] = manifest.get("selectors", [])
    report["checkpoints"] = []
    for entry in manifest.get("checkpoints", []):
        mem = vector_dir / "checkpoints" / f"{entry['name']}.mem"
        report["checkpoints"].append(
            {
                "name": entry["name"],
                "desc_index": entry["desc_index"],
                "sha256": _sha256(mem) if mem.exists() else None,
            }
        )
    report["logits_sha256"] = (
        _sha256(vector_dir / "checkpoints" / "logits.mem")
        if (vector_dir / "checkpoints" / "logits.mem").exists()
        else None
    )
    report["exclusions"] = list(EXCLUSIONS)

    # Tool versions are refreshed on every invocation (merge keeps the rest).
    report.setdefault("tool_versions", {})
    report["tool_versions"]["xsim"] = _xsim_version(e2e_log or errors_log)
    report["tool_versions"]["python"] = sys.version.split()[0]
    report["tool_versions"]["numpy"] = numpy.__version__

    if stall_mask is not None:
        report["runs"][str(stall_mask)] = {
            "stall_mask": stall_mask,
            "cycles": cycles,
            "status": "PASS" if passed else "FAIL",
        }
    # Merge the per-run records captured by run_regression.ps1 (the shared
    # log only retains the most recent round).
    for mask, entry in _read_run_files(report_dir).items():
        report["runs"][mask] = entry
    if errors_log:
        report["errors_tb"] = {
            "status": "PASS" if "TEST_PASS tb_heatvit_errors" in errors_log else "FAIL",
            "cases": 10,
        }
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
