"""Shared helpers for the P8 RTL performance tooling.

Descriptor ROM loading, stage bucketing, Vivado report parsing.
Only the standard library is required.
"""

from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from verification.heatvit_ref.descriptor import (  # noqa: E402
    Descriptor,
    FLAG_DYNAMIC_K,
    FLAG_DYNAMIC_M,
    FLAG_DYNAMIC_N,
    FLAG_HEAD_MODE,
)

ROM = REPO / 'rtl' / 'generated' / 'heatvit_descriptors.mem'
REPORTS = REPO / 'build' / 'reports'
E2E_SUMMARY = REPORTS / 'p7_5_e2e_summary.json'
E2E_SUMMARY_FALLBACK = REPORTS / 'e2e_summary.json'

CLOCK_MHZ = 100.0
PEAK_MAC_PER_CYCLE = 192          # 3 banks x 8x8 int8 MACs
MAC_PER_ACTIVE_CYCLE = 64         # one bank does 8x8 = 64 MACs / cycle

OPCODE_NAMES = {
    0: 'NOP', 1: 'PATCHIFY', 2: 'COPY_ADD_POS', 3: 'GEMM', 4: 'LAYERNORM',
    5: 'RESIDUAL', 6: 'QKV_UNPACK', 7: 'HEAD_CONCAT', 8: 'ATTN_SOFTMAX',
    9: 'SELECTOR_SOFTMAX', 10: 'REDUCE_MEAN', 11: 'CONCAT_LOCAL_GLOBAL',
    12: 'HEAD_FUSE', 13: 'SELECTOR_FINALIZE', 14: 'FINISH',
}

# Stage names and the descriptor index at which each stage's checkpoint
# fires (i.e. stage i spans descriptors (prev_cp, cp[i]] inclusive).
CP_NAMES = ['patch', 'block_01', 'block_02', 'block_03', 'selector_01',
            'block_04', 'block_05', 'block_06', 'selector_02', 'block_07',
            'block_08', 'block_09', 'selector_03', 'block_10', 'block_11',
            'block_12', 'final_ln', 'logits']
CP_INDEX = [2, 15, 28, 41, 53, 66, 79, 92, 104, 117, 130, 143, 155, 168,
            181, 194, 195, 196]

# Live token count feeding each stage (e2e vector manifest: 197 -> 100 ->
# 55 -> 28; selectors consume the input count of their stage).
TOKENS_BY_STAGE = [197, 197, 197, 197, 197, 100, 100, 100, 100,
                   55, 55, 55, 55, 28, 28, 28, 28, 28]


def read_text_any(path):
    """Read a text file regardless of PowerShell/Vivado log encodings.

    PowerShell ``*>`` redirection writes UTF-16LE with BOM; Vivado itself
    writes plain ASCII/UTF-8.  Sniff the BOM and fall back to UTF-8.
    """
    raw = Path(path).read_bytes()
    if raw[:2] == b'\xff\xfe':
        return raw.decode('utf-16-le', errors='replace')
    if raw[:2] == b'\xfe\xff':
        return raw.decode('utf-16-be', errors='replace')
    return raw.decode('utf-8', errors='replace')


def load_descriptors():
    """Parse the 198-word descriptor ROM into Descriptor objects."""
    words = []
    for line in ROM.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            words.append(int(line, 16))
    if len(words) != 198:
        raise SystemExit('descriptor ROM has %d words, expected 198'
                         % len(words))
    return [Descriptor.unpack(w) for w in words]


def stage_of(idx):
    """Bucket a descriptor index into its stage slot (0..17)."""
    for i, end in enumerate(CP_INDEX):
        if idx <= end:
            return i
    return len(CP_INDEX) - 1


def eff_mnk(d, tokens):
    """Resolve effective M/N/K exactly as heatvit_tensor_executor does.

    Dynamic M: param0[1:0] selects DYN_M_CURRENT (m = tokens) or
    DYN_M_CANDIDATES (m = tokens - 1).  Dynamic N/K replace the packed
    value with the live token count outright.
    """
    m, n, k = d.m, d.n, d.k
    if d.flags & (1 << FLAG_DYNAMIC_M):
        mode = d.param0 & 0x3
        m = tokens if mode == 0 else (tokens - 1 if mode == 1 else m)
    if d.flags & (1 << FLAG_DYNAMIC_N):
        n = tokens
    if d.flags & (1 << FLAG_DYNAMIC_K):
        k = tokens
    return m, n, k


def gemm_macs_per_stage(tokens_by_stage):
    """Sum effective m*n*k over OP_GEMM descriptors per stage.

    Head-mode GEMMs compute one head per bank, so their packed n is the
    per-bank width: total MACs carry the ``heads`` multiplier.
    """
    stages = [0] * len(CP_NAMES)
    for i, d in enumerate(load_descriptors()):
        if d.opcode == 3:  # OP_GEMM
            m, n, k = eff_mnk(d, tokens_by_stage[stage_of(i)])
            mult = d.heads if (d.flags & (1 << FLAG_HEAD_MODE)) else 1
            stages[stage_of(i)] += mult * m * n * k
    return stages


UNPRUNED_TOKENS = [197] * len(CP_NAMES)


def unpruned_macs_per_stage():
    """Theoretical MACs if no token were ever pruned (197 everywhere)."""
    return gemm_macs_per_stage(UNPRUNED_TOKENS)


def e2e_macs_per_stage():
    """Theoretical MACs at the e2e vector's live token counts."""
    return gemm_macs_per_stage(TOKENS_BY_STAGE)


# --- Vivado report parsing (kept in sync with tools/p6/p6_summary.py) ---

UTIL_KEYS = {
    'Slice LUTs': 'lut',
    'LUT as Memory': 'lutram',
    'Slice Registers': 'ff',
    'Block RAM Tile': 'bram_tile',
    'DSPs': 'dsp',
}


def parse_util(path):
    out = {}
    if not path.exists():
        return None
    for line in path.read_text(errors='replace').splitlines():
        if '|' not in line or 'Site Type' in line:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 6:
            continue
        key = cells[0].rstrip('*')
        if key in UTIL_KEYS:
            used, avail = cells[1].replace(',', ''), cells[4].replace(',', '')
            try:
                out[UTIL_KEYS[key]] = {'used': int(used),
                                       'available': int(avail)}
            except ValueError:
                out[UTIL_KEYS[key]] = {'used': cells[1],
                                       'available': cells[4]}
    return out or None


def parse_timing(path):
    if not path.exists():
        return None
    lines = path.read_text(errors='replace').splitlines()
    for i, line in enumerate(lines):
        if 'WNS(ns)' not in line or 'TNS(ns)' not in line:
            continue
        for cand in lines[i + 1:i + 6]:
            toks = cand.replace('|', ' ').split()
            if len(toks) >= 2:
                try:
                    wns, tns = float(toks[0]), float(toks[1])
                except ValueError:
                    continue
                met = 'All user specified timing constraints are met.' in \
                    '\n'.join(lines)
                return {'wns_ns': wns, 'tns_ns': tns, 'met': met}
    return None


def load_e2e_runs():
    """Return {'runs': {mask: {'cycles', 'status'}}, 'token_counts', ...}."""
    import json
    path = E2E_SUMMARY if E2E_SUMMARY.exists() else E2E_SUMMARY_FALLBACK
    if not path.exists():
        raise SystemExit('missing e2e summary JSON')
    return json.loads(path.read_text(encoding='utf-8'))
