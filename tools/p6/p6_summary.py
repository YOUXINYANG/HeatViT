#!/usr/bin/env python
"""P6: parse Vivado utilization / timing reports into a compact summary.

Reads build/reports/p6_{synth,impl}_util.txt and p6_timing_summary.txt and
writes:
  - build/reports/p6_summary.json  (machine-readable)
  - build/reports/p6_summary.txt   (human-readable table vs xc7k325t totals)

Usage:
  python tools/p6/p6_summary.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / 'build' / 'reports'

# XC7K325T totals (DS180 / Vivado utilization report "Available" column).
DEVICE = {
    'part': 'xc7k325tfbg900-3',
    'slice_luts': 203800,
    'slice_regs': 407600,
    'bram_tile': 445,
    'dsp': 840,
}

# report_utilization row label -> (json key, device key for totals).
KEYS = {
    'Slice LUTs': ('lut', 'slice_luts'),
    'LUT as Memory': ('lutram', None),
    'Slice Registers': ('ff', 'slice_regs'),
    'Block RAM Tile': ('bram_tile', 'bram_tile'),
    'RAMB36/FIFO': ('bram36', None),
    'RAMB18': ('bram18', None),
    'DSPs': ('dsp', 'dsp'),
}


def parse_util(path):
    out = {}
    if not path.exists():
        raise SystemExit('missing report: %s' % path)
    for line in path.read_text(errors='replace').splitlines():
        if '|' not in line or 'Site Type' in line:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 6:
            continue
        key = cells[0].rstrip('*')
        if key in KEYS:
            used, avail = cells[1].replace(',', ''), cells[4].replace(',', '')
            try:
                out[KEYS[key][0]] = {'used': int(used), 'available': int(avail)}
            except ValueError:
                out[KEYS[key][0]] = {'used': cells[1], 'available': cells[4]}
    if 'lut' not in out:
        raise SystemExit('parse failed (no Slice LUTs row): %s' % path)
    return out


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


def main():
    synth = parse_util(REPORTS / 'p6_synth_util.txt')
    impl = parse_util(REPORTS / 'p6_impl_util.txt') if \
        (REPORTS / 'p6_impl_util.txt').exists() else None
    timing = parse_timing(REPORTS / 'p6_timing_summary.txt')

    summary = {
        'part': DEVICE['part'],
        'clock_mhz': 100.0,
        'device': DEVICE,
        'synth': synth,
        'impl': impl,
        'timing': timing,
    }
    (REPORTS / 'p6_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    rows = [
        ('Slice LUTs', 'lut', 'slice_luts'),
        ('  LUT as Memory', 'lutram', None),
        ('Slice Registers', 'ff', 'slice_regs'),
        ('Block RAM Tile', 'bram_tile', 'bram_tile'),
        ('  RAMB36/FIFO', 'bram36', None),
        ('  RAMB18', 'bram18', None),
        ('DSPs', 'dsp', 'dsp'),
    ]
    lines = ['P6 HeatViT resource summary (xc7k325tfbg900-3, 100 MHz)',
             '',
             '%-20s%12s%12s%12s%10s' % ('Resource', 'Synth', 'Impl',
                                        'Avail', 'Impl%')]
    for label, key, devkey in rows:
        s = synth.get(key, {}).get('used', '-')
        i = impl.get(key, {}).get('used', '-') if impl else '-'
        a = (impl or synth).get(key, {}).get('available', None)
        if a is None and devkey:
            a = DEVICE[devkey]
        pct = ''
        if isinstance(i, int) and a:
            pct = '%8.1f%%' % (i * 100.0 / a)
        lines.append('%-20s%12s%12s%12s%10s' % (label, s, i, a, pct))
    if timing:
        lines.append('')
        lines.append('WNS = %s ns, TNS = %s ns, %s' % (
            timing['wns_ns'], timing['tns_ns'],
            'MET' if timing['met'] else 'VIOLATED'))
    elif impl is None:
        lines.append('')
        lines.append('implementation not run (synth-only statistics)')
    text = '\n'.join(lines) + '\n'
    (REPORTS / 'p6_summary.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
