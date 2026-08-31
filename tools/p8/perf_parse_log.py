#!/usr/bin/env python
"""P8-1: parse the instrumented e2e simulation log into a stage profile.

Reads the perf_desc / perf_sum lines that the tb_heatvit_e2e performance
monitor prints, buckets the 197 executed descriptors by stage, and writes
build/reports/perf_profile.{json,txt} with per-stage cycle distribution,
measured MAC activity, external-memory traffic and derived utilization
metrics.

Usage:
  python tools/p8/perf_parse_log.py [log]
  (default: build/xsim/tb_heatvit_e2e/xsim.log)
"""

import json
import re
import sys
from pathlib import Path

from _perf_common import (
    CLOCK_MHZ,
    CP_NAMES,
    MAC_PER_ACTIVE_CYCLE,
    PEAK_MAC_PER_CYCLE,
    REPORTS,
    REPO,
    e2e_macs_per_stage,
    read_text_any,
    stage_of,
)

DESC_RE = re.compile(
    r'^perf_desc idx=(\d+) op=(\d+) start=(\d+) end=(\d+) dur=(\d+) '
    r'mac=(\d+)/(\d+)/(\d+)', re.MULTILINE)
SUM_RE = re.compile(
    r'^perf_sum cycles=(\d+) busy=(\d+) desc=(\d+) '
    r'mac_total=(\d+)/(\d+)/(\d+) cmd_beats=(\d+) rd_bytes=(\d+) '
    r'wr_bytes=(\d+) rd_beats=(\d+) wr_beats=(\d+) cmd_stall=(\d+)',
    re.MULTILINE)
CYCLE_RE = re.compile(r'^e2e_cycles=(\d+)', re.MULTILINE)


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / 'build' / 'xsim' / 'tb_heatvit_e2e' / 'xsim.log')
    if not log_path.exists():
        raise SystemExit('missing simulation log: %s' % log_path)
    text = read_text_any(log_path)

    if 'TEST_PASS tb_heatvit_e2e' not in text:
        raise SystemExit('log has no TEST_PASS line; sim did not pass')

    descs = []
    for m in DESC_RE.finditer(text):
        descs.append({
            'idx': int(m.group(1)),
            'op': int(m.group(2)),
            'start': int(m.group(3)),
            'end': int(m.group(4)),
            'dur': int(m.group(5)),
            'mac': [int(m.group(6)), int(m.group(7)), int(m.group(8))],
        })

    m_sum = SUM_RE.search(text)
    if not m_sum:
        raise SystemExit('log has no perf_sum line (TB too old?)')
    tot = {
        'cycles': int(m_sum.group(1)),
        'busy': int(m_sum.group(2)),
        'desc_count': int(m_sum.group(3)),
        'mac_total': [int(m_sum.group(4)), int(m_sum.group(5)),
                      int(m_sum.group(6))],
        'cmd_beats': int(m_sum.group(7)),
        'rd_bytes': int(m_sum.group(8)),
        'wr_bytes': int(m_sum.group(9)),
        'rd_beats': int(m_sum.group(10)),
        'wr_beats': int(m_sum.group(11)),
        'cmd_stalls': int(m_sum.group(12)),
    }

    m_cyc = CYCLE_RE.search(text)
    e2e_cycles = int(m_cyc.group(1)) if m_cyc else tot['cycles']

    # Sanity: exactly 197 descriptors (0..196) in monotonically
    # increasing order; OP_FINISH is scheduler-internal and must not
    # appear.
    if tot['desc_count'] != 197:
        raise SystemExit('expected 197 executed descriptors, got %d'
                         % tot['desc_count'])
    idxs = [d['idx'] for d in descs]
    if idxs != list(range(197)):
        raise SystemExit('descriptor index sequence broken: %s' % idxs)
    if any(d['op'] == 14 for d in descs):
        raise SystemExit('OP_FINISH must not be executed by the executor')

    # Stage bucketing (checkpoint boundaries are inclusive ends).
    stages = [{'stage': CP_NAMES[i], 'cycles': 0, 'gemm_cycles': 0,
               'mac_active_cycles': 0, 'desc_count': 0,
               'geom_macs': 0} for i in range(len(CP_NAMES))]
    for i, macs in enumerate(e2e_macs_per_stage()):
        stages[i]['geom_macs'] = macs

    for rec in descs:
        s = stage_of(rec['idx'])
        stages[s]['cycles'] += rec['dur']
        stages[s]['desc_count'] += 1
        if rec['op'] == 3:
            stages[s]['gemm_cycles'] += rec['dur']
            stages[s]['mac_active_cycles'] += sum(rec['mac'])

    # Derived metrics.
    mac_ops = sum(tot['mac_total']) * MAC_PER_ACTIVE_CYCLE
    gemm_cycles = sum(s['gemm_cycles'] for s in stages)
    total_bytes = tot['rd_bytes'] + tot['wr_bytes']

    metrics = {
        'cycles': e2e_cycles,
        'busy_cycles': tot['busy'],
        'busy_duty': tot['busy'] / e2e_cycles,
        'fps': CLOCK_MHZ * 1e6 / e2e_cycles,
        'seconds_per_image': e2e_cycles / (CLOCK_MHZ * 1e6),
        'mac_ops': mac_ops,
        'mac_utilization': mac_ops / (e2e_cycles * PEAK_MAC_PER_CYCLE),
        'gemm_cycles': gemm_cycles,
        'gemm_cycle_share': gemm_cycles / e2e_cycles,
        'gemm_time_utilization': mac_ops / (gemm_cycles
                                            * PEAK_MAC_PER_CYCLE)
        if gemm_cycles else 0.0,
        'rd_bytes': tot['rd_bytes'],
        'wr_bytes': tot['wr_bytes'],
        'total_bytes': total_bytes,
        'bw_mbps': total_bytes / e2e_cycles * CLOCK_MHZ,
        'rd_bytes_per_cycle': tot['rd_bytes'] / e2e_cycles,
        'cmd_beats': tot['cmd_beats'],
        'rd_beats': tot['rd_beats'],
        'wr_beats': tot['wr_beats'],
        'cmd_stalls': tot['cmd_stalls'],
        'cmd_stall_ratio': tot['cmd_stalls'] / e2e_cycles,
    }

    profile = {
        'log': str(log_path),
        'metrics': metrics,
        'stages': stages,
    }
    (REPORTS / 'perf_profile.json').write_text(
        json.dumps(profile, indent=2) + '\n', encoding='utf-8')

    # Human-readable report.
    lines = [
        'P8-1 HeatViT e2e performance profile (%s)' % log_path.name,
        '',
        'Run: %d cycles | %.3f FPS @%g MHz | %.2f s/image' % (
            e2e_cycles, metrics['fps'], CLOCK_MHZ,
            metrics['seconds_per_image']),
        'MAC activity: %.3f G active MACs | whole-run utilization '
        '%.2f%% | within-GEMM utilization %.2f%%' % (
            mac_ops / 1e9, metrics['mac_utilization'] * 100,
            metrics['gemm_time_utilization'] * 100),
        'Memory: read %d B + write %d B = %d B | %.2f MB/s | cmd stall '
        '%.3f%% | busy duty %.2f%%' % (
            tot['rd_bytes'], tot['wr_bytes'], total_bytes,
            metrics['bw_mbps'], metrics['cmd_stall_ratio'] * 100,
            metrics['busy_duty'] * 100),
        '',
        '%-16s %12s %9s %9s %9s %8s %10s' % (
            'Stage', 'cycles', 'share%', 'GEMM%', 'MACact%',
            'descs', 'geom MACs'),
    ]
    for s in stages:
        lines.append(
            '%-16s %12d %8.2f%% %8.2f%% %8.2f%% %8d %10.2fG' % (
                s['stage'], s['cycles'],
                s['cycles'] * 100.0 / e2e_cycles,
                s['gemm_cycles'] * 100.0 / s['cycles'] if s['cycles']
                else 0.0,
                s['mac_active_cycles'] * MAC_PER_ACTIVE_CYCLE * 100.0 /
                (s['cycles'] * PEAK_MAC_PER_CYCLE) if s['cycles'] else 0.0,
                s['desc_count'], s['geom_macs'] / 1e9))
    text = '\n'.join(lines) + '\n'
    (REPORTS / 'perf_profile.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
