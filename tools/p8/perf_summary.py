#!/usr/bin/env python
"""P8-0: one-page HeatViT RTL performance summary from existing artifacts.

Combines the descriptor ROM geometry (theoretical MACs), the final P7-5
e2e cycle counts, the routed utilization report and the timing summary
into build/reports/perf_summary.{json,txt}.  When build/reports/
perf_profile.json exists (P8-1 post-processing of the instrumented e2e
simulation), its measured MAC / bandwidth numbers are merged in.

Usage:
  python tools/p8/perf_summary.py
"""

import json
import re

from _perf_common import (
    CLOCK_MHZ,
    CP_NAMES,
    PEAK_MAC_PER_CYCLE,
    REPORTS,
    REPO,
    e2e_macs_per_stage,
    load_e2e_runs,
    parse_timing,
    parse_util,
    unpruned_macs_per_stage,
)

PROFILE = REPORTS / 'perf_profile.json'
UTIL_P75 = REPORTS / 'p7_5_p6_impl_util.txt'
TIMING_P75 = REPORTS / 'p7_5_p6_timing_summary.txt'
POWER_REF = REPORTS / 'p6_power.txt'


def load_power_reference():
    """P7-4-era vectorless power report; reference only, clearly labeled."""
    if not POWER_REF.exists():
        return None
    out = {}
    for line in POWER_REF.read_text(errors='replace').splitlines():
        m_total = re.match(r'\|\s*Total On-Chip Power \(W\)\s*\|\s*([\d.]+)',
                           line)
        m_dyn = re.match(r'\|\s*Dynamic \(W\)\s*\|\s*([\d.]+)', line)
        if m_total:
            out['total_w'] = float(m_total.group(1))
        if m_dyn:
            out['dynamic_w'] = float(m_dyn.group(1))
    out['note'] = 'P7-4 routed design, vectorless (no SAIF), low confidence'
    return out


def fmt_macs(v):
    return '%.2f G' % (v / 1e9)


def main():
    e2e = load_e2e_runs()
    runs = e2e['runs']
    token_counts = e2e.get('vectors', {}).get('token_counts', [197, 100, 55, 28])

    unpruned = unpruned_macs_per_stage()
    e2e_geom = e2e_macs_per_stage()
    unpruned_total = sum(unpruned)
    e2e_geom_total = sum(e2e_geom)

    util = parse_util(UTIL_P75)
    timing = parse_timing(TIMING_P75)
    power_ref = load_power_reference()

    profile = None
    if PROFILE.exists():
        profile = json.loads(PROFILE.read_text(encoding='utf-8'))

    summary = {
        'clock_mhz': CLOCK_MHZ,
        'peak_mac_per_cycle': PEAK_MAC_PER_CYCLE,
        'peak_gmac_per_s': PEAK_MAC_PER_CYCLE * CLOCK_MHZ * 1e6 / 1e9,
        'token_counts': token_counts,
        'macs': {
            'unpruned_total': unpruned_total,
            'e2e_geometry_total': e2e_geom_total,
            'per_stage': [
                {'stage': CP_NAMES[i], 'unpruned': unpruned[i],
                 'e2e_geometry': e2e_geom[i]} for i in range(len(CP_NAMES))
            ],
        },
        'runs': {},
        'resources': util,
        'timing': timing,
        'power_reference': power_ref,
    }
    for mask, run in sorted(runs.items(), key=lambda kv: int(kv[0])):
        cycles = run['cycles']
        summary['runs'][mask] = {
            'stall_mask': int(mask),
            'cycles': cycles,
            'status': run.get('status'),
            'fps': CLOCK_MHZ * 1e6 / cycles,
            'seconds_per_image': cycles / (CLOCK_MHZ * 1e6),
            'effective_gmac_per_s':
                e2e_geom_total / cycles * CLOCK_MHZ * 1e6 / 1e9,
            'mac_utilization_estimate':
                e2e_geom_total / (cycles * PEAK_MAC_PER_CYCLE),
        }

    if profile:
        summary['measured'] = profile['metrics']
        summary['measured_stages'] = profile['stages']
        for mask, run in summary['runs'].items():
            if str(run['stall_mask']) == '0' and 'measured' in summary:
                m = summary['measured']
                run['measured_gmac_per_s'] = \
                    m['mac_ops'] / run['cycles'] * CLOCK_MHZ * 1e6 / 1e9
                run['measured_mac_utilization'] = m['mac_utilization']

    (REPORTS / 'perf_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    # One-page human-readable report.
    lines = [
        'P8 HeatViT RTL performance summary '
        '(xc7k325tfbg900-3, %g MHz)' % CLOCK_MHZ,
        '',
        'Throughput (e2e, token pruning active: %s)' %
        (' -> '.join(str(t) for t in token_counts)),
    ]
    for mask, run in summary['runs'].items():
        lines.append(
            '  STALL_MASK=%s: %10d cycles | %5.2f s/image | %5.3f FPS' % (
                mask, run['cycles'], run['seconds_per_image'], run['fps']))
    lines += [
        '',
        'Compute (3 banks x 8x8 int8 = %d MAC/cycle = %.1f GMAC/s peak)'
        % (PEAK_MAC_PER_CYCLE, summary['peak_gmac_per_s']),
        '  theoretical MACs, unpruned (197 tokens everywhere):   %s'
        % fmt_macs(unpruned_total),
        '  theoretical MACs, e2e live token counts %s: %s'
        % ('/'.join(str(t) for t in token_counts), fmt_macs(e2e_geom_total)),
    ]
    if profile:
        m = profile['metrics']
        lines += [
            '  measured active MACs (RTL mac_active counters):     %s'
            % fmt_macs(m['mac_ops']),
            '  MAC utilization vs peak (whole run):                %.2f%%'
            % (m['mac_utilization'] * 100),
            '  MAC utilization within GEMM descriptors:            %.2f%%'
            % (m['gemm_time_utilization'] * 100),
            '  external memory traffic: read %d B, write %d B '
            '(%.2f MB/s @100 MHz)' % (
                m['rd_bytes'], m['wr_bytes'], m['bw_mbps']),
            '  command stall ratio: %.3f%% | busy duty: %.2f%%' % (
                m['cmd_stall_ratio'] * 100, m['busy_duty'] * 100),
        ]
    else:
        lines.append('  (measured numbers pending: run the instrumented '
                     'e2e sim + tools/p8/perf_parse_log.py)')
    if util:
        lines += [
            '',
            'Resources (routed, P7-5): LUT %d (%.1f%%), FF %d, BRAM %d, '
            'DSP %d' % (
                util.get('lut', {}).get('used', '?'),
                util.get('lut', {}).get('used', 0) * 100.0 /
                util.get('lut', {}).get('available', 1),
                util.get('ff', {}).get('used', '?'),
                util.get('bram_tile', {}).get('used', '?'),
                util.get('dsp', {}).get('used', '?')),
        ]
    if timing:
        lines.append('Timing: WNS %+.3f ns, TNS %.3f ns (%s)' % (
            timing['wns_ns'], timing['tns_ns'],
            'MET' if timing['met'] else 'VIOLATED'))
    if power_ref:
        lines.append('Power (reference, vectorless): %.3f W total / '
                     '%.3f W dynamic' % (
                         power_ref.get('total_w', float('nan')),
                         power_ref.get('dynamic_w', float('nan'))))
    text = '\n'.join(lines) + '\n'
    (REPORTS / 'perf_summary.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
