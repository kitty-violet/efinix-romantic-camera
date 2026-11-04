"""Interpret the held last CNN issue-cycle count at a verified drained boundary.

Offline only. An in-flight APB+0x58 read is explicitly not a frame duration.
This counter includes issue/DMA waits but excludes acquisition/display latency.
"""
import argparse
import json
from pathlib import Path
from run_units import sha
from video_service_drained_receipt import extract


def budget(report, manifest):
    boundary = extract(report, manifest)
    samples = report['register_samples']
    if len(samples) < 2:
        raise ValueError('Two stopped register snapshots required')
    first, last = samples[-2:]
    for row in (first, last):
        if row['phase'] != 5 or row['apb']['08'] or row['apb']['60']:
            raise ValueError('In-flight or failed CNN count is not a frame duration')
    for field in ('58', '5c'):
        if first['apb'][field] != last['apb'][field]:
            raise ValueError('Last frame changed during stopped observation')
    cycles = last['apb']['58']
    if not boundary['committed'] or last['apb']['5c'] != boundary['committed']:
        raise ValueError('No completed CNN frame bound to this counter')
    if type(cycles) is not int or not 0 < cycles < 2**26:
        raise ValueError('Empty or invalid current 26-bit frame cycle count')
    if manifest['frequency_mhz'] != 150:
        raise ValueError('Wrong clock contract')
    seconds = cycles / 150_000_000
    return dict(status='DRAINED_LAST_CNN_ISSUE_CYCLE_MEASURED', cycles=cycles,
                issue_active_ms=seconds*1000, target_15fps_budget_cycles=10_000_000,
                issue_budget_headroom_cycles=10_000_000-cycles,
                isolated_issue_rate_ceiling_hz=1/seconds,
                committed=boundary['committed'], reset_epoch=boundary['reset_epoch'],
                scope='Last completed issue-active frame including its waits, not per-layer or end-to-end latency',
                proves_sustained_15fps=False)


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','observation','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve existing evidence')
    m=json.loads(a.manifest.read_text());m['manifest_sha256']=sha(a.manifest)
    r=budget(json.loads(a.observation.read_text()),m)
    r['evidence']={str(f.resolve()):sha(f) for f in (a.manifest,a.observation,Path(__file__))}
    a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))


if __name__=='__main__':main()
