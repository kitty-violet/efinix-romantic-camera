"""Extract a stable, drained boundary from an existing board observation.

Offline only. Requires two agreeing process pages at the end of the report;
live counts from different times are rejected instead of being labeled losses.
"""
import argparse, json
from pathlib import Path
from run_units import sha


def extract(report, manifest):
    if report.get('status') == 'FAIL_PREFIX_SAVED' or report.get('error'):
        raise ValueError('Failed collection')
    if report['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Different release')
    samples=report['register_samples']
    if not samples or samples[-1]['phase'] != 5 or samples[-1]['apb']['08']:
        raise ValueError('Service is not prepared and stopped')
    if report['control']['errors'] or samples[-1]['apb']['60']:
        raise ValueError('Controller or APB error')
    def recent(kind):
        rows=[p['fields'] for p in report['pages'] if p['page_kind']==kind]
        if len(rows)<2: raise ValueError('Two complete pages required: '+kind)
        return rows[-2:]
    processes=recent('process'); health=recent('health'); paths=recent('path')
    keys=('captured','compute_admitted','committed')
    for p in processes:
        if not p['status']&32 or p['status']&0xcf or any(s in (1,2,3) for s in p['owner_states']):
            raise ValueError('Pending frame, owner/pipeline error, or draining state')
        # Packed word: prepared[0], pattern[1], inhibit[2], discard[3],
        # clear[4], failed[5], state[8:6], fault_code[12:9].
        if not p['recovery_status']&1 or p['recovery_status']&0x1ffc:
            raise ValueError('Not prepared, recovery active, or display recovery fault')
    if any(processes[0][k]!=processes[1][k] for k in keys):
        raise ValueError('Counts changed between process pages')
    epochs=[h['reset_epochs'] for h in health]
    if epochs[0]!=epochs[1] or epochs[-1]==65535:
        raise ValueError('Reset changed or history saturated')
    if any(p['mode']!='LIVE_A' for p in paths):
        raise ValueError('Not LIVE_A')
    return dict(phase=5, mode='LIVE_A', work_drained=True, apb_errors=0, owner_error=False,
                control_errors=0, stable_process_counts=True, reset_epoch=epochs[-1],
                bitstream_sha256=manifest['bitstream_sha256'],
                **{k:processes[-1][k] for k in keys},
                raw_health_first_faults=health[-1]['first_faults'],
                scope='Stopped/drained admitted RGB batch boundary; not camera input total')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve existing evidence')
    m=json.loads(a.manifest.read_text(encoding='utf-8'));m['manifest_sha256']=sha(a.manifest)
    r=extract(json.loads(a.observation.read_text(encoding='utf-8')),m)
    r.update(observation=str(a.observation.resolve()),observation_sha256=sha(a.observation),extractor_sha256=sha(__file__))
    a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))


if __name__=='__main__':main()
