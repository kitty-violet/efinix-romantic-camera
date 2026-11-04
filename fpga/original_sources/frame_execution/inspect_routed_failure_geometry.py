"""Compare actual reported failing nets against a new same-map placement.

Geometry is diagnostic only: shorter endpoints never establish a route delay.
"""
import argparse
import json
import re
from pathlib import Path
from run_units import ROOT, sha
from review_local_decode_paths import locations


def inspect(failed, target):
    report = failed / 'outflow/ti60f225_oob.timing.rpt'
    assert sha(failed/'outflow/ti60f225_oob.map.v') == sha(target/'outflow/ti60f225_oob.map.v')
    loc = locations(target)
    paths = []
    for block in re.split(r'\+\+\+\+ Path \d+ \+{5,}', report.read_text())[1:]:
        match = re.search(r'^Slack\s*:\s*(-[\d.]+)', block, re.M)
        if not match:
            continue
        def field(name):
            return re.search(r'^'+name+r'\s*:\s*(.*)', block, re.M)[1].strip()
        data = block.split('\nData Path\n')[1].split('\nCapture Clock Path\n')[0]
        rows = []
        for line in data.splitlines():
            m = re.match(r'^\s*(\S+)\s+(EFX_\w+|net)\s+([-+\d.]+)\s+([-+\d.]+)\s+(\d+)\s+\((\d+),(\d+)\)', line)
            if m:
                rows.append(dict(name=m[1], kind=m[2], delay_ns=float(m[3]), fanout=int(m[5]), xy=[int(m[6]),int(m[7])]))
        expected = float(re.search(r'^\+ Clock To Q \+ Data Path Delay\s*:\s*([\d.]+)', block, re.M)[1])
        assert rows and abs(sum(r['delay_ns'] for r in rows)-expected) <= .002
        nets = []
        for i, row in enumerate(rows):
            if row['kind'] != 'net':
                continue
            assert 0 < i < len(rows)-1
            assert rows[i-1]['kind'] != 'net' and rows[i+1]['kind'] != 'net'
            src, dst = rows[i-1]['name'].split('|')[0], rows[i+1]['name'].split('|')[0]
            assert src in loc and dst in loc, (src, dst)
            nets.append(dict(signal=row['name'],start=src,end=dst,fanout=row['fanout'],old_delay_ns=row['delay_ns'],
                old_distance=sum(abs(a-b) for a,b in zip(row['xy'],rows[i+1]['xy'])),
                new_distance=sum(abs(a-b) for a,b in zip(loc[src][:2],loc[dst][:2])),
                new_start_xyz=loc[src],new_end_xyz=loc[dst]))
        paths.append(dict(slack_ns=float(match[1]),start=field('Path Begin'),end=field('Path End'),
            data_delay_ns=expected,net_delay_ns=round(sum(n['old_delay_ns'] for n in nets),3),
            old_distance=sum(n['old_distance'] for n in nets),new_distance=sum(n['new_distance'] for n in nets),nets=nets))
    assert paths
    return dict(source=str(report),source_sha256=sha(report),paths=paths,
        shortened_paths=sum(p['new_distance'] < p['old_distance'] for p in paths),
        path_count=len(paths),reported_paths_only=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--failed',type=Path,nargs='+',required=True)
    p.add_argument('--placement',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();target=a.placement.resolve();out=a.output.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    comparisons=[inspect(f.resolve(),target) for f in a.failed]
    result=dict(placement=str(target),placement_sha256=sha(target/'outflow/ti60f225_oob.out.placeloc'),
        map_sha256=sha(target/'outflow/ti60f225_oob.map.v'),script_sha256=sha(__file__),comparisons=comparisons,
        scope='Actual prior reported failures and same-map new geometry; not a routing-delay prediction.',
        hardware_access=False,board_release=False)
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    for comparison in comparisons:
        print(json.dumps(dict(source=comparison['source'],shortened_paths=comparison['shortened_paths'],
            path_count=comparison['path_count'],paths=[dict(slack_ns=r['slack_ns'],old_distance=r['old_distance'],
                new_distance=r['new_distance'],worst_net=max(r['nets'],key=lambda n:n['old_delay_ns'])) for r in comparison['paths']])))

if __name__=='__main__':main()
