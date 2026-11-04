"""Analyze only completed old-run command windows, never extrapolate a frame."""
import argparse
import collections
import csv
import json
from pathlib import Path
from run_units import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
    partial=json.loads((root/'partial_measurement.json').read_text())
    csvfile=root/'partial_prefix/cnn_command_events.csv'
    assert sha(csvfile)==partial['input_hashes']['cnn_command_events.csv']
    payload=ROOT/'tmp/frame-execution/board-payload2'
    package=json.loads((payload/'payload.json').read_text())
    binary=payload/'compute.bin';item=next(x for x in package['payloads'] if x['file']=='compute.bin')
    assert sha(binary)==item['sha256']
    data=binary.read_bytes();mapping={}
    for off in range(0,len(data),64):
        value=int.from_bytes(data[off:off+64],'little')
        index=(value>>14)&0x7fff;node=(value>>29)&31
        assert index not in mapping;mapping[index]=node
    with csvfile.open(newline='') as f:rows=[{k:int(v) for k,v in x.items()} for x in csv.DictReader(f)]
    assert len(rows)==partial['completed_compute_commands'] and len({x['frame'] for x in rows})==1
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):
        assert row['end_cycle']-row['begin_cycle']==row['elapsed_cycles']
        if i:assert row['begin_cycle']>=rows[i-1]['end_cycle']
        groups[mapping[row['program_index']]].append(row['elapsed_cycles'])
    span=rows[-1]['end_cycle']-rows[0]['begin_cycle'];inside=sum(x['elapsed_cycles'] for x in rows)
    result=dict(status='COMPLETED_PREFIX_COMMAND_ANALYSIS_ONLY',complete_frame=False,commands=len(rows),
        observed_cycle_span=span,command_window_cycles=inside,outside_command_window_cycles=span-inside,
        per_node=[dict(node=n,commands=len(v),window_cycles=sum(v),min_cycles=min(v),max_cycles=max(v)) for n,v in sorted(groups.items())],
        source_csv_sha256=sha(csvfile),descriptor_sha256=sha(binary),
        limits=['First observed command through last completed command only.',
                'Command windows include stalls; outside windows are not automatically DDR transfer time.',
                'Pending command, remaining network, output and display are excluded; no full-frame extrapolation.'])
    (root/'partial_command_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
