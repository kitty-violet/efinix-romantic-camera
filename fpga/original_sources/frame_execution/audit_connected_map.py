"""Audit a completed frozen map without changing its original runner outcome.

LUT names are flattened: source-file attribution is diagnostic, not a count of
removable logic. RAM hierarchy remains intact and is counted independently.
"""
import argparse, collections, json, re
from pathlib import Path
from run_units import ROOT, sha
from map_connected_board import BASE

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    manifest=json.loads((root/'inputs.json').read_text())
    for record in manifest['inputs']:
        assert sha(record['snapshot'])==record['sha256'],record['snapshot']
    net=root/'outflow/ti60f225_oob.map.v'
    warn=root/'outflow/ti60f225_oob.warn.log'
    report=root/'outflow/ti60f225_oob.map.rpt'
    assert re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
    counts=collections.Counter();by_source=collections.defaultdict(collections.Counter)
    ram=collections.Counter();block=[];kind=None;instance=None
    for line in net.read_text().splitlines():
        # Efinity may put the opening parenthesis on the following line.
        match=re.match(r'\s*EFX_([A-Z0-9_]+)\s+(\S+)',line)
        if match:
            kind,instance=match.groups();counts[kind]+=1;block=[]
            if kind in ('RAM10','DPRAM10'):
                ram['/'.join(instance.lstrip('\\').split('/')[:3])]+=1
        if kind:
            block.append(line)
            if ';' in line:
                source=re.search(r'//\s+(.+?)\(\d+\)',line)
                by_source[Path(source[1]).name if source else 'unattributed'][kind]+=1
                kind=None
    pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
    old_text=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
    current_text=warn.read_text(errors='replace')
    inherited=set(re.findall(pattern,old_text,re.M))
    current=set(re.findall(pattern,current_text,re.M))
    unknown=current-inherited
    assert not unknown,sorted(unknown)
    assert 'EFX-0654' not in current_text,'combinational loop'
    report_counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',report.read_text(),re.M)}
    for key in ('RAM10','DPRAM10','DSP24','DSP48'):
        assert report_counts[key]==counts[key],key
    result=dict(status='FROZEN_MAP_AUDIT_ONLY_NOT_RELEASE',resources=report_counts,
        dumped_net_primitives=dict(counts),
        counting_boundary='Official synthesis report and emitted net differ for logic. Both preserved; neither is placed XLR. RAM/DSP agree.',
        physical_dsp=counts['DSP24']//2+counts['DSP48'],
        ram_blocks=counts['RAM10']+counts['DPRAM10'],
        ram_hierarchy=dict(ram.most_common()),
        source_attribution_partial={k:dict(v) for k,v in sorted(by_source.items(),key=lambda x:x[1]['LUT4'],reverse=True)},
        inherited_undriven_signatures=sorted(current),new_undriven=[],
        original_runner_failed=(root/'failure.json').exists(),
        inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(report),warning_sha256=sha(warn),
        audit_script_sha256=sha(__file__),resource_fit_proven=False,
        physical_timing_pass=False,full_baseline=False,board_download_allowed=False)
    assert counts==collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M))
    # Do not confuse the emitted net count with the official report count.
    result['supersedes_incomplete_audits']=['resource_audit.json','resource_audit_v2.json']
    out=root/'resource_audit_v3.json';assert not out.exists(),'preserve frozen audit'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','resources','physical_dsp','ram_blocks','original_runner_failed')}))
    print('RAM',json.dumps(result['ram_hierarchy']))
    print('SOURCE_PARTIAL',json.dumps(dict(list(result['source_attribution_partial'].items())[:24])))

if __name__=='__main__':main()
