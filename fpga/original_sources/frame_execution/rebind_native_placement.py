"""Copy a finished placement; rebind only Gray endpoint instance names.

The old placement and its warnings are preserved. No netlist, coordinates,
timing values, clock relationships, pin settings or placement settings change.
This is a route starting point, not corrected placement timing evidence.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_snapshot_cells import cells

SOURCE=ROOT/'tmp/frame-execution/native-split-place1'

def translate(sdc,net,placed):
    cc=cells(net);mapping={};buses=0
    actual={line.split()[0] for line in placed.splitlines() if line and not line.startswith('#')}
    def line_replace(line):
        nonlocal buses
        if not line.startswith('set_bus_skew '):return line
        buses+=1
        assert line.endswith(' 2.500')
        def pin(match):
            old=match[0];name,port=old.rsplit('|',1)
            assert port in ('CLK','D')
            candidates=[n for n in actual if n==name or re.fullmatch(r'(?:protected_inst/)+',n[:-len(name)]) and n.endswith(name)]
            assert len(candidates)==1,(old,candidates)
            target=candidates[0];assert target.endswith('~FF')
            if target in cc:assert cc[target][0]=='EFX_FF' and port in cc[target][1]
            new=target+'|'+port
            mapping[old]=new;return new
        return re.sub(r'u_ddr3_top/controller/[^ {}]+\|(?:CLK|D)',pin,line)
    changed='\n'.join(line_replace(x) for x in sdc.split('\n'))
    assert buses==10 and len(mapping)==140
    assert len({n for n in mapping.values() if n.endswith('|D')})==70
    restored=changed
    for old,new in sorted(mapping.items(),key=lambda x:-len(x[1])):
        if old!=new:restored=restored.replace(new,old)
    assert restored==sdc,'Changed more than exact instance names'
    return changed,mapping

def audit(root):
    record=json.loads((root/'endpoint_rebinding.json').read_text())
    assert record['source']==str(SOURCE)
    for path,h in record['original'].items():assert sha(path)==h,path
    before=(SOURCE/'ti60f225_oob.sdc').read_text()
    expected,mapping=translate(before,(root/'outflow/ti60f225_oob.map.v').read_text(),(root/'outflow/ti60f225_oob.out.placeloc').read_text())
    assert expected==(root/'ti60f225_oob.sdc').read_text() and mapping==record['pins']
    for name,h in record['geometry'].items():assert sha(root/name)==h==sha(SOURCE/name),name
    assert record['script_sha256']==sha(__file__)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume-copy',action='store_true');a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
    if root.exists():
        assert a.resume_copy and not (root/'endpoint_rebinding.json').exists()
        assert sha(root/'physical_inputs.json')==sha(SOURCE/'physical_inputs.json')
    else:assert not a.resume_copy
    result=json.loads((SOURCE/'result.json').read_text())
    assert result['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert not (SOURCE/'outflow/ti60f225_oob.troutingtraces').exists()
    oldmeta=json.loads((SOURCE/'physical_inputs.json').read_text())
    for row in oldmeta['inputs']:assert sha(row['snapshot'])==row['sha256']
    sdc,pins=translate((SOURCE/'ti60f225_oob.sdc').read_text(),(SOURCE/'outflow/ti60f225_oob.map.v').read_text(),(SOURCE/'outflow/ti60f225_oob.out.placeloc').read_text())
    if not root.exists():shutil.copytree(SOURCE,root)
    (root/'ti60f225_oob.sdc').write_text(sdc,encoding='utf8')
    for row in oldmeta['inputs']:
        path=Path(row['snapshot']);row['snapshot']=str(root/path.relative_to(SOURCE));row['sha256']=sha(row['snapshot'])
    oldmeta['placement_origin']=str(SOURCE)
    oldmeta['endpoint_rebinding_sha256']=None
    oldmeta['clock_profile']['before_endpoint_rebinding_sha256']=oldmeta['clock_profile']['after_sha256']
    oldmeta['clock_profile']['after_sha256']=sha(root/'ti60f225_oob.sdc')
    record=dict(status='EXACT_GRAY_ENDPOINT_NAME_REBINDING_NOT_TIMING_PASS',source=str(SOURCE),
        script_sha256=sha(__file__),pins=pins,buses=10,destination_bits=70,
        geometry={n:sha(SOURCE/n) for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb',
            'outflow/ti60f225_oob.place','outflow/ti60f225_oob.out.placeloc','work_pnr/ti60f225_oob.net_proto',
            'fixed_dsp.placeloc','ti60f225_oob.xml','ti60f225_oob.peri.xml','efx_pnr_settings.ini')},
        original={str(SOURCE/n):sha(SOURCE/n) for n in ('physical_inputs.json','result.json',
            'ti60f225_oob.sdc','placement_review_inputs.json','outflow/ti60f225_oob.place.out')},
        timing_values_unchanged=True,clock_relationships_unchanged=True,placement_rerun=False,
        rationale='Restore ten existing2.500ns Gray constraints onto renamed mapped FFs before route; placement-only estimates remain historical. Final route, original setup/hold and nonempty Gray evidence are mandatory.',
        hardware_access=False,board_release=False)
    (root/'endpoint_rebinding.json').write_text(json.dumps(record,indent=2))
    oldmeta['endpoint_rebinding_sha256']=sha(root/'endpoint_rebinding.json')
    (root/'physical_inputs.json').write_text(json.dumps(oldmeta,indent=2))
    # Preserve exact generated placement wrapper ancestry in the new folder.
    for suffix in ('-native-wrapper','-place-wrapper','-place-driver'):
        for ext in ('.py','.json'):
            shutil.copy2(SOURCE.parent/(SOURCE.name+suffix+ext),root.parent/(root.name+suffix+ext))
    audit(root);print('GRAY_REBINDING_PREPARED',sum(a!=b for a,b in pins.items()),flush=True)

if __name__=='__main__':main()
