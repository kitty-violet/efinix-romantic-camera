"""Add complete CNN register/driver tiles for the measured address/enable paths.

The LUT function need not match the old map: only its legal tile role is
reused. No new netlist connectivity or truth table is substituted.
"""
import argparse,json,collections
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import plan as base_plan,read,locations,REFERENCE,PREFIX
from audit_canonical_multiseed import audit

def plan(root):
    proof=base_plan(root);assigned={n:tuple(v) for n,v in proof['assignments'].items()}
    old=Path(audit(REFERENCE)['physical']);a=read(old/'outflow/ti60f225_oob.map.v');b=read(Path(root)/'outflow/ti60f225_oob.map.v')
    pos=locations(old/'outflow/ti60f225_oob.out.placeloc');tiles=collections.defaultdict(list)
    for n,loc in pos.items():tiles[loc[:2]].append(n)
    drivers=collections.defaultdict(list)
    for n,(k,p,par) in b.items():
        if k=='EFX_LUT4':drivers[p['O']].append(n)
    occupied={p[:2] for p in assigned.values()};pairs=[]
    for n in sorted(a.keys()&b.keys()):
        if n in assigned or not n.startswith(PREFIX) or a[n]!=b[n] or b[n][0]!='EFX_FF' or n not in pos:continue
        loc=pos[n];other=[o for o in tiles[loc[:2]] if o!=n]
        if loc[:2] in occupied or len(other)!=1 or a.get(other[0],('unknown',))[0]!='EFX_LUT4':continue
        old_driver=other[0];newdrivers=drivers.get(b[n][1]['D'],[])
        if len(newdrivers)!=1 or newdrivers[0] in assigned:continue
        driver=newdrivers[0]
        assert a[old_driver][1]['O']==a[n][1]['D']
        assert loc[2]==2 and pos[old_driver][2]==0
        assert b[n][2]['D_POLARITY']=="1'b1"
        assigned[n]=loc;assigned[driver]=pos[old_driver];occupied.add(loc[:2])
        pairs.append(dict(register=n,new_driver=driver,old_driver=old_driver,register_location=loc,driver_location=pos[old_driver],
            register_ports_parameters_identical=True,driver_function_equivalence_claimed=False))
    assert len(pairs)>100
    proof.update(status='COMPLETE_TILES_WITH_CNN_REGISTER_DRIVER_ANCHORS_NOT_TIMING_PASS',
        assignments=assigned,assignment_count=len(assigned),additional_register_driver_pairs=pairs,
        base_generator_sha256=proof['generator_sha256'],generator_sha256=sha(__file__),
        types=dict(collections.Counter(b[n][0] for n in assigned)))
    assert len(set(assigned.values()))==len(assigned)
    assert PREFIX+'core/dut/engine/mac/source/bases[30]~FF' in assigned
    return proof

def generate(root):
    root=Path(root);proof=plan(root);target=root/'fixed_dsp.placeloc'
    assert set(locations(target))<=proof['assignments'].keys()
    target.write_text('# Complete old legal tiles with unchanged new mapped logic.\n'+''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in sorted(proof['assignments'].items())))
    proof['assignments_sha256']=sha(target)
    (root/'reuse_placement.json').write_text(json.dumps(proof,indent=2))
    (root/'reuse_placement_generator.py').write_bytes(Path(__file__).read_bytes())
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();r=plan(a.board);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(dict(count=r['assignment_count'],pairs=len(r['additional_register_driver_pairs']),types=r['types'])))
