"""Repair measured LUT nets without moving any sequential/control-set sites.

The rejected local plan remains immutable. All five FF-bearing moves are
removed, so its introduced reset-column assignments cannot recur. Only the
tool's actual placement and timing checks can approve the remaining LUT moves.
"""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import read,locations

REJECTED=ROOT/'tmp/frame-execution/control-ddr-local-physical1/pnr-seed1'

def plan(root):
    original=REJECTED/'reuse_placement.json'
    proof=json.loads(original.read_text())
    assert proof['status']=='SAME_MAP_LOCAL_REPAIR_NOT_TIMING_PASS'
    net=Path(root)/'outflow/ti60f225_oob.map.v'
    assert sha(net)==proof['new_map_sha256']
    cc=read(net)
    assignments=proof['assignments']
    removed=[];kept=[]
    for move in proof['moves']:
        if any(cc[n][0]!='EFX_LUT4' for n in move['cells']):
            for n in move['cells']:
                assignments[n]=[*move['before'],assignments[n][2]]
            removed.append(move)
        else:kept.append(move)
    base=Path(proof['reference']).parent/'outflow/ti60f225_oob.out.placeloc'
    # The reference's physical folder is recorded by its audit, not guessed.
    from audit_canonical_multiseed import audit
    base=Path(audit(Path(proof['reference']))['physical'])/'outflow/ti60f225_oob.out.placeloc'
    assert sha(base)==proof['old_placement_sha256']
    before=locations(base)
    changed=[n for n,pos in assignments.items() if tuple(pos)!=before[n]]
    assert len(removed)==5 and len(kept)==11 and len(changed)==11
    assert all(cc[n][0]=='EFX_LUT4' for n in changed)
    assert all(tuple(assignments[n])==before[n] for n in cc if cc[n][0]!='EFX_LUT4')
    assert len(set(map(tuple,assignments.values())))==len(assignments)
    assert 'LUT__171161' in changed
    proof.update(status='SAME_MAP_LUT_ONLY_REPAIR_NOT_TIMING_PASS',moves=kept,
        removed_register_tile_moves=removed,rejected_proof_sha256=sha(original),
        rejected_proof=str(original),generator_sha256=sha(__file__),
        all_sequential_sites_unchanged=True,changed_primitive_count=len(changed))
    return proof

def generate(root):
    root=Path(root);proof=plan(root);target=root/'fixed_dsp.placeloc'
    assert all(tuple(proof['assignments'][n])==v for n,v in locations(target).items())
    target.write_text('# Same-map LUT-only repair; all sequential/control-set sites unchanged.\n'+
        ''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in sorted(proof['assignments'].items())))
    proof['assignments_sha256']=sha(target)
    (root/'reuse_placement.json').write_text(json.dumps(proof,indent=2))
    (root/'reuse_placement_generator.py').write_bytes(Path(__file__).read_bytes())
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();proof=plan(a.board)
    a.output.write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:proof[k] for k in ('status','assignment_count','changed_primitive_count','all_sequential_sites_unchanged')}))
