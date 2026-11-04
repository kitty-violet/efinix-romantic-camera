"""Keep only 36 measured arena-path cells at a previously legal placement.

No logic or clock constraints change. Require complete tiles/chains and exact
port/parameter identity after mapper renaming. All other ordinary logic is free.
"""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,EF,sha
from native_local_drivers import graph
from camera_joint_native import OUT
from probe_full_native_split_map import params
from fixed_control_ddr_reuse_locations import locations
from audit_canonical_multiseed import audit
REFERENCE=ROOT/'tmp/frame-execution/camera-recovery-route1/canonical-seed3'

def plan(current):
 previous=audit(REFERENCE);assert previous['current_timing_pass']
 old=Path(previous['physical']);an=old/'outflow/ti60f225_oob.map.v';bn=Path(current)/'outflow/ti60f225_oob.map.v'
 op=old/'outflow/ti60f225_oob.out.placeloc'
 at,bt=an.read_text(),bn.read_text();a,b=graph(at),graph(bt);ap,bp=params(at),params(bt);pos=locations(op)
 aliases={}
 for n in a.keys()&b.keys():
  if a[n][0]!=b[n][0] or ap.get(n,{})!=bp.get(n,{}):continue
  for p in OUT.get(a[n][0],[]):
   assert len(a[n][1].get(p,[]))==len(b[n][1].get(p,[])),(n,p)
   for x,y in zip(a[n][1].get(p,[]),b[n][1].get(p,[])):
    assert x not in aliases or aliases[x]==y,(n,p,x)
    aliases[x]=y
 tiles=collections.defaultdict(set);drivers={};loads=collections.defaultdict(set);neighbors=collections.defaultdict(set)
 for n,p in pos.items():tiles[p[:2]].add(n)
 for n,(k,pins) in a.items():
  for p,ts in pins.items():
   for t in ts:
    if p in OUT.get(k,[]):drivers[t]=n
    else:loads[t].add((n,p))
 for n,(k,p) in a.items():
  if k not in ('EFX_ADD','EFX_SRL8'):continue
  for t in p.get('CI' if k=='EFX_ADD' else 'D',[]):
   o=drivers.get(t)
   if o and a[o][0]==k:neighbors[n].add(o);neighbors[o].add(n)
 seeds={'LOCAL_DRIVER_086','LOCAL_DRIVER_088'}
 for n in tuple(seeds):
  assert a[n][0]=='EFX_FF'
  for t in a[n][1]['Q']:seeds.update(s for s,p in loads[t])
 chosen=set(seeds)
 while True:
  expanded=set(chosen)
  for n in chosen:
   assert n in a and n in pos,n
   expanded.update(tiles[pos[n][:2]]);expanded.update(neighbors[n])
  if expanded==chosen:break
  chosen=expanded
 assert len(chosen)==36 and collections.Counter(a[n][0] for n in chosen)=={'EFX_FF':2,'EFX_SRL8':34}
 for n in chosen:
  assert n in b and a[n][0]==b[n][0] and ap.get(n,{})==bp.get(n,{})
  assert {p:[aliases.get(t,t) for t in ts] for p,ts in a[n][1].items()}==b[n][1],n
  assert tiles[pos[n][:2]]<=chosen and neighbors[n]<=chosen
 # Both FFs were alone in the passed placement: no old pack partner is lost.
 assert all(len(tiles[pos[n][:2]])==1 for n in chosen)
 assignments={n:list(pos[n]) for n in sorted(chosen)}
 assert len(set(map(tuple,assignments.values())))==len(assignments)
 return dict(status='SMALL_COMPLETE_UNCHANGED_ISLAND_PROVED_NOT_TIMING_PASS',assignments=assignments,
  reference=str(REFERENCE),types={'EFX_FF':2,'EFX_SRL8':34},complete_tiles_and_chains=True,
  exact_all_ports_parameters_including_clock_ce_reset=True,logic_or_constraints_changed=False,
  evidence={str(p):sha(p) for p in [an,bn,op,REFERENCE/'result.json',Path(__file__),
   ROOT/'tmp/board-test/camera-factor-balanced-retained-hotspots1.json',
   ROOT/'tmp/board-test/camera-factor-balanced-release-preflight1.json',
   EF/'doc/topics/loc-rules.html']},hardware_access=False,board_release=False)

def generate(root):
 root=Path(root);proof=plan(root);target=root/'fixed_dsp.placeloc';prior=locations(target)
 assert len(prior)==401 and not (prior.keys()&proof['assignments'].keys())
 assert not ({p[:2] for p in prior.values()} & {tuple(p[:2]) for p in proof['assignments'].values()})
 before=sha(target);text=target.read_text();target.write_text(text+'# Complete unchanged 36-cell arena island; all other ordinary cells free.\n'+
  ''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in proof['assignments'].items()))
 assert all(locations(target)[n]==p for n,p in prior.items())
 proof.update(prior_assignment_sha256=before,assignments_sha256=sha(target),total_assignments=437)
 # Compose the CNN RAM proof with this separately audited append, preserving its lineage.
 ramfile=root/'cnn_ram_placement.json';ram=json.loads(ramfile.read_text());assert ram['assignments_sha256']==before
 ram['before_island_assignment_sha256']=before;ram['assignments_sha256']=sha(target)
 ram['island_generator_sha256']=sha(__file__);ramfile.write_text(json.dumps(ram,indent=2))
 (root/'arena_island_placement.json').write_text(json.dumps(proof,indent=2))
 (root/'arena_island_generator.py').write_bytes(Path(__file__).read_bytes())
 return proof

def verify_placed(root):
 root=Path(root);r=json.loads((root/'arena_island_placement.json').read_text())
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert sha(root/'fixed_dsp.placeloc')==r['assignments_sha256']
 assert sha(root/'arena_island_generator.py')==sha(__file__)
 actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
 for n,p in r['assignments'].items():assert actual[n]==tuple(p),(n,p,actual.get(n))
 return dict(status='36_CELL_UNCHANGED_COMPLETE_ISLAND_LOCATIONS_VERIFIED',count=36,
  assignment_proof_sha256=sha(root/'arena_island_placement.json'),placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists();r=plan(a.board);a.output.write_text(json.dumps(r,indent=2));print(r['status'],len(r['assignments']))
