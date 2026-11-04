"""Preserve proved stable tiles outside changed-cell physical neighborhoods.

Leave 15896 ordinary cells and every changed tile free, not a full placement
freeze. The unchanged hard cells sharing retained components are included.
Old routing is never imported. All resources and strict STA remain required.
"""
import json
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import locations
from native_local_drivers import graph
ANALYSIS=ROOT/'tmp/board-test/cnn-budget-physical-neighborhoods1.json'

def plan(current):
 r=json.loads(ANALYSIS.read_text())
 assert r['status']=='READ_ONLY_UNCHANGED_ISLAND_FEASIBILITY_NOT_APPLIED'
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert sha(Path(current)/'outflow/ti60f225_oob.map.v')==r['evidence'][str(ROOT/'tmp/frame-execution/cnn-run-budget-place2/outflow/ti60f225_oob.map.v')]
 old=ROOT/'tmp/frame-execution/camera-factor-compact-bound3'
 pos=locations(old/'outflow/ti60f225_oob.out.placeloc')
 s=r['scopes']['16'];assert s['ordinary_cells']==44485 and s['ordinary_free']==15896
 assert s['complete_tiles_and_chains'] and s['all_ports_including_clock_ce_reset_and_parameters_exact']
 chosen={n:pos[n] for n in s['retained_cells']}
 assert len(chosen)==44895
 prior=locations(Path(current)/'fixed_dsp.placeloc');assert len(prior)==437
 tiles={p[:2] for p in chosen.values()}
 for n,p in prior.items():
  if p[:2] in tiles:assert n in chosen and p==chosen[n],(n,'incompatible shared anchor')
 return dict(status='UNCHANGED_NEIGHBORHOODS_PROVED_NOT_PHYSICAL_PASS',assignments=chosen,
  ordinary_cells_retained=44485,ordinary_cells_free=15896,hard_cells_retained=410,
  radius=16,changed_cells=len(r['changed_cells']),complete_tiles_chains=True,
  data_clock_enable_reset_parameters_exact=True,reference=str(old),old_route_not_reused=True,
  evidence={**r['evidence'],str(ANALYSIS):sha(ANALYSIS),str(Path(__file__)):sha(__file__)},
  hardware_access=False,board_release=False)

def generate(root):
 root=Path(root);r=plan(root);target=root/'fixed_dsp.placeloc';before=sha(target);prior=locations(target)
 additions={n:tuple(p) for n,p in r['assignments'].items() if n not in prior}
 for n,p in r['assignments'].items():
  if n in prior:assert prior[n]==tuple(p),(n,prior[n],p)
 occupied={tuple(p):n for n,p in prior.items()};assert len(occupied)==len(prior)
 for n,p in additions.items():assert p not in occupied,(n,p,occupied.get(p))
 body=target.read_text()+'# Stable complete tiles outside radius16 changed-cell neighborhoods.\n'
 body+=''.join(f'{n} {x} {y} {z}\n' for n,(x,y,z) in sorted(additions.items()))
 target.write_text(body,encoding='utf8');after=sha(target)
 for name in ('cnn_ram_placement.json','arena_island_placement.json'):
  f=root/name;p=json.loads(f.read_text());assert p['assignments_sha256']==before
  p.update(before_neighborhood_assignments_sha256=before,assignments_sha256=after,neighborhood_generator_sha256=sha(__file__))
  f.write_text(json.dumps(p,indent=2),encoding='utf8')
 r.update(before_assignments_sha256=before,assignments_sha256=after,added_assignments=len(additions),total_assignments=len(prior)+len(additions))
 (root/'neighborhood_placement.json').write_text(json.dumps(r,indent=2),encoding='utf8')
 (root/'neighborhood_generator.py').write_bytes(Path(__file__).read_bytes());return r

def verify_placed(root):
 root=Path(root);r=json.loads((root/'neighborhood_placement.json').read_text())
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert sha(root/'fixed_dsp.placeloc')==r['assignments_sha256']
 assert sha(root/'neighborhood_generator.py')==sha(__file__)
 actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
 for n,p in r['assignments'].items():assert actual[n]==tuple(p),(n,p,actual.get(n))
 return dict(status='UNCHANGED_NEIGHBORHOOD_LOCATIONS_VERIFIED',count=44485,hard_count=410,
  assignment_proof_sha256=sha(root/'neighborhood_placement.json'),placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)
