"""Retain the unchanged arithmetic engine, with complete old tiles/chains.

Only ~15% of ordinary cells are constrained. The control plane, observation
logic and all remaining ordinary cells stay free. Never reuse routing traces.
"""
import json
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import locations
from native_local_drivers import graph
ANALYSIS=ROOT/'tmp/board-test/cnn-budget-stable-islands-analysis1.json'

def plan(current):
 r=json.loads(ANALYSIS.read_text())
 assert r['status']=='READ_ONLY_UNCHANGED_ISLAND_FEASIBILITY_NOT_APPLIED'
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert sha(Path(current)/'outflow/ti60f225_oob.map.v')==r['evidence'][str(ROOT/'tmp/frame-execution/cnn-run-budget-place2/outflow/ti60f225_oob.map.v')]
 old=ROOT/'tmp/frame-execution/camera-factor-compact-bound3'
 g=graph((old/'outflow/ti60f225_oob.map.v').read_text());pos=locations(old/'outflow/ti60f225_oob.out.placeloc')
 s=r['scopes']['engine_only']
 assert s['complete_tiles_and_chains'] and s['all_ports_including_clock_ce_reset_and_parameters_exact']
 assert s['rejected_components']==0 and s['ordinary_free']==51297
 chosen={n:pos[n] for n in s['retained_cells'] if g[n][0] in ('EFX_FF','EFX_LUT4','EFX_ADD','EFX_SRL8')}
 assert len(chosen)==9084
 prior=locations(Path(current)/'fixed_dsp.placeloc');assert len(prior)==437
 for n in s['retained_cells']:
  if g[n][0] not in ('EFX_FF','EFX_LUT4','EFX_ADD','EFX_SRL8'):
   assert n in prior and prior[n]==pos[n],(n,'hard anchor mismatch')
 target_tiles={p[:2] for p in chosen.values()}
 for n,p in prior.items():
  if p[:2] in target_tiles:
   assert n in s['retained_cells'] and p==pos[n],(n,'incompatible existing tile occupant')
 return dict(status='UNCHANGED_ENGINE_COMPLETE_TILES_PROVED_NOT_PHYSICAL_PASS',assignments=chosen,
  ordinary_cells_retained=len(chosen),ordinary_cells_free=s['ordinary_free'],
  data_clock_enable_reset_parameters_exact=True,complete_tiles_chains=True,existing_hard_and_shared_tile_anchors_compatible=True,
  reference=str(old),old_route_not_reused=True,
  evidence={**r['evidence'],str(ANALYSIS):sha(ANALYSIS),str(Path(__file__)):sha(__file__)},
  hardware_access=False,board_release=False)

def generate(root):
 root=Path(root);r=plan(root);target=root/'fixed_dsp.placeloc';before=sha(target);prior=locations(target)
 assert len(prior)==437
 additions={n:tuple(p) for n,p in r['assignments'].items() if n not in prior}
 for n,p in r['assignments'].items():
  if n in prior:assert prior[n]==tuple(p),(n,prior[n],p)
 # All already-fixed hard/arena tile occupants keep their proved coordinates.
 occupied={tuple(p):n for n,p in prior.items()}
 assert len(occupied)==len(prior)
 for n,p in additions.items():assert p not in occupied,(n,p,occupied.get(p))
 body=target.read_text()+'# Exact unchanged arithmetic engine complete tiles/chains; remaining ordinary logic free.\n'
 body+=''.join(f'{n} {x} {y} {z}\n' for n,(x,y,z) in sorted(additions.items()))
 target.write_text(body,encoding='utf8');after=sha(target)
 for name in ('cnn_ram_placement.json','arena_island_placement.json'):
  f=root/name;p=json.loads(f.read_text());assert p['assignments_sha256']==before
  p.update(before_engine_assignments_sha256=before,assignments_sha256=after,engine_generator_sha256=sha(__file__))
  f.write_text(json.dumps(p,indent=2),encoding='utf8')
 r.update(before_assignments_sha256=before,assignments_sha256=after,added_assignments=len(additions),total_assignments=len(prior)+len(additions))
 (root/'engine_island_placement.json').write_text(json.dumps(r,indent=2),encoding='utf8')
 (root/'engine_island_generator.py').write_bytes(Path(__file__).read_bytes())
 return r

def verify_placed(root):
 root=Path(root);r=json.loads((root/'engine_island_placement.json').read_text())
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert sha(root/'fixed_dsp.placeloc')==r['assignments_sha256']
 assert sha(root/'engine_island_generator.py')==sha(__file__)
 actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
 for n,p in r['assignments'].items():assert actual[n]==tuple(p),(n,p,actual.get(n))
 return dict(status='UNCHANGED_ENGINE_9084_ORDINARY_CELLS_LOCATIONS_VERIFIED',count=9084,
  assignment_proof_sha256=sha(root/'engine_island_placement.json'),placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)
