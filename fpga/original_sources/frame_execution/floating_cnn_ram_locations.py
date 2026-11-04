"""Release optional historical CNN RAM anchors, retain complete DSP cascades.

The113 unchanged RAM primitives have no physical cascade ports. Their old
coordinates were an implementation preference, not a pin/clock constraint.
Official pack/place and strict routed STA remain required for new locations.
"""
import json
from pathlib import Path
from run_units import sha
from fixed_cnn_ram_locations import generate as old_generate,memory_cells
def generate(root,reference):
 root=Path(root);r=old_generate(root,reference)
 assignments=r['assignments'];assert len(assignments)==113
 f=root/'fixed_dsp.placeloc';before=f.read_text();lines=before.splitlines()
 removed=[l for l in lines if len(w:=l.split())==4 and w[0] in assignments]
 assert len(removed)==113
 retained=[l for l in lines if l not in removed]
 n=sum(1 for l in retained if len(l.split())==4 and not l.startswith('#'));assert n==288,n
 f.write_text('\n'.join(retained)+'\n')
 r.update(status='CNN_RAM_FREE_PLACEMENT_WITH_UNCHANGED_DSP_CASCADES',released_assignments=assignments,assignments={},
  hard_primitive_assignments=288,ordinary_anchors=0,ram_has_physical_cascade_ports=False,
  assignments_sha256=sha(f),generator_sha256=sha(__file__),reference_generator_sha256=sha(Path(__file__).with_name('fixed_cnn_ram_locations.py')),
  rationale='Remove optional historical locations of113 separate CNN RAMs so placement can balance new video/observation connectivity; retain all288 DSP primitives/cascades and original clock/pin constraints.')
 (root/'cnn_ram_placement.json').write_text(json.dumps(r,indent=2));(root/'cnn_ram_generator.py').write_bytes(Path(__file__).read_bytes());return r
def verify_placed(root):
 root=Path(root);p=json.loads((root/'cnn_ram_placement.json').read_text());old=Path(p['reference_physical'])
 assert p['assignments']=={} and len(p['released_assignments'])==113
 assert sha(root/'outflow/ti60f225_oob.map.v')==p['candidate_map_sha256']
 assert sha(root/'fixed_dsp.placeloc')==p['assignments_sha256']
 before=memory_cells(old/'outflow/ti60f225_oob.map.v');after=memory_cells(root/'outflow/ti60f225_oob.map.v');assert before==after
 placed={w[0]:list(map(int,w[1:])) for l in (root/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines() if len(w:=l.split())==4 and not l.startswith('#')}
 actual={n:placed[n] for n in after[0]};assert len(actual)==113 and len({tuple(x) for x in actual.values()})==113
 assert all(x[2]==0 for x in actual.values())
 fixed={w[0]:list(map(int,w[1:])) for l in (root/'fixed_dsp.placeloc').read_text().splitlines() if len(w:=l.split())==4 and not l.startswith('#')}
 assert len(fixed)==288 and all(placed[n]==loc for n,loc in fixed.items())
 return dict(status='CNN_RAM_FREE_LOCATIONS_AND_DSP_ANCHORS_CHECKED',ram_count=113,changed_ram_locations=sum(actual[n]!=p['released_assignments'][n] for n in actual),fixed_dsp_primitives=288,
  placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)
