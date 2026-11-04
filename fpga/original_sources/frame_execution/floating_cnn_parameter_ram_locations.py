"""Free only29 optional parameter-RAM anchors beside the added observation RAM.

The84 high-bandwidth arena RAMs and288 complete DSP/cascade assignments stay
fixed. The parameter RAMs have no dedicated physical cascade ports. This is
an implementation choice, not a clock/pin exception or a functional change.
"""
import json
from pathlib import Path
from run_units import sha
from fixed_cnn_ram_locations import generate as original_generate,verify_placed as original_verify,memory_cells

def generate(root,reference):
 root=Path(root);r=original_generate(root,reference)
 original=r['assignments'];prefix='video_processing_pipeline/pipeline/cnn/core/'
 released={n:p for n,p in original.items() if n.startswith((prefix+'header/',prefix+'store/'))}
 kept={n:p for n,p in original.items() if n not in released}
 assert len(released)==29 and len(kept)==84
 assert all(n.startswith(prefix+'arena/') for n in kept)
 cells,_=memory_cells(root/'outflow/ti60f225_oob.map.v')
 assert all(cells[n]=='EFX_RAM10' for n in released)
 f=root/'fixed_dsp.placeloc';lines=f.read_text().splitlines()
 selected=[l for l in lines if len(w:=l.split())==4 and w[0] in released]
 assert len(selected)==29
 f.write_text('\n'.join(l for l in lines if l not in selected)+'\n')
 count=sum(len(l.split())==4 and not l.startswith('#') for l in f.read_text().splitlines())
 assert count==372
 r.update(status='CNN_PARAMETER_RAM_29_FREE_WITH_ARENA_AND_DSP_ANCHORS',
  assignments=kept,released_assignments=released,hard_primitive_assignments=372,
  ordinary_anchors=0,ram_has_physical_cascade_ports=False,
  assignments_sha256=sha(f),generator_sha256=sha(__file__),
  reference_generator_sha256=sha(Path(__file__).with_name('fixed_cnn_ram_locations.py')),
  rationale='The17 new observation RAMs occupy the same eastern RAM region as29 optional old parameter RAM coordinates. Release those29 only; keep84 arena RAMs and288 DSP/cascade assignments. Review global placement before routing.')
 (root/'cnn_ram_placement.json').write_text(json.dumps(r,indent=2))
 (root/'cnn_ram_generator.py').write_bytes(Path(__file__).read_bytes())
 return r

def verify_placed(root):
 root=Path(root);p=json.loads((root/'cnn_ram_placement.json').read_text())
 assert len(p['assignments'])==84 and len(p['released_assignments'])==29
 # Original checker binds unchanged RAM kinds/parameters, source/constraint
 # reference hashes and all remaining authored RAM coordinates.
 result=original_verify(root)
 placed={w[0]:list(map(int,w[1:])) for l in (root/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines()
         if len(w:=l.split())==4 and not l.startswith('#')}
 actual={n:placed[n] for n in p['released_assignments']}
 assert len({tuple(x) for x in actual.values()})==29 and all(x[2]==0 for x in actual.values())
 fixed={w[0]:list(map(int,w[1:])) for l in (root/'fixed_dsp.placeloc').read_text().splitlines()
        if len(w:=l.split())==4 and not l.startswith('#')}
 assert len(fixed)==372 and all(placed[n]==v for n,v in fixed.items())
 result.update(status='CNN_ARENA84_FIXED_PARAMETER29_FREE_AND_DSP288_CHECKED',
   fixed_arena_count=84,free_parameter_count=29,fixed_dsp_primitives=288,
   changed_parameter_locations=sum(actual[n]!=p['released_assignments'][n] for n in actual))
 return result
