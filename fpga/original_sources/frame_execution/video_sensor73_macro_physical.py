"""Test the previously routed complete RAM/DSP macro floorplan, not fixed logic.

All ordinary LUT/FF placement remains free. The same73 map, original SDC and
pin/clock settings are retained. New placement and route must pass independently.
"""
import json,pickle
from pathlib import Path
from run_units import ROOT,sha
from video_sensor73_physical import apply_profile,BOARD,REFERENCE,PARENT
from review_input_isp_geometry import locations


def plan():
    key,g,pa=pickle.loads((BOARD/'sensor73-map-parse.pkl').read_bytes())
    assert key==sha(BOARD/'outflow/ti60f225_oob.map.v')
    oldkey,_,old,_,oldpa=pickle.loads((PARENT/'retained-audit-parse.pkl').read_bytes())
    assert oldkey[1]==sha(REFERENCE/'outflow/ti60f225_oob.map.v')
    prior=locations(REFERENCE/'outflow/ti60f225_oob.out.placeloc')
    oldanchors=locations(REFERENCE/'fixed_dsp.placeloc')
    ram={n for n,(k,p) in g.items() if k in ('EFX_RAM10','EFX_DPRAM10')}
    assert len(ram)==245 and ram=={n for n,(k,p) in old.items() if k in ('EFX_RAM10','EFX_DPRAM10')}
    clock_pins={'CLK','RCLK','WCLK','CLKA','CLKB'}
    for n in ram:
        assert g[n][0]==old[n][0]
        # INIT content changes are the intended sensor setting; all port
        # geometry/clock mode parameters must match the passed reference.
        assert {p:v for p,v in pa[n].items() if not p.startswith('INIT_')}=={p:v for p,v in oldpa[n].items() if not p.startswith('INIT_')},n
        assert {p:v for p,v in g[n][1].items() if p in clock_pins}=={p:v for p,v in old[n][1].items() if p in clock_pins},n
    assert len({tuple(prior[n]) for n in ram})==245
    assignments={**oldanchors,**{n:prior[n] for n in ram}}
    assert len(assignments)==533 and all(g[n][0] in ('EFX_RAM10','EFX_DPRAM10','EFX_DSP24','EFX_DSP48') for n in assignments)
    assert all(tuple(assignments[n])==tuple(v) for n,v in oldanchors.items())
    return assignments,dict(map_sha256=key,ram_cells=245,dsp_cells=288,
                           all_ram_modes_and_clock_connections_match=True,
                           extra_macro_anchors=129,ordinary_anchors=0)


def generate(root,reference):
    assignments,check=plan()
    current=locations(root/'fixed_dsp.placeloc')
    assert len(current)==288 and all(tuple(assignments[n])==tuple(p) for n,p in current.items())
    target=root/'fixed_dsp.placeloc'
    target.write_text(''.join(n+' '+' '.join(map(str,p))+'\n' for n,p in sorted(assignments.items())))
    report=dict(status='PRIOR_PASSED_MACRO_FLOORPLAN_TEST_NOT_RELEASE',
                assignments=assignments,total_hard_anchors=533,ordinary_anchors=0,
                assignments_sha256=sha(target),reference=str(REFERENCE),compatibility=check,
                rationale=['Keep all245 memory macros in their actually routed reference locations, instead of relocating119 of them.',
                           'This tests macro distribution jointly with free ordinary-cell placement; it does not lock LUTs/FFs or import old routing.',
                           'Current73 modes and clock connections are checked before starting. Global control fanout, congestion and STA still require fresh evidence.'],
                evidence={str(f):sha(f) for f in (Path(__file__),REFERENCE/'outflow/ti60f225_oob.out.placeloc',
                          REFERENCE/'fixed_dsp.placeloc',BOARD/'sensor73-map-audit.json',
                          ROOT/'tmp/board-test/video-service-review1/sensor73-identical-net-geometry-regressions.json')},hardware_access=False)
    (root/'cnn_ram_placement.json').write_text(json.dumps(report,indent=2))
    return report


def verify_placed(root):
    proof=json.loads((root/'cnn_ram_placement.json').read_text())
    for f,h in proof['evidence'].items():assert sha(f)==h,f
    assert proof['ordinary_anchors']==0 and proof['total_hard_anchors']==533
    loc=locations(root/'outflow/ti60f225_oob.out.placeloc')
    assert all(tuple(loc[n])==tuple(p) for n,p in proof['assignments'].items())
    return dict(status='SENSOR73_533_HARD_ANCHORS_PLACED_PASS',count=533,ordinary_anchors=0,
                placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'))


if __name__=='__main__':
    a,r=plan();print(json.dumps(dict(status='MACRO_FLOORPLAN_COMPATIBILITY_ONLY',**r,anchors=len(a))))
