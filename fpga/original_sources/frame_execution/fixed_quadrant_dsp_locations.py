"""Place whole eight-tap chains by half-height region with central shared RQ.

Uses exactly the previously verified 160 Ti60 DSP sites and sub-sites. Only
whole cascade chains move; this is a placement hypothesis, not timing proof.
"""
import collections,json
from pathlib import Path
from run_units import sha
from fixed_board_dsp_locations import generate as original

def generate(root,rq_only=False):
    assert not rq_only
    root=Path(root);proof=original(root)
    file=root/'fixed_dsp.placeloc'
    before={w[0]:tuple(map(int,w[1:])) for line in file.read_text().splitlines() if len(w:=line.split())==4}
    sites=collections.defaultdict(set)
    for x,y,sub in before.values():sites[x].add(y)
    xs=sorted(sites);assert len(xs)==10 and all(len(v)==16 for v in sites.values())
    assert all(all(b-a==20 for a,b in zip(sorted(ys),sorted(ys)[1:])) for ys in sites.values())
    prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
    after={};groups=[]
    for row in range(4):
        columns=xs[(row//2)*5:(row//2+1)*5];half=row%2
        for pair,x in enumerate([columns[i] for i in (0,1,3,4)]):
            ys=sorted(sites[x])[half*8:(half+1)*8]
            for tap,y in enumerate(ys):
                for sub in (0,1):
                    name=prefix+f'mac/arithmetic/ROW[{row}].CHANNEL[{pair*2+sub}].TAP[{tap}].mac'
                    assert name in before;after[name]=(x,y,sub)
        x=columns[2];ys=sorted(sites[x])[half*8:(half+1)*8]
        for channel,y in enumerate(ys):after[prefix+f'rq/LANE[{row*8+channel}].rq/multiply_add']=(x,y,0)
        groups.append(dict(row=row,mac_columns=[columns[i] for i in (0,1,3,4)],rq_column=x,y_min=min(ys),y_max=max(ys)))
    assert set(after)==set(before) and len(after)==288
    assert len(set(after.values()))==288
    assert {p[:2] for p in after.values()}=={p[:2] for p in before.values()}
    # Each DSP24 pair shares a physical site; every DSP48 occupies a whole site.
    occupied=collections.defaultdict(list)
    for name,(x,y,sub) in after.items():occupied[x,y].append(('DSP48' if '/rq/LANE' in name else 'DSP24',sub))
    assert len(occupied)==160
    assert all(sorted(v)==[('DSP24',0),('DSP24',1)] or v==[('DSP48',0)] for v in occupied.values())
    file.write_text(''.join(f'{n} {x} {y} {sub}\n' for n,(x,y,sub) in sorted(after.items())))
    proof.update(layout='four_quadrants_with_central_rq',original_generator_sha256=proof['generator_sha256'],
        generator_sha256=sha(__file__),assignments_sha256=sha(file),groups=groups,
        fixed_dsp_sites_unchanged=True,ordinary_logic_unfixed=True,
        rationale='Each spatial row occupies one half-height region. Adjacent spatial rows share column choices for weight distribution; RQ is central among four MAC cascade columns.',
        route_or_timing_proven=False)
    (root/'quadrant_dsp_plan.json').write_text(json.dumps(proof,indent=2))
    return proof
