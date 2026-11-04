"""Verify the native placement honored every complete DSP cascade assignment."""
import argparse,json,re
from pathlib import Path
from run_units import sha

def audit(root):
    meta=json.loads((root/'physical_inputs.json').read_text());plan=json.loads((root/'quadrant_dsp_plan.json').read_text())
    assert plan['layout']==meta['dsp_layout']['layout']=='four_quadrants_with_central_rq'
    assert sha(root/'outflow/ti60f225_oob.map.v')==meta['base_map_sha256']
    prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
    expected={}
    for g in plan['groups']:
        row=g['row'];ys=list(range(g['y_min'],g['y_max']+1,20));assert len(ys)==8
        for pair,x in enumerate(g['mac_columns']):
            for tap,y in enumerate(ys):
                for sub in (0,1):expected[prefix+f'mac/arithmetic/ROW[{row}].CHANNEL[{pair*2+sub}].TAP[{tap}].mac']=[x,y,sub]
        for channel,y in enumerate(ys):expected[prefix+f'rq/LANE[{row*8+channel}].rq/multiply_add']=[g['rq_column'],y,0]
    net=(root/'outflow/ti60f225_oob.map.v').read_text()
    dsp=set(re.findall(r'EFX_DSP(?:24|48)\s+\\([^\s]+)',net));assert dsp==set(expected) and len(dsp)==288
    for filename in ('fixed_dsp.placeloc','outflow/ti60f225_oob.out.placeloc'):
        actual={w[0]:list(map(int,w[1:])) for line in (root/filename).read_text().splitlines() if len(w:=line.split())==4 and w[0] in dsp}
        assert actual==expected,filename
    assert len({tuple(v) for v in expected.values()})==288 and len({tuple(v[:2]) for v in expected.values()})==160
    return dict(status='NATIVE_288_DSP_PRIMITIVES_160_SITES_MATCH_GROUP_PLAN',primitive_count=288,physical_dsp=160,
        map_sha256=meta['base_map_sha256'],physical_inputs_sha256=sha(root/'physical_inputs.json'),
        plan_sha256=sha(root/'quadrant_dsp_plan.json'),placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),
        assignment_sha256=sha(root/'fixed_dsp.placeloc'),hardware_access=False,timing_pass=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    out=root/'grouped_dsp_actual_locations.json';assert not out.exists()
    result=audit(root);out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
