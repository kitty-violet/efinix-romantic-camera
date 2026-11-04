"""Reuse the checked six-bank compiler with an explicit larger graph binding."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from . import compile_component_row_pair as compiler
from .b1_capacity_component_net import plan,budget
from .capacity_fixed_headers import coefficient_image
from .train_luma_regions import ROOT,RUN,LEVELS
from .runtime import file_hash,write_json


def fused_plan():
    result=[]
    for row in plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=result.pop();assert p['id']==n['inputs'][0]
            n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        result.append(n)
    result[-1]['id']='rgb_head';return dict(nodes=result)


def diagnostic(root):
    root.mkdir(exist_ok=False);rng=np.random.default_rng(2026100456);parameters={};residual={};scales={'input':1/128}
    for n in plan()['nodes']:
        scales[n['id']]=1/128
        if n['kernel']:
            shape=(n['channels'],1 if n['kind']=='dw3x3' else n['input_channels'],n['kernel'],n['kernel'])
            parameters[n['id']]=dict(weight_oihw=rng.integers(-2,3,shape).tolist(),bias_int32=[0]*n['channels'],multiplier_q17=[65536]*n['channels'],input_scale=[1,128],output_scale=[1,128])
        elif n['kind']=='residual_add':residual[n['id']]=dict(main_scaled=True,multiplier=131071,post_multiplier=131071,parameters=dict(main_scale=[1,128],skip_scale=[1,128],common_scale=[1,128],output_scale=[1,128]))
    package=dict(schema='b1-capacity-component-int8-q17-v1',plan=plan(),parameters=parameters,residual=residual,scales=scales,levels=LEVELS,synthetic=True,trained=False)
    write_json(root/'runtime_parameters.json',package);coefficient_image(package,root);return root/'runtime_parameters.json'


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--package',type=Path);p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=48);a=p.parse_args()
    sys.path.insert(0,str(compiler.HW));base=ROOT/'runs'/RUN
    if a.probe:
        out=base/a.name;diagnostic(out);trials=[]
        for width in [64,56,48,40]:
            row=dict(width=width,positions=[])
            try:
                for i,x in enumerate([0,width,1280-width]):
                    _,r=compiler.compile_one(fused_plan(),i,x,width);row['positions'].append(r);print('CAPACITY_ALLOCATION',width,x,r['high'],flush=True)
                row['status']='THREE_POSITIONS_PASS_NOT_FULL_FRAME'
            except (AssertionError,MemoryError) as e:row.update(status='FAIL',error=str(e));print('CAPACITY_ALLOCATION_FAIL',width,str(e)[:350],flush=True)
            trials.append(row);write_json(out/'allocation_probe.json',dict(budget=budget(),trials=trials,rtl=False,fit=False,full_frame=False))
            if row['status'].startswith('THREE'):break
        return
    assert a.package is not None
    package=json.loads(a.package.read_text());assert package['plan']==plan()
    compiler.fused_plan=fused_plan;compiler.coefficient_image=coefficient_image
    sys.argv=[sys.argv[0],'--package',str(a.package),'--name',a.name,'--width',str(a.width)];compiler.main()
    out=base/a.name;m=json.loads((out/'program.json').read_text());sources=[Path(__file__),Path(__file__).with_name('b1_capacity_component_net.py'),Path(__file__).with_name('capacity_fixed_headers.py')]
    for f in sources:(out/'frozen'/f.name).write_bytes(f.read_bytes());m['source_hashes'][str(f)]=file_hash(f)
    m.update(capacity_variant=True,budget=budget());write_json(out/'program.json',m)


if __name__=='__main__':main()
