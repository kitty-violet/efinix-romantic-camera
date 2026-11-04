"""Actual trained RGB weight calibration and independent integer verification."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .detail_rgb_net import DetailRGBNet
from .region_detail_target import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .probe_region_class_int8 import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--luma',action='store_true');p.add_argument('--context',action='store_true');p.add_argument('--native',action='store_true');p.add_argument('--coarse',action='store_true');a=p.parse_args()
    if a.native or a.coarse:a.context=True
    if a.context:a.luma=True
    configure_workspace();contract_guard();set_reproducible(2026100422)
    cls=DetailRGBNet
    if a.luma:
        from .train_luma_regions import LumaNet,display
        cls=LumaNet
    if a.context:
        from .equalize_context import EqualizedContext
        cls=EqualizedContext
    if a.native:
        from .equalize_native import EqualizedNative
        cls=EqualizedNative
    if a.coarse:
        from .equalize_coarse import EqualizedCoarse
        cls=EqualizedCoarse
    root=ROOT/'runs'/RUN;folder=root/('luma' if a.luma else 'rgb');out=root/a.name;assert out.parent==root;out.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in state['binding']['code'].items());net=cls().cuda().eval();net.load_state_dict(state['model'])
    data=json.loads((folder/'data.json').read_text(encoding='utf-8'));groups={}
    for r in data['pairs']:groups.setdefault(r['source'],[]).append(r)
    paths=[]
    for rows in groups.values():paths.extend(rows[i]['current'] for i in sorted({0,len(rows)//2,len(rows)-1}))
    if state['scales'] is None:calibration=net.calibrate(as_tensor(read_image(p))[None].cuda() for p in paths)
    else:net.scales=state['scales'];calibration=state['calibration']
    package=net.integer_package();package.update(checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),trained_steps=state['step'],training_mode=state['mode'])
    write_json(out/'runtime_parameters.json',package);write_json(out/'calibration.json',calibration)
    rng=np.random.default_rng(2026100422);arrays=[rng.integers(-128,128,(1,3,8,16 if a.coarse else 12),dtype=np.int64),np.full((1,3,8,8),-128,np.int64),
        np.full((1,3,8,8),127,np.int64),(read_image(paths[0])[:16,:16].astype(np.int64)-128).transpose(2,0,1)[None]]
    tests=[]
    with torch.no_grad():
        for i,x in enumerate(arrays):
            cpu=execute(package,x,False);gpu=execute(package,torch.tensor(x,device='cuda',dtype=torch.float32),True)
            learned=net(torch.tensor(x,device='cuda',dtype=torch.float32)/128,True);errors={};fake={}
            for n,y in gpu.items():
                errors[n]=int(np.abs(y.cpu().numpy().astype(np.int64)-cpu[n]).max());assert errors[n]==0,(i,n,errors[n])
                aq=(learned[n]/net.scales[n]).round().cpu().numpy().astype(np.int64);fake[n]=int(np.abs(aq-cpu[n]).max());assert fake[n]==0,(i,n,fake[n])
            tests.append(dict(index=i,shape=list(x.shape),cpu_gpu=errors,qat_integer=fake))
        rows=[]
        fp=cls().cuda().eval();fp.load_state_dict(state['model'])
        for r in data['validation']:
            arr=read_image(r['source']);x=as_tensor(arr)[None].cuda();qnet=net(x)
            q=execute(package,torch.tensor(arr.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))['rgb']
            diff=float((qnet/net.scales['rgb']-q).abs().max());assert diff<.0001,(r['id'],diff)
            image=to_rgb(display(q)[0]) if a.luma else (q[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
            Image.fromarray(image).save(out/(r['id']+'.png'))
            f=to_rgb(display(fp(x))[0] if a.luma else fp(x)[0]);rows.append(dict(id=r['id'],qat_integer_max=diff,fp32_integer_mae=float(np.abs(f.astype(float)-image).mean())))
    write_json(out/'verification.json',dict(status='TRAINED_INTEGER_NUMERIC_PASS_NOT_RTL_STYLE',tests=tests,records=rows,
        runtime_sha256=file_hash(out/'runtime_parameters.json'),qat=state['mode']=='qat',calibration_frames=len(paths)))
    print('DETAIL_INTEGER_PASS',json.dumps(rows),flush=True)

if __name__=='__main__':main()
