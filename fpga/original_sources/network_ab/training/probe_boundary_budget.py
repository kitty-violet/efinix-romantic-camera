"""Bounded boundary-path feasibility/fit diagnostic, not a deployment network.

The production contract is only read. R3 is replaced by a learned stem->PW8
feature path before RGB. Existing operator types, but a DIFFERENT graph/ABI.
No claim that a lower ideal MAC count implies bank, DDR, timing or style success.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F

from network_ab.model_contract import compile_contract, load_contract
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace, file_hash, write_json, contract_guard
from .engine import set_reproducible
from .train_b1_distillation import as_tensor, to_rgb, read_image, edges
from .repair_style_retention import gray

RUN = 'boundary-budget-diagnostic-20261004'
ROOT = Path('E:/fpga')
HW = ROOT/'hw/efinity/prototypes/frame_execution'
INITIAL = ROOT/'model-training-workspace/runs/b1-distillation-20261003/final-2000/b1_style_fp32.pt'


def research_plan():
    plan = copy.deepcopy(compile_contract(load_contract()))
    nodes = [n for n in plan['nodes'] if not n['id'].startswith('res3_')]
    for n in nodes:
        n['inputs'] = ['res2_add' if p == 'res3_add' else p for p in n['inputs']]
    extra = copy.deepcopy(next(n for n in nodes if n['id'] == 'up2_low_pw'))
    extra.update(id='boundary.pw', inputs=['stem'], input_channels=8,
                 input_shape_hwc=[720,1280,8], output_shape_hwc=[720,1280,8],
                 activation='none', terms=8, raw_weights=64, macs=720*1280*64)
    add = copy.deepcopy(next(n for n in nodes if n['id'] == 'res2_add'))
    add.update(id='boundary_add', inputs=['boundary.pw','up2_high_dw'], channels=8,
               input_channels=8, input_shape_hwc=[720,1280,8], output_shape_hwc=[720,1280,8])
    nodes[-1]['inputs'] = ['boundary_add']
    nodes[-1:-1] = [extra,add]
    return dict(nodes=nodes, production=False,
                macs=sum(n['macs'] for n in nodes),
                note='Research only. Existing operations, changed graph and producer ABI.')


class BoundaryNet(nn.Module):
    def __init__(self, initial=INITIAL):
        super().__init__()
        old=build_fp32_model()
        old.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['model'])
        self.operators=old.operators
        for key in list(self.operators):
            if key.startswith('res3_'):del self.operators[key]
        self.operators['boundary__pw']=nn.Sequential(nn.Conv2d(8,8,1,bias=False),nn.BatchNorm2d(8))
        nn.init.zeros_(self.operators['boundary__pw'][0].weight)
        self.nodes=research_plan()['nodes']
        self.eval()

    def forward(self,x):
        values={'input':x}
        for n in self.nodes:
            a=values[n['inputs'][0]]
            if n['kind']=='residual_add':v=a+values[n['inputs'][1]]
            elif n['kind']=='nearest_x2':v=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                v=self.operators[n['id'].replace('.','__')](a)
                if n['activation']=='clamp_rgb':v=v.clamp(-1,127/128)
            values[n['id']]=v
        return values['rgb_head']


def fused_plan():
    original=research_plan();out=[]
    for n in original['nodes']:
        n=copy.deepcopy(n)
        if n['kind']=='residual_add':
            pw=out.pop();assert pw['id']==n['inputs'][0] and pw['kind']=='pw1x1'
            n.update(kind='pw_fused',parameter_node=pw['id'],inputs=[pw['inputs'][0],n['inputs'][1]],
                     input_channels=pw['input_channels'],input_shape_hwc=pw['input_shape_hwc'],terms=pw['terms'],macs=pw['macs'])
        out.append(n)
    return dict(nodes=out)


def tile(plan,index,x,width):
    byid={n['id']:n for n in plan['nodes']}
    ranges={'rgb_head':(x,min(x+width,1280)-1)}
    for n in reversed(plan['nodes']):
        a,b=ranges[n['id']]
        if n['kind']=='nearest_x2':a,b=a//2,b//2
        elif n['kind'] in ('conv3x3','dw3x3'):a,b=a*n['stride']-1,b*n['stride']+1
        for p in n['inputs']:
            w=byid[p]['output_shape_hwc'][1] if p in byid else 1280
            lo,hi=max(0,a),min(w-1,b)
            if p in ranges:lo,hi=min(lo,ranges[p][0]),max(hi,ranges[p][1])
            ranges[p]=(lo,hi)
    a,b=ranges['input'];a=a//4*4
    return dict(tile_index=index,commit_interval=dict(x0=x,x1=min(x+width,1280)-1),
        source_interval=dict(x0=a,x1=b),steps=[dict(operation=n['id'],
        output_interval=dict(x0=ranges[n['id']][0],x1=ranges[n['id']][1])) for n in plan['nodes']])


def allocate(run):
    sys.path.insert(0,str(HW))
    from fourway_frame_program import compile_tile
    sources=[HW/'fourway_frame_program.py',HW/'compile_frame.py',Path(__file__),ROOT/'network_ab/model_b_contract.json']
    hashes={str(p):file_hash(p) for p in sources}
    plan=fused_plan();attempts=[];chosen=None
    # Three real full-height positions before any expensive all-stripe program.
    for width in (256,128,64,32,16):
        record=dict(width=width,positions=[])
        for j,x in enumerate((0,512,1280-width)):
            try:
                p=compile_tile(plan,dict(name='research_boundary'),tile(plan,j,x,width),0,4,
                    aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
                rec=dict(x=x,high=p['finite_high_water'],peak_bytes=p['peak_live_padded_bytes'],commands=dict(Counter(c['kind'] for c in p['commands'])))
                record['positions'].append(rec)
            except (MemoryError,AssertionError) as e:
                record['rejected']=dict(x=x,exception=type(e).__name__,message=str(e));break
        attempts.append(record);print('BOUNDARY_ALLOC',json.dumps(record),flush=True)
        if 'rejected' not in record:chosen=width;break
    assert all(file_hash(p)==h for p,h in hashes.items())
    result=dict(status='THREE_STRIPES_FIT_ONLY_NOT_FULL_SCHEDULE' if chosen else 'REJECTED_BY_CURRENT_FINITE_ALLOCATOR',
        width=chosen,attempts=attempts,source_hashes=hashes,ideal_macs=research_plan()['macs'],
        production_abi_compatible=False,rtl_run=False,physical_resources_verified=False,timing_verified=False)
    write_json(run/'allocation.json',result)
    return result


def objective(y,t):
    # Diagnostic overfit: directly match an aligned teacher. No TV, blur,
    # phase or time regularizer that can reward smoothing in this experiment.
    y,t=y[:,:,32:-32,32:-32],t[:,:,32:-32,32:-32]
    return F.l1_loss(y,t)+2*sum(F.l1_loss(a,b) for a,b in zip(edges(gray(y)),edges(gray(t))))


def checks(run):
    set_reproducible(2026100411)
    net=BoundaryNet().cuda().eval();opt=torch.optim.Adam(net.parameters(),lr=.0005)
    calls=[]
    hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in net.modules() if isinstance(m,nn.Conv2d)]
    x=torch.zeros(1,3,720,1280,device='cuda');target=torch.ones_like(x)*.1
    loss=objective(net(x),target);loss.backward();opt.step()
    for h in hooks:h.remove()
    assert sum(calls)==research_plan()['macs']==874598400
    x=torch.randn(2,3,256,256,device='cuda')*.2;target=x*.7
    opt.zero_grad(set_to_none=True);loss=objective(net(x),target);loss.backward();opt.step()
    path=run/'preflight.pt';torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),path)
    def update(n,o):
        o.zero_grad(set_to_none=True);l=objective(n(x),target);l.backward();o.step()
        return {k:v.detach().cpu().clone() for k,v in n.state_dict().items()}
    expected=update(net,opt);restored=BoundaryNet().cuda().eval();o=torch.optim.Adam(restored.parameters(),lr=.0005)
    s=torch.load(path,map_location='cpu',weights_only=True);restored.load_state_dict(s['model']);o.load_state_dict(s['optimizer']);actual=update(restored,o)
    err=max(float((expected[k]-v).abs().max()) for k,v in actual.items() if v.is_floating_point())
    assert err<2e-6
    write_json(run/'preflight.json',dict(status='PASS',macs=sum(calls),convolutions=len(calls),
        parameters=sum(p.numel() for p in net.parameters()),fp32=True,full720_backward=True,batch2=True,restore_max_error=err))
    print('BOUNDARY_PREFLIGHT_PASS',err,flush=True)


def fit(run,steps):
    assert json.loads((run/'preflight.json').read_text())['status']=='PASS'
    manifest=json.loads((ROOT/'model-training-workspace/runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))
    # Training source only; never optimize the familiar stairs/plaza regressions.
    row=manifest['pairs'][0]
    source,target=read_image(row['current']),read_image(row['current_target'])
    # Fixed native crop for an explicitly non-generalizing capacity diagnosis.
    xy=(480,240);x,y=xy;s=source[y:y+320,x:x+320];t=target[y:y+320,x:x+320]
    Image.fromarray(s).save(run/'train_crop_source.png');Image.fromarray(t).save(run/'train_crop_teacher.png')
    write_json(run/'fit_scope.json',dict(source=row,xy=xy,wh=[320,320],steps=steps,
        warning='ONE TRAINING CROP OVERFIT DIAGNOSTIC. Not generalization, temporal stability or style acceptance.'))
    a,b=as_tensor(s)[None].cuda(),as_tensor(t)[None].cuda();outputs={};scores={}
    for kind in ('exact_b1','boundary_r2'):
        set_reproducible(2026100411)
        net=BoundaryNet() if kind=='boundary_r2' else build_fp32_model()
        if kind=='exact_b1':net.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True)['model'])
        net=net.cuda().eval();opt=torch.optim.Adam(net.parameters(),lr=.001)
        with torch.no_grad():initial=float(objective(net(a),b))
        logs=[];start=time.perf_counter()
        for step in range(1,steps+1):
            opt.zero_grad(set_to_none=True);loss=objective(net(a),b)
            assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
            logs.append(float(loss.detach()))
            if step%100==0:print('BOUNDARY_FIT',kind,step,logs[-1],flush=True)
        with torch.no_grad():p=net(a);outputs[kind]=to_rgb(p[0]);final=float(objective(p,b))
        Image.fromarray(outputs[kind]).save(run/(kind+'_fitted.png'))
        torch.save(dict(model=net.state_dict(),steps=steps,research_only=True,training_crop_only=True),run/(kind+'_diagnostic.pt'))
        scores[kind]=dict(initial=initial,final=final,seconds=time.perf_counter()-start,loss=logs)
        # Held-out images are for observation only, and are not called blind.
        folder=run/kind;folder.mkdir()
        with torch.no_grad():
            for r in manifest['validation']:
                v=to_rgb(net(as_tensor(read_image(r['source']))[None].cuda())[0]);Image.fromarray(v).save(folder/(r['id']+'.png'))
        del net,opt
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    canvas=Image.new('RGB',(1280,354),'#16202a');d=ImageDraw.Draw(canvas)
    for i,(label,img) in enumerate([('训练裁切原图',s),('教师目标',t),('原B1：单块拟合',outputs['exact_b1']),('边界通路：单块拟合',outputs['boundary_r2'])]):
        d.text((320*i+5,5),label,font=font,fill='white');canvas.paste(Image.fromarray(img),(320*i,34))
    canvas.save(run/'fit_comparison.png');write_json(run/'fit_scores.json',scores)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','checks','fit']);ap.add_argument('--steps',type=int,default=300);a=ap.parse_args()
    root=configure_workspace();contract_guard();run=root/'runs'/RUN
    if a.action=='prepare':
        run.mkdir(exist_ok=False);write_json(run/'research_graph.json',research_plan());allocate(run)
    elif a.action=='checks':checks(run)
    else:fit(run,a.steps)


if __name__=='__main__':main()
