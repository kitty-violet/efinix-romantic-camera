"""Controlled TRAIN-fit test: allow clipped features to move inward only.

Forward values/export are unchanged. This is a deliberately biased backward
estimator, not the mathematical derivative of hard clipping. Never use its
fit score as validation or a quality claim.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .train_b1_ordered_regions import OrderedGradientB1
from .train_b1_count_regions import count_targets,count_loss
from .train_b1_quantum_regions import RowScaledAdam
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .runtime import write_json,file_hash


class InwardClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,raw,low,high):
        ctx.save_for_backward(raw);ctx.low=low;ctx.high=high
        return value

    @staticmethod
    def backward(ctx,gradient):
        raw,=ctx.saved_tensors
        active=(raw>=ctx.low)&(raw<=ctx.high)
        inward=((raw<ctx.low)&(gradient<0))|((raw>ctx.high)&(gradient>0))
        return None,gradient*(active|inward),None,None


class InwardGradientB1(OrderedGradientB1):
    def training_activation_proxy(self,name,value,raw,row):
        # Audited residual feature branches have the highest blocked mass.
        if name not in ('res1_conv1.pw','res2_conv1.pw','res3_conv1.pw'):return value
        return InwardClip.apply(value,raw,0 if row['activation']=='relu' else -128,127)


def regression():
    raw=torch.tensor([-3.,-3.,3.,130.,130.],requires_grad=True)
    value=raw.clamp(0,127);actual=InwardClip.apply(value,raw,0,127)
    assert torch.equal(actual,value)
    (actual*torch.tensor([-1.,1.,2.,1.,-1.])).sum().backward()
    assert torch.equal(raw.grad,torch.tensor([-1.,0.,2.,1.,0.]))
    print('INWARD_CLIP_DIRECTION_REGRESSION_PASS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=300)
    ap.add_argument('--variants',nargs='+',choices=['ordinary','inward','region','kernel','margin'],default=['ordinary','inward']);a=ap.parse_args()
    regression();torch.set_num_threads(2);torch.manual_seed(712)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);state=torch.load(a.warm,map_location='cpu',weights_only=True)
    data=m.load_data();palette=torch.tensor(state['palette']);pk=palette_key(palette);xs=[];codes=[];ids=[];regions=[]
    for suffix,(xx,yy) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix));assert row['split']=='train';ids.append(row['id'])
        source=m.t.r.read(row['path'])[yy:yy+320,xx:xx+320]
        xs.append(torch.tensor(source.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        codes.append(torch.tensor(region_class_map(row['target'],row['region'],pk)[yy:yy+320,xx:xx+320].astype('int64')))
        regions.append(torch.tensor(m.load_region(row['region'])[yy:yy+320,xx:xx+320].astype('int64')))
    x=torch.stack(xs);truth=[v[...,20:-20,20:-20] for v in count_targets(torch.stack(codes))]
    wanted=truth[0].argmax(1)*25+truth[1].argmax(1)*5+truth[2].argmax(1)
    pure=torch.stack([v.amax(1)>.999 for v in truth]).all(0)
    records=[];initial_output=None;started=time.perf_counter()
    for variant in a.variants:
        cls=InwardGradientB1 if variant=='inward' else OrderedGradientB1
        net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin');net.__class__=cls
        net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=state['residual_encoding']
        groups=[];fixed=False
        for key,w in net.weight.items():
            if key=='up1_high_dw':fixed=True
            w.requires_grad_(not fixed);net.bias[key].requires_grad_(not fixed)
            if not fixed:
                scale=(getattr(net,key+'_m')/131072).float().flatten()
                groups.append(dict(params=[w],lr=.001,row_gain=(scale/(1/256))[:,None,None,None]))
                groups.append(dict(params=[net.bias[key]],lr=.004))
        opt=RowScaledAdam(groups)
        if variant=='kernel':
            from .b1_boundary_kernel_learning import enable_kernel,project_kernel
            enable_kernel(net,opt,.001)
        for step in range(a.steps+1):
            nodes=net(x,all_nodes=True);logits=(nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
            loss=count_loss(logits,truth)
            if variant=='margin':
                from .probe_b1_hard_count_margin import hard_count_margin
                loss=.5*loss+hard_count_margin(logits,truth)
            if variant=='region':
                from .train_b1_region_tail_counts import region_tail_loss
                tail,_=region_tail_loss(logits,truth,torch.stack(regions),False)
                loss=loss+.35*tail
            if step%50==0 or step==a.steps:
                bits=nodes['up2_low_dw'][:,:15,20:-20,20:-20]>0
                actual=bits[:,:7].sum(1)*25+bits[:,7:11].sum(1)*5+bits[:,11:].sum(1)
                rec=dict(variant=variant,step=step,loss=float(loss.detach()),accuracy_pure=float((actual[pure]==wanted[pure]).float().mean()))
                records.append(rec);print('INWARD_FIT',rec,flush=True)
            if step==0:
                output=nodes['rgb_head'].detach().clone()
                if initial_output is None:initial_output=output
                else:assert torch.equal(initial_output,output)
            if step==a.steps:
                for i in range(len(x)):m.t.r.c.rgb_image(nodes['rgb_head'][i]/128).save(dest/f'{variant}_{i}.png')
                break
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for key,w in net.weight.items():
                    if w.requires_grad:
                        limit=127*(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
                        w.copy_(w.maximum(-limit).minimum(limit))
            if variant=='kernel':project_kernel(net)
        del nodes,net,opt
    write_json(dest/'result.json',dict(warm=a.warm,warm_sha256=file_hash(a.warm),train_ids=ids,records=records,
        exact_same_initial_forward=True,steps_per_arm=a.steps,seconds=time.perf_counter()-started,
        device='cpu',modified_inference=False,validation_fitted=False,visual_acceptance=False))


if __name__=='__main__':main()
