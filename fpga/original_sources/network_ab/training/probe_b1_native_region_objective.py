"""Controlled TRAIN fit: supervise the final native colour coordinates.

The previous count objective is attached before the last upsampling/rounding.
This probe adds native-resolution interval and within-labelled-region losses.
The soft decoder and region reductions exist only in the training objective;
all saved pictures are reloaded integer B1 outputs without an image filter.
"""
import argparse
import copy
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize
from . import train_b1_minimum_area as m
from .probe_b1_bounded_lbfgs import BoundedRows, BiasUnits
from .train_b1_count_regions import count_targets, count_loss
from .fit_b1_sixteen_regions import palette_key, region_class_map
from .minimum_area_blocks import components, boundary_mask
from .audit_b1_native_colour_regions import nearest_codes
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json, file_hash


def effective(net, key):
    scale=(getattr(net,key+'_m')/131072).float().flatten()
    w=(net.weight[key].detach()/scale[:,None,None,None]).round().clamp(-127,127)*scale[:,None,None,None]
    b=(net.bias[key].detach()/scale).round()*scale
    return w,b


def native_coordinates(net, logits, hard=False):
    """Continuous relaxation of the fixed existing two final decoder stages."""
    bits=(logits>=0).to(logits.dtype) if hard else logits.sigmoid()
    z=F.pad(bits*127,(0,0,0,0,0,1))
    w,b=effective(net,'up2_low_pw')
    z=F.conv2d(z,w,b).clamp(0,127)
    # This layer is exactly an integer count times its coordinate increment.
    # Rounding here is retained for the algebra check, relaxed for training.
    if hard:z=z.round()
    z=z.repeat_interleave(2,-2).repeat_interleave(2,-1)
    w,b=effective(net,'up2_high_dw')
    return F.conv2d(F.pad(z,(1,1,1,1),mode='replicate'),w,b,groups=8).clamp(0,127)[:,:3]


def region_variance(values, regions, valid):
    """Equal-region variance of colour coordinates, never across a boundary."""
    losses=[]
    for image, lab, mask in zip(values,regions,valid):
        ids=lab[mask];v=image[:,mask]
        if not len(ids):losses.append(image.sum()*0);continue
        _,ids=torch.unique(ids,return_inverse=True)
        n=int(ids.max())+1;count=torch.bincount(ids,minlength=n).to(v.dtype)
        index=ids[None].expand(3,-1)
        total=v.new_zeros(3,n).scatter_add(1,index,v)
        mean=total/count.clamp_min(1)[None]
        sq=(v-mean[:,ids]).square().mean(0)
        energy=sq.new_zeros(n).scatter_add(0,ids,sq)/count.clamp_min(1)
        eligible=count>=16
        losses.append(energy[eligible].mean() if eligible.any() else image.sum()*0)
    return torch.stack(losses).mean()


def native_region_loss(net, logits, wanted, regions, valid, margin=40):
    coords=native_coordinates(net,logits)
    if margin:
        coords=coords[...,margin:-margin,margin:-margin]
        wanted=wanted[...,margin:-margin,margin:-margin]
        regions=regions[...,margin:-margin,margin:-margin]
        valid=valid[...,margin:-margin,margin:-margin]
    # A correct output bin has a half-code radius. Keep a .2-code safety gap.
    interval=F.relu((coords-wanted).abs()-.3).square().mean()
    variance=region_variance(coords,regions,valid)
    return interval+variance,dict(native_interval=float(interval.detach()),native_region_variance=float(variance.detach()))


def regression(net):
    torch.manual_seed(914)
    z=torch.where(torch.rand(2,15,9,11)>.5,torch.tensor(30.),torch.tensor(-30.))
    bits=F.pad((z>=0).float()*127,(0,0,0,0,0,1))
    actual=fragment(net,bits,'up2_low_dw')['up2_high_dw'][:,:3]
    approx=native_coordinates(net,z,hard=True).round()
    error=float((actual-approx).abs().max());assert error==0
    labels=torch.zeros(1,16,16,dtype=torch.long);labels[:,:,8:]=1
    values=torch.zeros(1,3,16,16);values[:,:,:,8:]=3
    valid=torch.ones_like(labels,dtype=torch.bool)
    assert region_variance(values,labels,valid)==0
    values[:,:,2:4,2:4]=1;values.requires_grad_(True)
    loss=region_variance(values,labels,valid);loss.backward()
    assert loss>0 and (values.grad[:,:,2:4,2:4]>0).all()
    assert torch.count_nonzero(values.grad[:,:,:,8:])==0
    assert torch.allclose(loss,region_variance(values.detach(),labels*73+9,valid))
    # A smaller double precision differentiability check includes all reductions.
    test=torch.randn(1,3,4,8,dtype=torch.float64,requires_grad=True)
    lab=torch.zeros(1,4,8,dtype=torch.long);lab[:,:,4:]=1
    assert torch.autograd.gradcheck(lambda x:region_variance(x,lab,torch.ones_like(lab,dtype=torch.bool)),(test,))
    print('NATIVE_REGION_OBJECTIVE_REGRESSION_PASS',dict(hard_decoder_error=error,boundary_protected=True),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=150);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True)
    palette=torch.tensor(state['palette']);pk=palette_key(palette)
    xs=[];codes=[];labs=[];valid=[];ids=[]
    data=m.load_data()
    for suffix,(xx,yy) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix))
        assert row['split']=='train';ids.append(row['id'])
        image=m.t.r.read(row['path'])[yy:yy+320,xx:xx+320]
        code=region_class_map(row['target'],row['region'],pk)[yy:yy+320,xx:xx+320]
        xs.append(torch.tensor(image.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        codes.append(torch.tensor(code.astype('int64')));labs.append(torch.tensor(components(code).astype('int64')))
        valid.append(torch.tensor(cv2.dilate(boundary_mask(code),np.ones((3,3),np.uint8))==0))
    x=torch.stack(xs);codes=torch.stack(codes);labs=torch.stack(labs);valid=torch.stack(valid)
    targets=[v[...,20:-20,20:-20] for v in count_targets(codes)]
    wanted=torch.stack((codes//25,(codes//5)%5,codes%5),1).float()
    pal=(palette.numpy()*128+128).round().clip(0,255)
    results=[]
    for strength in (0.,1.):
        torch.manual_seed(712);sub=dest/('control' if strength==0 else 'native-region');sub.mkdir()
        net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
        net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
        regression(net)
        initial=net.export();train_keys=[]
        for key,w in list(net.weight.items()):
            train=key not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
            w.requires_grad_(train);net.bias[key].requires_grad_(train)
            if train:
                train_keys.append(key)
                limit=127*(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
                parametrize.register_parametrization(net.weight,key,BoundedRows(limit))
                parametrize.register_parametrization(net.bias,key,BiasUnits())
        assert initial==net.export()
        parameters=[p for p in net.parameters() if p.requires_grad]
        opt=torch.optim.LBFGS(parameters,lr=1,max_iter=3,max_eval=5,history_size=40,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-10)
        calls=0;history=[];began=time.perf_counter();details={}
        def closure():
            nonlocal calls,details
            net.quantized=False;opt.zero_grad(set_to_none=True)
            nodes=net(x,all_nodes=True);logits=(nodes['up1_high_dw'][:,:15]-63.5)/4
            ce=count_loss(logits[...,20:-20,20:-20],targets)
            if strength:
                extra,details=native_region_loss(net,logits,wanted,labs,valid)
                loss=ce+strength*extra
            else:loss=ce
            loss.backward();calls+=1
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            return loss
        @torch.no_grad()
        def report(step):
            net.quantized=True;nodes=net(x,all_nodes=True)
            counts=nodes['up2_high_dw'][:,:3,40:-40,40:-40]
            whole=(counts==wanted[...,40:-40,40:-40]).all(1)
            rec=dict(step=step,calls=calls,final720_exact_colour=float(whole.float().mean()),details=details)
            stats=[]
            for i in range(2):
                output=np.array(m.t.r.c.rgb_image(nodes['rgb_head'][i]/128))
                code,member=nearest_codes(output,pal);lab=components(code);size=np.bincount(lab.ravel())
                stats.append(dict(small_area_fraction=float(size[size<256].sum()/size.sum()),regions=len(size),**member))
                if step in (0,a.steps):
                    m.t.r.c.rgb_image(nodes['rgb_head'][i]/128).save(sub/f'integer_{step:04d}_{i}.png')
            rec['native_area']=stats;history.append(rec)
            print('NATIVE_REGION_MATCHED_PROBE',strength,rec,flush=True)
            write_json(sub/'progress.json',dict(history=history,seconds=time.perf_counter()-began,training_only=True))
        report(0)
        for step in range(1,a.steps+1):
            opt.step(closure)
            if step%25==0 or step==a.steps:report(step)
        for key in train_keys:
            parametrize.remove_parametrizations(net.weight,key,leave_parametrized=True)
            parametrize.remove_parametrizations(net.bias,key,leave_parametrized=True)
        net.quantized=True;blob=net.export();(sub/'coefficients-padded.bin').write_bytes(blob)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu')
        with torch.no_grad():
            theirs,_=actual(x*128,True);ours=net(x,all_nodes=True)
            error=max(float((v-theirs[k]).abs().max()) for k,v in ours.items());assert error==0
        result=dict(state);result.update(model=copy.deepcopy(net.state_dict()),qat=False,step=a.steps,
            native_region_objective_weight=strength,training_only=True,visual_acceptance=False)
        torch.save(result,sub/'candidate.pt')
        results.append(dict(weight=strength,history=history,calls=calls,seconds=time.perf_counter()-began,all_node_integer_error=error))
    write_json(dest/'result.json',dict(warm=a.warm,warm_sha256=file_hash(a.warm),train_ids=ids,results=results,
        code_sha256=file_hash(__file__),added_inference_ops=0,visual_acceptance=False))


if __name__=='__main__':main()
