"""Controlled source-to-SDF fit, comparing value and finite-difference supervision.

The target derivative is measured from the exact sampled target; it is not a
global unit-gradient assumption. Same data, initialization and optimizer in
both arms. No geometric operation is added to B1 inference.
"""
import argparse,copy,sys,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_reference_boundary_profile import reference_boundary_profile
from .audit_b1_reachable_loss_fixedpoint import objective
from .train_b1_quantum_regions import RowScaledAdam
from .centered_b1_optimizer import CenteredB1Adam,calibration_means,regression as centering_regression
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def derivative_loss(field,truth):
    p=field[:,:30]/127;t=truth/127;terms=[]
    for axis in (-1,-2):
        dp=torch.diff(p,dim=axis);dt=torch.diff(t,dim=axis)
        # Target derivative includes saturation, sampling, corners and thin
        # regions. Penalize excess changes even where the target is constant.
        e=F.smooth_l1_loss(dp,dt,beta=.04,reduction='none')
        edge=dt.abs()>.01
        terms.append(e.mean()+(e*edge).sum()/edge.sum().clamp_min(1))
    return sum(terms)/2


def regression():
    t=(63.5+torch.arange(-8,8)*16).clamp(0,127)[None,None,None].expand(1,30,16,16).clone()
    assert derivative_loss(t,t)==0
    for p in (63.5+(t-63.5)*.5,(63.5+(t-63.5)*2).clamp(0,127)):
        p=p.detach().requires_grad_();v=derivative_loss(p,t);assert v>0;v.backward();assert p.grad.norm()>0 and torch.isfinite(p.grad).all()
    flat=torch.zeros_like(t);assert derivative_loss(flat,flat)==0
    noise=flat.clone();noise[...,::2,::2]=32;assert derivative_loss(noise,flat)>0
    print('DISTANCE_GEOMETRY_PASS target0/too_soft/too_steep/constant0/oscillation',flush=True)


def optimizer(net,unfreeze_stride=False,centred=False,means=None):
    groups=[]
    for k,w in net.weight.items():
        fixed=k.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) or (not unfreeze_stride and k in ('down1__dw','down2__dw'))
        w.requires_grad_(not fixed);net.bias[k].requires_grad_(not fixed)
        if not fixed:
            scale=(getattr(net,k+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=.00015,row_gain=(scale*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.008)])
    return CenteredB1Adam(groups,net,means) if centred else RowScaledAdam(groups)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--device',default='cuda');ap.add_argument('--stride-compare',action='store_true');ap.add_argument('--centred-compare',action='store_true');a=ap.parse_args()
    assert not (a.stride_compare and a.centred_compare)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261007)
    regression();state,initial=load(a.checkpoint);initial.__class__=OrdinalBandB1;initial.to(a.device)
    dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data(a.data);videos=sorted({r['video'] for r in data['training']});rows=[]
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).tolist()))
    xs=[];labs=[];codes=[]
    for video in (videos[0],videos[len(videos)//2],videos[-1]):
        rr=[r for r in data['training'] if r['video']==video];row=rr[len(rr)//2];rows.append(row);sl=np.s_[232:488,512:768]
        xs.append(torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy()).float()/128-1)
        labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        codes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
    x=torch.stack(xs).to(a.device);labels=torch.stack(labs).to(a.device);truth=distance_fields(torch.stack(codes).to(a.device),8)
    means=calibration_means(initial,x)
    if a.centred_compare:centering_regression()
    with torch.no_grad():target=fragment(initial,F.pad(initial.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
    cut=lambda z:z[...,40:-40,40:-40]
    sources={}
    for module in list(sys.modules.values()):
        src=getattr(module,'__file__',None)
        if src and 'network_ab' in src and src.endswith('.py'):
            src=Path(src).resolve();sources[str(src)]=file_hash(src)
    write_json(dest/'config.json',dict(args=vars(a),sources=sources,train_ids=[r['id'] for r in rows],crop=[512,232,768,488],no_augmentation=True,no_temporal=True,local_diagnostic_only=True))
    shutil.copyfile(__file__,dest/Path(__file__).name)
    records=[];images=[target.detach().cpu()]
    modes=('value','centered_updates') if a.centred_compare else (('value','trainable_stride') if a.stride_compare else ('value','value_and_derivative'))
    for mode in modes:
        net=copy.deepcopy(initial);opt=optimizer(net,mode=='trainable_stride',mode=='centered_updates',means);parameters=[p for p in net.parameters() if p.requires_grad];history=[]
        def save(step):
            s=copy.deepcopy(state);s.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,local_geometry_fit=mode,local_training_ids=[r['id'] for r in rows],added_inference_operations=0,visual_acceptance=False)
            torch.save(s,dest/f'{mode}_{step:06d}.pt')
        for step in range(a.steps+1):
            nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128;field=nodes['up1_low_dw'][...,10:-10,10:-10];t=truth[...,10:-10,10:-10]
            full,band=distance_objective(field,t);shape=derivative_loss(field,t)
            loss=16*full+8*band+(8*shape if mode=='value_and_derivative' else 0)
            if step%100==0 or step==a.steps:
                with torch.no_grad():rgb,v=objective(cut(p),cut(target),cut(labels),reference_boundary_profile)
                row=dict(step=step,loss=float(loss.detach()),full=float(full.detach()),band=float(band.detach()),derivative=float(shape.detach()),rgb=float(rgb),mae=float((cut(p)-cut(target)).abs().mean().detach()*128),profile_shape=float(v['rgb_profile_shape']),profile_contrast=float(v['rgb_profile_contrast']))
                history.append(row);print('DISTANCE_GEOMETRY_FIT',mode,row,flush=True);write_json(dest/f'{mode}_progress.json',history)
            if step%500==0:save(step)
            if step==a.steps:break
            opt.zero_grad();loss.backward()
            if isinstance(opt,CenteredB1Adam):opt.prepare_gradients()
            torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for k,w in net.weight.items():
                    if w.requires_grad:
                        lim=127*(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];w.copy_(w.maximum(-lim).minimum(lim))
            if step==0:
                # Actual post-update round trip in a fresh model and optimizer.
                save(1);ss=torch.load(dest/f'{mode}_000001.pt',map_location=a.device,weights_only=True);fresh=copy.deepcopy(initial);fresh.load_state_dict(ss['model']);fresh.package['residual_encoding']=copy.deepcopy(ss['residual_encoding']);fo=optimizer(fresh,mode=='trainable_stride',mode=='centered_updates',means);fo.load_state_dict(ss['optimizer'])
                with torch.no_grad():assert torch.equal(fresh(x)[0],net(x)[0])
                # Adam's scalar step counter may reside on CPU after load even
                # when moment tensors are on CUDA. Compare values explicitly.
                assert all(torch.equal(v.cpu(),fo.state_dict()['state'][k][n].cpu()) for k,s in opt.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v))
                print('DISTANCE_GEOMETRY_GPU_RESUME_PASS',mode,flush=True)
        save(a.steps)
        with torch.no_grad():
            final=net(x,all_nodes=True);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=a.device);nn,_=actual(net.round(x*128),True)
            assert all(torch.equal(v,nn[k]) for k,v in final.items()) and len(raw)==20480
            (dest/f'{mode}.bin').write_bytes(raw)
        images.append((final['rgb_head']/128).cpu());records.append(dict(mode=mode,history=history,all_node_error=0,bytes=20480))
        board=Image.new('RGB',(256*len(images),870),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
        for i in range(3):
            titles=('已知目标','原优化器','中心化参数更新') if a.centred_compare else (('已知目标','固定采样系数','可训练采样系数') if a.stride_compare else ('已知目标','只拟合距离','距离及局部变化'))
            for j,(name,p) in enumerate(zip(titles,images)):
                board.paste(m.t.r.c.rgb_image(p[i]),(j*256,i*290+34));draw.text((j*256+2,i*290+4),name,font=font,fill='white')
        board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,train_ids=[r['id'] for r in rows],steps=a.steps,local_fit_not_generalization=True,checkpoint_sha256=file_hash(a.checkpoint),visual_acceptance=False))
    assert all(file_hash(path)==sha for path,sha in sources.items())
    print('DISTANCE_GEOMETRY_COMPLETE',flush=True)


if __name__=='__main__':main()
