"""Actual source-pair gradient audit of geometry vs stability objectives.

All terms use the same real TRAIN pair/crop and current B1, with the existing
flow target alignment. Compare in the saved optimizer's whitened coordinates,
before norm clipping/Adam; these are not claims about a final Adam step.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_coherent_patch_native import objective
from .train_b1_contour_coherence import contour_coherence
from .train_b1_distillation import read_flow
from .train_b1_real_region_texture import align_pair
from .coherent_shared_model import warp
from .whitened_b1_optimizer import transform
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(2026100811)
    state,net=load(a.checkpoint);net.__class__=ReadoutB1;net.requires_grad_(True)
    data=m.load_data(a.data);by_path={r['path']:r for r in data['training']}
    by_id={r['id']:r for r in data['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records']
    optim={g['node_key']:g for g in state['optimizer']['param_groups'] if 'node_key' in g}
    named=list(net.named_parameters());params=[p for _,p in named]
    def vector(grads):
        gg=dict(zip((name for name,_ in named),grads));parts=[]
        for key in net.weight:
            gw=gg['weight.'+key];gb=gg['bias.'+key];group=optim[key]
            centered=gw.flatten(1)-gb[:,None]*group['patch_mean']
            transformed=transform(centered,group['input_whitener'].transpose(-1,-2))
            parts.extend((transformed.flatten(),gb.flatten()))
        return torch.cat(parts)
    records=[]
    for spec in (mined[1],mined[5],mined[6]):
        row=by_id[spec['row']['id']]
        candidates=[p for p in data['pairs'] if p['source']==row['video']]
        assert candidates,row['video'];pair=candidates[len(candidates)//2]
        rr=[by_path[pair[k]] for k in ('current','next')];xx,yy=spec['xy'];w,h=spec['wh']
        def batch(key):return torch.stack([torch.from_numpy(m.t.r.read(r[key])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float()/128-1 for r in rr])
        x,t=batch('path'),batch('target')
        lab=torch.stack([torch.from_numpy(m.load_region(r['region'])[yy:yy+h,xx:xx+w].astype(np.int64)) for r in rr])
        f,v=read_flow(pair['flow']);f=f[yy:yy+h,xx:xx+w];v=v[yy:yy+h,xx:xx+w].copy()
        iy,ix=np.mgrid[:h,:w];v*=((ix+f[...,0]>=40)&(ix+f[...,0]<w-40)&(iy+f[...,1]>=40)&(iy+f[...,1]<h-40))
        flow=torch.from_numpy(f.copy()).permute(2,0,1)[None];valid=torch.from_numpy(v)[None,None]
        t,lab,_,fraction=align_pair(x,t,lab,lab,flow,valid);cut=lambda z:z[...,40:-40,40:-40]
        p=net(x)[0];primary=objective(cut(p),cut(t),cut(lab))[0]+.5*contour_coherence(cut(p),cut(t))
        shifted=net(torch.roll(x[:1],(0,1),(-2,-1)))[0]
        phase=1.5*F.l1_loss(cut(p[:1]),cut(torch.roll(shifted,(0,-1),(-2,-1))))
        perturbed=(x[:1]+1/128+torch.randn_like(x[:1])*(.5/128)).clamp(-1,127/128)
        noisy=net(perturbed)[0];noise=5*F.smooth_l1_loss(cut(noisy),cut(p[:1]).detach(),beta=2/128)
        agreement=(x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True)
        confidence=valid*torch.exp(-(agreement/.08)**2)
        temporal=2*(cut((p[1:]-warp(p[:1],flow)).abs())*cut(confidence)).sum()/(3*cut(confidence).sum()+1e-8)
        terms=dict(primary=primary,shift=phase,noise=noise,temporal=temporal);vectors={}
        for name,loss in terms.items():
            grads=torch.autograd.grad(loss,params,retain_graph=True)
            assert all(torch.isfinite(z).all() for z in grads)
            vectors[name]=vector(grads).detach()
        main=vectors['primary'];stats={}
        for name,z in vectors.items():
            stats[name]=dict(weighted_loss=float(terms[name].detach()),gradient_norm=float(z.norm()),
                norm_ratio_to_primary=float(z.norm()/main.norm().clamp_min(1e-12)),
                cosine_with_primary=float((z@main)/(z.norm()*main.norm()).clamp_min(1e-12)))
        side=sum(vectors[k] for k in ('shift','noise','temporal'))
        stats['combined_stability']=dict(gradient_norm=float(side.norm()),norm_ratio_to_primary=float(side.norm()/main.norm()),
            cosine_with_primary=float((side@main)/(side.norm()*main.norm()).clamp_min(1e-12)))
        records.append(dict(source=row['video'],pair_current=pair['current'],xy=[xx,yy],target_aligned_fraction=float(fraction),values=stats))
        print('CURRENT_STABILITY_GRADIENT',row['id'],stats,flush=True)
    out=Path(a.out);out.mkdir(exist_ok=False)
    write_json(out/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),
        source_sha256=file_hash(__file__),scope='Three actual TRAIN pairs. Gradients in saved whitened coordinates, before clipping or Adam. Not a complete optimizer-step or generalization proof.'))


if __name__=='__main__':main()
