"""Test whether the reachable contour target is a fixed point of RGB losses.

Known TRAIN latent fields only; no source model is trained or delivered.
This checks objective self-consistency independently of encoder capacity.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_ordered_distance_geometry import distance_fields,set_ramp
from .audit_b1_optimized_decoder_ceiling import decode
from .train_b1_ordered_distance_rims import distance_objective
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_region_tail_flatness import region_tail_flatness
from .train_b1_packed720_local_rims import local_rim_loss
from .train_b1_boundary_profile import boundary_profile
from .train_b1_contour_coherence import contour_coherence
from .train_b1_ordered_boundary_fields import rank_loss
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def objective(p,t,labels,profile_fn=boundary_profile):
    region,v=joint_region_loss(p,t,labels);tail=region_tail_flatness(p,labels)
    rim=local_rim_loss(p,t,labels);profile,terms=profile_fn(p,t,labels)
    curve=contour_coherence(p,t)
    v.update(terms);v.update(tail=tail,rim=rim,curve=curve)
    return .7*region+4*v['region_flatness']+v['adjacent_region_contrast']+4*tail+4*curve+4*rim+4*profile,v


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--reference-profile',action='store_true');ap.add_argument('--steps',type=int,default=100);a=ap.parse_args()
    from .train_b1_reference_boundary_profile import reference_boundary_profile
    profile_fn=reference_boundary_profile if a.reference_profile else boundary_profile
    torch.set_num_threads(2);state,net=load(a.checkpoint);net.requires_grad_(False);set_ramp(net,11)
    dest=Path(a.out);dest.mkdir(exist_ok=False);d=m.load_data(a.data)
    videos=sorted({r['video'] for r in d['training']});rows=[]
    for video in (videos[0],videos[len(videos)//2],videos[-1]):
        candidates=[r for r in d['training'] if r['video']==video];rows.append(candidates[len(candidates)//2])
    setup=state['binary_coverage720'];levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255);pk=tuple(map(tuple,palette.tolist()))
    truth=[];labels=[]
    for row in rows:
        code=torch.from_numpy(region_codes(row['target'],row['region'],pk)[232:488,512:768].astype(np.int64))[None]
        truth.append(distance_fields(code,8)[0]);labels.append(torch.from_numpy(m.load_region(row['region'])[232:488,512:768].copy().astype(np.int64)))
    truth=torch.stack(truth);labels=torch.stack(labels);z=torch.nn.Parameter(F.pad(truth,(0,0,0,0,0,2)))
    with torch.no_grad():target=decode(net,z)
    p=decode(net,z);loss,v=objective(p,target,labels,profile_fn);gradient=torch.autograd.grad(loss,z)[0]
    start=dict(loss=float(loss.detach()),gradient_norm=float(gradient.norm()),values={k:float(q.detach()) for k,q in v.items()})
    opt=torch.optim.Adam([z],lr=.5)
    for step in range(a.steps):
        p=decode(net,z);rgb,v=objective(p,target,labels,profile_fn);full,band=distance_objective(z,truth)
        loss=rgb+16*full+8*band+10*rank_loss(z);opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():z.clamp_(0,127);z[:,30:]=0
    with torch.no_grad():final=decode(net,z);rgb,v=objective(final,target,labels,profile_fn)
    board=Image.new('RGB',(512,870),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i in range(3):
        for j,(name,image) in enumerate((('可达目标·非源预测',target),(f'优化loss{a.steps}步',final))):
            board.paste(m.t.r.c.rgb_image(image[i]),(j*256,i*290+34));draw.text((j*256+2,i*290+3),name,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(train_ids=[r['id'] for r in rows],
      start=start,reference_profile=a.reference_profile,steps=a.steps,final_rgb_loss=float(rgb),final_rgb_mae_from_target=float((final-target).abs().mean()*128),
      final_values={k:float(q) for k,q in v.items()},source_prediction=False,not_deployable=True,
      source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint)))
    print('REACHABLE_LOSS_FIXEDPOINT_DONE',start['loss'],float(rgb),float((final-target).abs().mean()*128),flush=True)


if __name__=='__main__':main()
