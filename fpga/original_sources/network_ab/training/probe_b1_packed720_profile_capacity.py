"""Known-target latent optimisation tests the complete-profile objective.

CPU diagnostic only: no source generalisation or deployable-weight claim.
Show only the supervised interior, avoiding an unoptimised border in reviews.
"""
import argparse,random
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_ternary_latent_curve import decode
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_packed720 import Packed720B1
from .train_b1_ternary_regions import region_codes
from .train_b1_boundary_profile import BoundaryProfileTraining
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=400)
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(718);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);net.__class__=Packed720B1;net.requires_grad_(False)
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    palette=(levels@torch.tensor(state['packed720']['matrix']).T+torch.tensor(state['packed720']['bias'])+128).clamp(0,255)
    pk=tuple(map(tuple,palette.tolist()));byid={r['id']:r for r in m.load_data(a.data)['validation']}
    xs=[];ts=[];labs=[];classes=[]
    for rid,box,label in m.t.r.c.CROPS:
        x0,y0,x1,y1=box;w,h=x1-x0,y1-y0;x0=x0//4*4;y0=y0//4*4;sl=np.s_[y0:y0+h,x0:x0+w];row=byid[rid]
        def read(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy()).float()/128-1
        xs.append(read(row['path']));ts.append(read(row['target']));labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)))
        classes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
    x,t=torch.stack(xs),torch.stack(ts)
    with torch.no_grad():full=net(x,all_nodes=True);initial=full['up1_low_pw'].clone();reference=decode(net,initial,False)
    assert torch.equal(reference['rgb_head'],full['rgb_head'])
    check=decode(net,initial,True);assert all(torch.equal(v,check[k]) for k,v in reference.items())
    tr=BoundaryProfileTraining.__new__(BoundaryProfileTraining);tr.net=net;tr.args=argparse.Namespace(classification=.5)
    tr.rng=random.Random(7);tr.labels=torch.stack(labs);tr.class_labels=torch.stack(classes);tr.motion_fraction=x.new_zeros(());tr.step=1
    z=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([z],lr=.3);history=[]
    for i in range(1,a.steps+1):
        tr.nodes=decode(net,z);p=tr.nodes['rgb_head']/128
        loss,v=tr.additional_loss(p[...,40:-40,40:-40],t[...,40:-40,40:-40],False)
        opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(z.grad).all();opt.step()
        with torch.no_grad():z.clamp_(0,127)
        if i%100==0:
            history.append(dict(step=i,loss=float(loss.detach()),**{k:float(v[k].detach()) for k in ('region_flatness','rgb_profile_shape','rgb_profile_contrast','profile_samples')}))
            print('PACKED720_PROFILE_LATENT',history[-1],flush=True)
    with torch.no_grad():
        nodes=decode(net,z,False);other=fragment(net,net.round(z),'up1_low_pw')
        assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        pictures=[('目标',t),('真实源预测',reference['rgb_head']/128),('目标潜变量拟合·非预测',nodes['rgb_head']/128)]
        board=Image.new('RGB',(1020,234*4),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
        for i,(_,_,label) in enumerate(m.t.r.c.CROPS):
            for j,(name,images) in enumerate(pictures):
                im=m.t.r.c.rgb_image(images[i,:,40:-40,40:-40]);board.paste(im,(340*j,234*i+34));draw.text((340*j+3,234*i+4),label+' | '+name,font=font,fill='white')
        board.save(dest/'diagnostic.png')
    torch.save(dict(latent=z.detach(),source_prediction=False),dest/'latent_not_weights.pt')
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),history=history,
        proxy_forward_error=0,fragment_error=0,target_optimised_latents=True,source_prediction=False,
        display_supervised_interior_only=True,accepted=False))


if __name__=='__main__':main()
