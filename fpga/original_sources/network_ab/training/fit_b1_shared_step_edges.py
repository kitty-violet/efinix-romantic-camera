"""Controlled, learnable two-colour edges through the actual fixed B1.

Unlike the previous RGB endpoint step guard, supervise the shared ordinal
distance fields directly on a fixed synthetic batch. This isolates fitting a
simple boundary from fitting natural semantic targets. Not a deployable style
result and not a capacity bound if the remaining error is nonzero.
"""
import argparse,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_ordered_distance_geometry import distance_fields
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_ordered_distance_rims import distance_objective
from .fit_b1_distance_geometry import optimizer
from .centered_b1_optimizer import calibration_means
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def examples(state,heldout=False):
    pairs=(((3,4,4),(11,4,4)),((5,3,4),(13,4,4)),((5,3,5),(9,5,3)))
    yy,xx=torch.meshgrid(torch.arange(160),torch.arange(192),indexing='ij')
    slopes=(.37,-.63,1.13) if heldout else (.2,-.5,1.)
    mat=torch.tensor(state['binary_coverage720']['matrix']).float()
    bias=torch.tensor(state['binary_coverage720']['bias']).float()
    xs=[];codes=[];metadata=[]
    for (a,b),slope in zip(pairs,slopes):
        ca,cb=[mat@torch.tensor(c).float()+bias for c in (a,b)]
        assert ca.min()>=-128 and ca.max()<=127 and cb.min()>=-128 and cb.max()<=127
        mask=xx>95.5+slope*(yy-79.5)+(1.5 if heldout else 0)
        rgb=torch.where(mask[None],cb[:,None,None],ca[:,None,None])
        xs.append(rgb/128);codes.append(torch.where(mask,b[0]*81+b[1]*9+b[2],a[0]*81+a[1]*9+a[2]))
        metadata.append(dict(a=a,b=b,slope=slope))
    return torch.stack(xs),torch.stack(codes),metadata


@torch.no_grad()
def measure(net,x,truth,target):
    nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128
    f,b=distance_objective(nodes['up1_low_dw'][...,10:-10,10:-10],truth[...,10:-10,10:-10])
    cut=lambda z:z[...,40:-40,40:-40]
    e=(cut(p)-cut(target))*128
    # Within-range pale shoulders can exist with zero envelope excess; report
    # full RGB error as well. This is not a semantic white-object detector.
    upper=F.max_pool2d(target,9,1,4);lower=-F.max_pool2d(-target,9,1,4)
    excess=cut(torch.maximum(F.relu(p-upper),F.relu(lower-p)))*128
    return dict(full=float(f),band=float(b),mae=float(e.abs().mean()),max_rgb_error=float(e.abs().max()),
        beyond_local_rgb_fraction=float((excess.amax(1)>4).float().mean())),p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--device',default='cpu');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026100712)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state,net=load(a.checkpoint);net.__class__=OrdinalBandB1;net.to(a.device)
    dest=Path(a.out);dest.mkdir(exist_ok=False);x,codes,meta=examples(state);hx,hcodes,hmeta=examples(state,True)
    x,hx=x.to(a.device),hx.to(a.device);truth=distance_fields(codes.to(a.device),8);ht=distance_fields(hcodes.to(a.device),8)
    with torch.no_grad():
        target=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
        htarget=fragment(net,F.pad(net.round(ht),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
        before=net(x)[0].clone()
    opt=optimizer(net,centred=True,means=calibration_means(net,x));params=[p for p in net.parameters() if p.requires_grad]
    history=[]
    write_json(dest/'config.json',dict(args=vars(a),synthetic=meta,heldout=hmeta,
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),not_natural_generalization=True))
    for step in range(a.steps+1):
        if step%250==0 or step==a.steps:
            seen,p=measure(net,x,truth,target);unseen,_=measure(net,hx,ht,htarget)
            history.append(dict(step=step,train=seen,heldout=unseen));write_json(dest/'history.json',history)
            ss=copy.deepcopy(state);ss.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,synthetic_shared_step_fit=True,visual_acceptance=False)
            torch.save(ss,dest/f'candidate_{step:06d}.pt')
            board=Image.new('RGB',(768,582),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
            for i in range(3):
                for j,(title,im) in enumerate((('双色输入',x),('已知可达目标',target),('拟合前',before),('拟合后',p))):
                    board.paste(m.t.r.c.rgb_image(im[i]),(j*192,i*194+34));draw.text((j*192+2,i*194+4),title,font=font,fill='white')
            board.save(dest/f'comparison_{step:06d}.png');print('SHARED_STEP_EDGE',history[-1],flush=True)
        if step==a.steps:break
        nodes=net(x,all_nodes=True);full,band=distance_objective(nodes['up1_low_dw'][...,10:-10,10:-10],truth[...,10:-10,10:-10])
        loss=16*full+8*band;opt.zero_grad();loss.backward();opt.prepare_gradients()
        torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
        raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=a.device)
        for inputs in (x,hx):
            nn,_=actual(net.round(inputs*128),True);pred=net(inputs,all_nodes=True)
            assert all(torch.equal(v,nn[k]) for k,v in pred.items())
        assert len(raw)==20480;(dest/'synthetic_only.bin').write_bytes(raw)
    write_json(dest/'review.json',dict(history=history,all_node_error=0,bytes=20480,steps=a.steps,synthetic_only=True,visual_acceptance=False))
    print('SHARED_STEP_EDGE_COMPLETE',flush=True)


if __name__=='__main__':main()
