"""Repair synthetic edge responses while exactly preserving integer kernel DC.

Only seven existing encoder depthwise spatial kernels can change. Every
channel retains its exact integer coefficient sum; biases/PW/scales stay fixed.
This is a capacity/causality experiment, not style acceptance.
"""
import argparse,copy,math,random
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_ternary_packed720 import Packed720B1
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_packed720_step_guard import boundary_batch,step_guard_loss
from .audit_b1_step_ringing import test_inputs,metrics
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


def codes(net,key):
    s=(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
    return net.round(net.weight[key]/s).clamp(-127,127),s


@torch.no_grad()
def project_dc(net,dc,old):
    rejected=0
    for key,total in dc.items():
        w=net.weight[key];q,s=codes(net,key);w.clamp_(-127*s,127*s)
        for _ in range(24):
            q,_=codes(net,key);off=q.clone();off[:,:,1,1]=0
            centre=total-off.sum((-2,-1));bad=centre.abs()>127
            if not bad.any():break
            rejected+=int(bad.sum());mask=bad[:,:,None,None]
            w.copy_(torch.where(mask,(w+old[key])*.5,w))
        q,_=codes(net,key);off=q.clone();off[:,:,1,1]=0
        centre=total-off.sum((-2,-1));bad=centre.abs()>127
        if bad.any():
            # A row exactly on an INT8 threshold can retain the new code under
            # finite-precision bisection. Revert that row to the known-valid
            # previous iterate instead of weakening the DC/range requirement.
            w.copy_(torch.where(bad[:,:,None,None],old[key],w))
            q,_=codes(net,key);off=q.clone();off[:,:,1,1]=0
            centre=total-off.sum((-2,-1));bad=centre.abs()>127
        assert not bad.any(), (key,centre[bad].tolist())
        w[:,:,1,1]=centre*s[:,:,0,0]
        q,_=codes(net,key);assert torch.equal(q.sum((-2,-1)),total)
    return rejected


@torch.inference_mode()
def review(net,data,dest,stage):
    sub=dest/stage;sub.mkdir();raw=net.export();assert len(raw)==20480
    (sub/'diagnostic.bin').write_bytes(raw)
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    src,records=test_inputs();src=src[::3]
    ends=torch.stack([torch.tensor(r[k]) for r in records[::3] for k in ('left','right')]).float()[:,:,None,None].expand(-1,-1,48,48)
    anchor=actual(ends-128)[:,:,24,24]+128;p=actual(src-128)+128
    cases=metrics(p[...,16:-16,32:-32],anchor[::2],anchor[1::2]);mean={k:float(np.mean([r[k] for r in cases])) for k in cases[0]}
    byid={r['id']:r for r in data['validation']};images={};numeric=[]
    for rid,box,label in m.t.r.c.CROPS:
        if rid in images:continue
        row=byid[rid];src=m.t.r.read(row['path']);x=torch.from_numpy(src.transpose(2,0,1).copy())[None].float()-128
        expected=net(x/128,all_nodes=True);got,_=actual(x,True)
        assert all(torch.equal(v,got[k]) for k,v in expected.items())
        out=got['rgb_head']/128;images[rid]=m.t.r.c.rgb_image(out[0]);images[rid].save(sub/(rid+'.png'))
        target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        numeric.append(dict(id=rid,mae=float((out-target).abs().mean()*128)))
    board=Image.new('RGB',(840,628),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        xx=i%2*420;yy=i//2*314;board.paste(images[rid].crop(box),(xx,yy+34));draw.text((xx+4,yy+4),stage+' | '+label,font=font,fill='white')
    board.save(sub/'comparison.png')
    record=dict(step=stage,synthetic=mean,source_images=numeric,byte_node_error=0,byte_size=len(raw))
    print('DC_RIM_REVIEW',record,flush=True);return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=500)
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(923);rng=random.Random(7193);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);net.__class__=Packed720B1;data=m.load_data(a.data);groups=[];dc={}
    for p in net.parameters():p.requires_grad_(False)
    for key,w in net.weight.items():
        if key=='up1_high_dw':break
        if w.shape[1:]==(1,3,3) and key not in ('down1__dw','down2__dw'):
            q,s=codes(net,key);dc[key]=q.sum((-2,-1));w.requires_grad_(True)
            groups.append(dict(params=[w],lr=.0006,row_gain=s*256))
    opt=RowScaledAdam(groups)
    colours=torch.cat((torch.arange(16,241,16).float()[:,None].expand(-1,3),torch.randint(16,241,(81,3)).float()))
    with torch.no_grad():anchors=net(colours[:,:,None,None].expand(-1,-1,16,16)/128-1)[0][:,:,8,8]*128+128
    history=[review(net,data,dest,'before')];losses=[]
    for step in range(1,a.steps+1):
        ids=[]
        for _ in range(4):
            while True:
                i,j=rng.randrange(len(colours)),rng.randrange(len(colours))
                if float((anchors[i]-anchors[j]).abs().max())>=24:break
            ids.append((i,j))
        src,t,core=boundary_batch(colours,anchors,ids,[rng.uniform(0,2*math.pi) for _ in ids],
                                [rng.uniform(-4,4) for _ in ids],[rng.choice((0,0,2,4,12)) for _ in ids])
        opt.zero_grad(set_to_none=True);p=net(src)[0]
        loss,terms=step_guard_loss(p[...,24:-24,24:-24],t[...,24:-24,24:-24],core[...,24:-24,24:-24])
        loss.backward();old={}
        for key in dc:
            w=net.weight[key];assert torch.isfinite(w.grad).all()
            # Centre compensates every off-centre update: chain rule for the
            # fixed-sum parameterization, not independent centre optimisation.
            w.grad.sub_(w.grad[:,:,1:2,1:2].clone());old[key]=w.detach().clone()
        opt.step();reject=project_dc(net,dc,old)
        losses.append(dict(step=step,total=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},range_backoffs=reject))
        if step%50==0:print('DC_RIM_STEP',losses[-1],flush=True)
        if step%250==0 or step==a.steps:
            with torch.no_grad():current=net(colours[:,:,None,None].expand(-1,-1,16,16)/128-1)[0][:,:,8,8]*128+128
            assert torch.equal(current,anchors),'Constant-colour response changed'
            history.append(review(net,data,dest,f'step_{step:06d}'))
            saved=copy.deepcopy(state);saved.update(model=net.state_dict(),step=step,dc_constrained_synthetic_only=True,
                visual_acceptance=False,residual_encoding=copy.deepcopy(net.package['residual_encoding']))
            torch.save(saved,dest/f'candidate_{step:06d}.pt')
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),code_sha256=file_hash(__file__),
        history=history,losses=losses,constant_colour_checks=96,constant_colour_error=0,
        integer_kernel_dc_error=0,synthetic_only_training=True,accepted=False))


if __name__=='__main__':main()
