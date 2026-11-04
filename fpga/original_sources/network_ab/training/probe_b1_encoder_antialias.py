"""Controlled spatial-kernel ablation inside the existing B1 graph.

Preserve each input/output channel's DC gain, replace only its spatial shape
with a binomial kernel. This is a diagnostic, not an implementation of added
BlurPool layers and not a claim that smoothing alone solves stylization.
"""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import write_json,file_hash


@torch.no_grad()
def install(net,keys):
    records=[]
    for key in keys:
        w=net.weight[key];scale=(getattr(net,key+'_m')/131072).float().flatten()
        effective=(w/scale[:,None,None,None]).round()*scale[:,None,None,None]
        dc=effective.sum((-2,-1),keepdim=True)
        shape=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
        proposed=dc*shape
        mult=(proposed.abs().flatten(1).amax(1)*131072/112).ceil().clamp(1,131071)
        getattr(net,key+'_m').copy_(mult.double()[None,:,None,None]);w.copy_(proposed)
        qs=mult/131072;actual=(w/qs[:,None,None,None]).round()*qs[:,None,None,None]
        xx=w.new_tensor([1,-1,1]);ny=(effective*xx).sum((-2,-1)).abs()
        records.append(dict(node=key,old_negative_fraction=float((effective<0).float().mean()),
            dc_error_max=float((actual.sum((-2,-1),keepdim=True)-dc).abs().max()),
            old_horizontal_nyquist_mean=float(ny.mean()),
            new_horizontal_nyquist_mean=float((actual*xx).sum((-2,-1)).abs().mean())))
    return records


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);initial=copy.deepcopy(net.state_dict())
    rows=m.load_data(args.data)['validation'];images={};records=[]
    cases=dict(current=[],stride=['down1__dw','down2__dw'],late=['up1_low_dw'],
        both=['down1__dw','down2__dw','up1_low_dw'])
    for name,keys in cases.items():
        net.load_state_dict(initial);kernels=install(net,keys)
        sub=dest/name;sub.mkdir();raw=net.export();(sub/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');tests=[]
        for row in rows:
            a=m.t.r.read(row['path']);x=torch.from_numpy(a.transpose(2,0,1).copy())[None].float()/128-1
            ours=net(x,all_nodes=True);ref,_=actual(x*128,True)
            err=max(float((v-ref[k]).abs().max()) for k,v in ours.items());assert err==0
            p=ref['rgb_head']/128;t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
            loss,terms=joint_region_loss(p,t,labels)
            shifted=actual(x.roll((1,1),(-2,-1))*128).roll((-1,-1),(-2,-1))/128
            delta=(shifted-p)[...,80:-80,80:-80].abs()*128
            im=m.t.r.c.rgb_image(p[0]);im.save(sub/(row['id']+'.png'));images[name,row['id']]=im
            tests.append(dict(id=row['id'],all_node_error=err,phase_mae=float(delta.mean()),
                phase_over16=float((delta.amax(1)>16).float().mean()),score=float(loss),**{k:float(v) for k,v in terms.items()}))
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng','bounded_training_model')}
        saved.update(model=copy.deepcopy(net.state_dict()),encoder_antialias_keys=keys,
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),added_training_steps=0,
            added_inference_operators=0,external_filter=False,visual_acceptance=False)
        torch.save(saved,sub/'initial.pt')
        mean={k:sum(r[k] for r in tests)/len(tests) for k in tests[0] if k!='id'}
        records.append(dict(name=name,kernels=kernels,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=tests,mean=mean))
        print('ENCODER_ANTIALIAS',name,mean,flush=True)
    by_id={r['id']:r for r in rows};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for key,box,label in m.t.r.c.CROPS:
        board=Image.new('RGB',(1260,628),'#17202a');d=ImageDraw.Draw(board)
        pics=[('target',Image.open(by_id[key]['target']))]+[(n,images[n,key]) for n in cases]
        for i,(name,im) in enumerate(pics):
            xx=i%3*420;yy=i//3*314;board.paste(im.crop(box),(xx,yy+34));d.text((xx+4,yy+5),label+' | '+name,font=font,fill='white')
        board.save(dest/(key+'_comparison.png'))
    write_json(dest/'review.json',dict(parent=args.checkpoint,records=records,added_training_steps=0,
        reference='https://richzhang.github.io/antialiased-cnns/',
        scope='Existing-kernel ablation; no added BlurPool. Phase diagnostic is not full-video acceptance.'))


if __name__=='__main__':main()
