"""Harden only the dominant final720 code without moving its decision centre.

Train-only calibration/refit of actual coefficients. Unlike earlier360p luma
gain probes this acts after the final upsample on all eight learned features.
Control refits only RGB; variants rescale final DW/bias then refit identical
RGB coefficients on the same TRAIN samples. No inference operation is added.
"""
import argparse,copy,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def preactivation(net,nodes):
    k='up2_high_dw';s=(net.up2_high_dw_m/131072).float().flatten()
    w=net.round(net.weight[k]/s[:,None,None,None]);b=net.round(net.bias[k]/s)
    z=F.conv2d(F.pad(nodes['up2_nn'],(1,)*4,mode='replicate'),w,groups=8)
    return net.round((z.double()+b.double()[None,:,None,None])*net.up2_high_dw_m/131072).float()


@torch.no_grad()
def quantize_row(net,key):
    w=net.weight[key];scales=(w.abs().flatten(1).amax(1)/126*131072).ceil().clamp(1,131071)
    getattr(net,key+'_m').copy_(scales.double()[None,:,None,None])
    assert ((w/scales[:,None,None,None]*131072).abs().amax()<=127)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);data=m.load_data(a.data);crops=[];targets=[];ids=[];pres=[]
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video]
        for i in sorted(set((0,len(rows)//2,len(rows)-1))):
            row=rows[i];rgb=m.t.r.read(row['path'])[200:520,480:800];t=m.t.r.read(row['target'])[200:520,480:800]
            x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()/128-1
            nodes=base(x,all_nodes=True);pre=preactivation(base,nodes)[...,40:-40,40:-40]
            pres.append(pre[0,:,::3,::3].flatten(1).T);crops.append(x);targets.append(torch.from_numpy(t[40:-40:3,40:-40:3].reshape(-1,3).copy()).float()-128);ids.append(row['id'])
    z=torch.cat(pres);lo,centre,hi=[torch.quantile(z,q,dim=0) for q in (.1,.5,.9)]
    width=(hi-lo).clamp_min(8);pictures={};records=[];calibration={}
    dominant=int(base.weight['rgb_head'][:,:,0,0].square().mean(0).argmax())
    for name,strength in (('head_control',0),('dominant2',2),('dominant4',4),('dominant8',8)):
        net=copy.deepcopy(base)
        if strength:
            gain=torch.ones(8);gain[dominant]=strength;centre=torch.full((8,),63.5);k='up2_high_dw'
            net.weight[k].mul_(gain[:,None,None,None]);net.bias[k].copy_(gain*(net.bias[k]-centre)+63.5);quantize_row(net,k)
        xx=[]
        for x in crops:
            f=net(x,all_nodes=True)['up2_high_dw'][0,:,40:-40:3,40:-40:3].flatten(1).T
            xx.append(f)
        f=torch.cat(xx).double();t=torch.cat(targets).double();mu=f.mean(0);sd=f.std(0).clamp_min(2)
        d=torch.cat(((f-mu)/sd,torch.ones(len(f),1)),1);ridge=torch.eye(9).double()*len(f)*.0001;ridge[-1,-1]=0
        theta=torch.linalg.solve(d.T@d+ridge,d.T@t);w=(theta[:8]/sd[:,None]).T;b=theta[-1]-mu@w.T
        net.weight['rgb_head'].copy_(w.float()[:,:,None,None]);net.bias['rgb_head'].copy_(b.float());quantize_row(net,'rgb_head')
        sub=out/name;sub.mkdir();raw=net.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw)
        rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));pictures[name,row['id']]=im
            records.append(dict(variant=name,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            contract_sha256=state['contract_sha256'],step=0,qat=True,trained_linear_head_only=True,visual_acceptance=False,
            final720_clip_design=dict(strength=strength,centre=centre.tolist(),dominant_channel=dominant,preserve_decision_centre=True),
            training_ids=ids,extra_inference_ops=0),sub/'candidate.pt')
        calibration[name]=dict(head_fit_mae=float((d@theta-t).abs().mean()),training_features=len(f))
        print('FINAL720_CODES',name,calibration[name],flush=True)
    rows={r['id']:r for r in data['validation']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    board=Image.new('RGB',(420*5,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(title,im) in enumerate([('原目标',Image.open(rows[rid]['target']))]+[(name,pictures[name,rid]) for name in calibration]):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+3),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),calibration=calibration,records=records,training_ids=ids,
        checkpoint_sha256=file_hash(a.checkpoint),validation_used_for_fit=False,visual_acceptance=False,
        kind='Only dominant output-contribution channel hardened about the same63.5 boundary, plus TRAIN head fit; no inference helper'))


if __name__=='__main__':main()
