"""TRAIN-calibrated positive channel rescaling inside the fixed B1 graph.

Scale a feature and inversely scale consumers. Residual merge branches share
one scale. Unlike unbounded ReLU theory, B1 clips at127, so calibration and
float-function error are explicit checks, not a global equivalence claim.
Concept: https://arxiv.org/abs/1906.04721 ; no AIMET installation/API used.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_projected_count_regions import region_geometry
from .runtime import write_json,file_hash


def load(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model'])
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
    return state,net


@torch.no_grad()
def apply_scales(net,scales):
    for node in net.plan['nodes']:
        name=node['id'];key=name.replace('.','__')
        if key not in net.weight:continue
        w,b=net.weight[key],net.bias[key]
        s=scales[name];previous=scales[node['inputs'][0]]
        if node['kind']=='dw3x3':w.mul_((s/previous)[:,None,None,None])
        else:w.mul_(s[:,None,None,None]/previous[None,:,None,None])
        b.mul_(s)
        if not (torch.equal(s,torch.ones_like(s)) and torch.equal(previous,torch.ones_like(previous))):
            # Existing per-output Q17 multiplier, not a new activation scale.
            quantum=(w.abs().flatten(1).amax(1)/126).clamp_min(1/131072)
            mult=(quantum*131072).ceil().clamp(1,131071)
            assert (w.abs().flatten(1).amax(1)<=127*mult/131072+1e-5).all()
            getattr(net,key+'_m').copy_(mult.double()[None,:,None,None])


def scales_for(net,maximum,variant):
    scale={'input':torch.ones(3)}
    for r in net.plan['nodes']:scale[r['id']]=torch.ones(r['channels'])
    end=next(i for i,r in enumerate(net.plan['nodes']) if r['id']=='up1_low_pw')
    names=[r['id'] for r in net.plan['nodes'][:end+1]]
    if variant=='logits':names=['up1_low_pw']
    for name in names:
        v=maximum[name]
        scale[name]=torch.where(v>.5,(116/(v+2)).clamp(1,4),torch.ones_like(v))
    if variant=='all_encoder':
        # The skip and main operand must carry identical units at every add.
        shared=['down2.pw']+[f'res{i}_{tail}' for i in range(1,4) for tail in ('conv2.pw','add')]
        common=torch.stack([scale[key] for key in shared]).amin(0)
        for key in shared:scale[key]=common.clone()
    for r in net.plan['nodes']:
        if r['kind']=='nearest_x2':scale[r['id']]=scale[r['inputs'][0]].clone()
    return scale


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);data=m.load_data(a.data)
    calibration=[]
    for source in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==source]
        calibration.extend(rows[i] for i in sorted(set((0,len(rows)//2,len(rows)-1))))
    fp=copy.deepcopy(base);fp.quantized=False;maximum={};cached=[]
    def read(row):return torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
    for row in calibration:
        x=read(row);nodes=fp(x,all_nodes=True)
        for name,z in nodes.items():
            v=z.abs().amax((0,2,3));maximum[name]=torch.maximum(maximum.get(name,v),v)
        # Only store input/path; avoid retaining every full intermediate tensor.
        cached.append(row)
    print('ENCODER_GAUGE_CALIBRATED',len(calibration),flush=True)
    variants={'base':base};checks={}
    for label in ('logits','all_encoder'):
        net=copy.deepcopy(base);scales=scales_for(net,maximum,label);apply_scales(net,scales)
        floating=copy.deepcopy(net);floating.quantized=False
        err=0.
        for row in calibration:
            x=read(row);err=max(err,float((fp(x)[0]-floating(x)[0]).abs().max()*128))
        # Only finite-sample floating equivalence is asserted; unseen activations
        # may exceed calibration range and must be assessed independently.
        assert err<.05,(label,err)
        out=dest/label;out.mkdir();s=copy.deepcopy(state);s['model']=net.state_dict()
        s.update(encoder_positive_gauge=label,training_steps_since_gauge=0,visual_acceptance=False)
        torch.save(s,out/'initial.pt');(out/'initial.bin').write_bytes(net.export())
        assert len(net.export())==20480
        variants[label]=net;checks[label]=dict(calibration_float_max_error_rgb=err,
            scales={k:v.tolist() for k,v in scales.items() if not torch.equal(v,torch.ones_like(v))})
        print('ENCODER_GAUGE_FLOAT_CHECK',label,err,flush=True)
    images={};records=[]
    for label,net in variants.items():
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        for row in data['validation']:
            x=read(row);nodes=net(x,all_nodes=True);other,_=actual(x*128,True)
            integer=other['rgb_head']
            errors={k:float((v-other[k]).abs().max()) for k,v in nodes.items()}
            assert max(errors.values())==0
            z=nodes['up2_low_dw'][0,:15];code=(81*z[:7].sum(0)+9*z[7:11].sum(0)+z[11:15].sum(0)).long().numpy()
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()-128
            im=m.t.r.c.rgb_image(integer[0]/128);images[label,row['id']]=im
            im.save(dest/f'{label}_{row["id"]}.png')
            records.append(dict(variant=label,id=row['id'],mae=float((integer-target).abs().mean()),**region_geometry(code),all_node_error=max(errors.values())))
        print('ENCODER_GAUGE_INTEGER',label,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    byid={r['id']:r for r in data['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('target',Image.open(byid[key]['target']))]+[(k,images[k,key]) for k in variants]
        for j,(title,im) in enumerate(choices):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),checks=checks,records=records,
        train_calibration_ids=[r['id'] for r in calibration],validation_calibration=False,byte_size=20480,added_inference_ops=0,visual_acceptance=False))


if __name__=='__main__':main()
