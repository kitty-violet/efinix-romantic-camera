"""Test quantization of learned decoder coordinates before native edge recovery.

Reuses the earlier quantizer-order principle, now with all eight learned codec
coordinates rather than fixed colour-count fields. No postfilter or new op.
TRAIN feature quantiles set offsets/scales; validation is never calibration.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import configure_workspace,file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=Path(args.out);dest.mkdir(exist_ok=False)
    data=m.load_data(args.data);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False).to(args.device);net.load_state_dict(state['model'])
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=False
    def tensor(path):return torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].to(args.device).float()/128-1
    groups={}
    for row in data['training']:groups.setdefault(row['video'],[]).append(row)
    rows=[]
    for group in groups.values():rows.extend(group[i] for i in sorted(set(np.linspace(0,len(group)-1,3).astype(int))))
    values=[]
    for row in rows:values.append(net(tensor(row['path']),all_nodes=True)['up2_low_pw'][0,:,::8,::8].flatten(1))
    activation=torch.cat(values,1);lo=(torch.quantile(activation,.001,dim=1)-1).clamp_min(0)
    hi=torch.quantile(activation,.999,dim=1)+1;span=(hi-lo).clamp_min(8)
    original=copy.deepcopy(net.state_dict());images={};records=[]
    for levels in (0,31,15,7):
        net.load_state_dict(original);key='up2_low_pw';nextkey='up2_high_dw'
        if levels:
            gain=levels/span
            w=net.weight[nextkey].clone()
            net.weight[key].mul_(gain[:,None,None,None]);net.bias[key].sub_(lo).mul_(gain)
            net.weight[nextkey].div_(gain[:,None,None,None]);net.bias[nextkey].add_(lo*w.sum((1,2,3)))
        # Calibrate this new FP32 checkpoint exactly once. Encoder function is
        # the same across the variants before Q17 rounding.
        net.calibrate();net.quantized=True;raw=net.export();sub=dest/f'levels{levels}';sub.mkdir()
        (sub/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=args.device);errors=[]
        for row in data['validation']:
            x=tensor(row['path']);t=tensor(row['target']);nodes=net(x,all_nodes=True);ref,_=actual(x*128,True)
            errors.append(max(float((v-ref[k]).abs().max()) for k,v in nodes.items()))
            p=ref['rgb_head']/128;lab=torch.tensor(m.load_region(row['region']).astype(np.int64),device=args.device)[None]
            loss,vals=joint_region_loss(p,t,lab);image=m.t.r.c.rgb_image(p[0]);image.save(sub/(row['id']+'.png'))
            images[levels,row['id']]=image;records.append(dict(levels=levels,id=row['id'],mae=float((p-t).abs().mean()*128),
                loss=float(loss),**{k:float(v) for k,v in vals.items()}))
        assert max(errors)==0 and len(raw)==20480
        saved=dict(model=copy.deepcopy(net.state_dict()),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            contract_sha256=state['contract_sha256'],qat=True,step=0,learned_codec_states=True,
            nominal_levels=levels,calibration_low=lo.cpu(),calibration_span=span.cpu(),
            parent_checkpoint=args.checkpoint,parent_sha256=file_hash(args.checkpoint),
            visual_acceptance=False,extra_inference_operations=0)
        torch.save(saved,sub/'initial.pt');print('CODEC_STATES',levels,'nodes',max(errors),flush=True)
    board=Image.new('RGB',(420*5,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);byid={r['id']:r for r in data['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        choices=[('target',Image.open(byid[key]['target']))]+[(f'codec states {k}',images[k,key]) for k in (0,31,15,7)]
        for j,(title,img) in enumerate(choices):
            board.paste(img.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(records=records,
        calibration_ids=[r['id'] for r in rows],low=lo.tolist(),span=span.tolist(),
        actual_bytes=True,no_added_ops=True,visual_acceptance=False,source_prediction=True))


if __name__=='__main__':main()
