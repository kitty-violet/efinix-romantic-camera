"""Average trained coefficients, then replay ONE actual B1 package per variant.

No image or video averaging occurs. Fixed decoder/scales must be identical.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--steps',nargs='+',type=int,required=True);a=ap.parse_args();torch.set_num_threads(2)
    root=Path(a.run);states=[torch.load(root/f'candidate_{s:06d}.pt',map_location='cpu',weights_only=True) for s in a.steps]
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);records=[];images={}
    base=states[-1]
    fixed_keys=('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
    for kind,chosen in [('last',states[-1:]),('last3',states[-3:]),('all',states)]:
        state=copy.deepcopy(base);model=state['model']
        for key in model:
            if key.startswith(('weight.','bias.')) and key.split('.',1)[1] not in fixed_keys:
                model[key]=torch.stack([s['model'][key] for s in chosen]).mean(0)
            else:assert all(torch.equal(model[key],s['model'][key]) for s in chosen)
        for key in fixed_keys:
            for prefix in ('weight.','bias.'):
                assert torch.equal(model[prefix+key],base['model'][prefix+key])
        net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
        net.load_state_dict(model);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
        blob=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu')
        sub=dest/kind;sub.mkdir();(sub/'coefficients-padded.bin').write_bytes(blob);ims={};errors=[]
        for row in m.load_data()['validation']:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
            out=net(x,all_nodes=True);ref,_=actual(x*128,True)
            error=max(float((ref[k]-v).abs().max()) for k,v in out.items() if k!='input');assert error==0
            im=m.t.r.c.rgb_image(ref['rgb_head'][0]/128);im.save(sub/(row['id']+'.png'));ims[row['id']]=im;errors.append(error)
        images[kind]=ims
        state.update(coefficient_average_steps=[s['step'] for s in chosen],visual_acceptance=False)
        torch.save(state,sub/'candidate.pt')
        records.append(dict(kind=kind,steps=state['coefficient_average_steps'],sha256=hashlib.sha256(blob).hexdigest(),
            bytes=len(blob),six_native29_node_errors=errors));print('COUNT_COEFFICIENT_AVERAGE',records[-1],flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);rows={r['id']:r for r in m.load_data()['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pictures=[Image.open(rows[key]['target'])]+[images[k][key] for k in ('last','last3','all')]
        for j,(im,title) in enumerate(zip(pictures,('区域目标','末次真实权重','最近3次权重均值·实际整数','全部指定权重均值·实际整数'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'review.json',dict(run=a.run,steps=a.steps,records=records,
        video_or_image_blending=False,one_network_per_output=True,visual_acceptance=False))


if __name__=='__main__':main()
