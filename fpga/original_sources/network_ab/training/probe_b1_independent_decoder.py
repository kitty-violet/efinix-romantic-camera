"""Target-field injection isolates current decoder limits; not prediction."""
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import bits
from .train_b1_independent_region_boundaries import distances
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash


def native_distances(mask):
    fields=[]
    for example in mask.cpu().numpy():
        channels=[]
        for channel in example:
            bit=(channel>=.5).astype(np.uint8)
            if bit.all():s=np.full(bit.shape,16,np.float32)
            elif not bit.any():s=np.full(bit.shape,-16,np.float32)
            else:
                s=cv2.distanceTransform(bit,cv2.DIST_L2,5)-cv2.distanceTransform(1-bit,cv2.DIST_L2,5)
                s-=.5*np.sign(s)
            channels.append(s)
        fields.append(np.stack(channels))
    values=torch.from_numpy(np.stack(fields)).to(mask.device)
    return (F.avg_pool2d(values,4)/4).clamp(-4,4)


@torch.inference_mode()
def phase_regression(net,table):
    yy,xx=torch.meshgrid(torch.arange(128),torch.arange(192),indexing='ij')
    variants={}
    for name in ('point_sampled','native_distance'):
        outputs=[];errors=[]
        for phase in range(4):
            ids=torch.where(xx+yy//2>100+phase,torch.tensor(157),torch.tensor(62))
            mask=table[ids].permute(2,0,1)[None]
            d=native_distances(mask) if name=='native_distance' else distances(mask[:,:,::4,::4])
            fields=(63.5+8*d).round().clamp(0,127)
            out=fragment(net,F.pad(fields,(0,0,0,0,0,1)),'up1_low_pw')['rgb_head']
            outputs.append(out)
            if phase:errors.append(float((out[...,16:-16,16:-16]-outputs[phase-1].roll(1,-1)[...,16:-16,16:-16]).abs().mean()))
        variants[name]=dict(one_pixel_translation_mae=errors,mean=sum(errors)/len(errors))
    return variants


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--native-distance',action='store_true')
    a=ap.parse_args();torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=state['residual_encoding']
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
    palette=torch.tensor(state['palette']);pk=palette_key(palette);table=bits('cpu')
    data=m.load_data();rows={r['id']:r for r in data['validation']};images={};records=[]
    dest=Path(a.out);dest.mkdir(exist_ok=False)
    for key in dict.fromkeys(c[0] for c in m.t.r.c.CROPS):
        row=rows[key];rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
        nodes,_=actual(x,True);direct=nodes['rgb_head'];part=fragment(net,nodes['up1_low_pw'],'up1_low_pw')
        errors={k:float((v-nodes[k]).abs().max()) for k,v in part.items() if k not in ('res1_add','res2_add')}
        assert max(errors.values())==0
        codes=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64))
        targets=palette[codes].permute(2,0,1)[None]
        mask=table[codes].permute(2,0,1)[None]
        distance=native_distances(mask) if a.native_distance else distances(mask[:,:,::4,::4])
        fields=(63.5+8*distance).round().clamp(0,127)
        injected=fragment(net,F.pad(fields,(0,0,0,0,0,1)),'up1_low_pw')['rgb_head']/128
        records.append(dict(id=key,fragment_errors=errors,predicted_mae=float((direct/128-targets).abs().mean()*128),
            target_field_injection_mae=float((injected-targets).abs().mean()*128)))
        images[key]=[m.t.r.c.rgb_image(targets[0]),m.t.r.c.rgb_image(direct[0]/128),m.t.r.c.rgb_image(injected[0])]
        for im,label in zip(images[key],('target','actual_prediction','TARGET_INJECTION_DIAGNOSTIC')):im.save(dest/(key+'_'+label+'.png'))
        print('INDEPENDENT_DECODER_DIAGNOSTIC',records[-1]['id'],records[-1]['predicted_mae'],records[-1]['target_field_injection_mae'],flush=True)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('区域目标','真实输入预测','注入正确字段·不是预测成果'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png')
    write_json(dest/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),target_injection=True,
        injection_is_not_deployable_prediction=True,visual_acceptance=False,device='cpu',native_distance=a.native_distance,
        artificial_translation_regression=phase_regression(net,table)))


if __name__=='__main__':main()
