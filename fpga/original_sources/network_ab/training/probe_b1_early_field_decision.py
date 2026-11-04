"""Manual180p field steepening BEFORE interpolation, inside B1 coefficients.

Unlike the earlier360p decision/vote probe, this changes the existing180p
pointwise classifier so interpolation sees flatter field interiors. It may
harm geometry; actual integer images and component statistics decide that.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .audit_b1_native_colour_regions import nearest_codes
from .minimum_area_blocks import components,boundary_mask
from .runtime import file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    initial=copy.deepcopy(net.state_dict());palette=(np.array(state['palette'])*128+128).round().clip(0,255)
    data=m.load_data();rows=data['validation'];images={};records=[]
    for gain in (1,4,16):
        net.load_state_dict(initial);key='up1_low_pw'
        if gain!=1:
            mult=getattr(net,key+'_m');assert (mult[:,:15]*gain).max()<=131071
            net.weight[key][:15]*=gain
            net.bias[key][:15]=gain*net.bias[key][:15]+(1-gain)*63.5
            mult[:,:15]*=gain
        sub=dest/f'gain{gain}';sub.mkdir();blob=net.export();(sub/'coefficients-padded.bin').write_bytes(blob)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu');details=[]
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);ref=net(x/128,all_nodes=True)
            error=max(float((v-ref[k]).abs().max()) for k,v in nodes.items());assert error==0
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);images[row['id'],gain]=im
            im.save(sub/(row['id']+'.png'));codes,stats=nearest_codes(np.array(im),palette)
            lab=components(codes);sizes=np.bincount(lab.ravel())
            stats.update(id=row['id'],all_node_error=error,regions=len(sizes),
                small_area_pixel_fraction=float(sizes[sizes<256].sum()/sizes.sum()),
                boundary_density=float(boundary_mask(codes).mean()))
            details.append(stats)
        if gain!=1:
            saved={k:v for k,v in state.items() if k not in ('optimizer','bounded_training_model','rng','torch_rng','cuda_rng')}
            saved.update(model=copy.deepcopy(net.state_dict()),qat=True,step=0,early_field_gain=gain,
                manual_coefficient_change=True,minimum_area_inference_guarantee=False,
                training_proxy_requires_update=True,visual_acceptance=False)
            torch.save(saved,sub/'candidate.pt')
        records.append(dict(gain=gain,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),records=details))
        print('EARLY_FIELD_DECISION',records[-1],flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by_id[key]['target'])]+[images[key,g] for g in (1,4,16)]
        for j,(im,title) in enumerate(zip(pics,('离线区域目标','原整数B1','180p字段增益4·整数B1','180p字段增益16·整数B1'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),records=records,
        added_operators=0,postfilter=False,visual_acceptance=False,actual_board=False))


if __name__=='__main__':main()
