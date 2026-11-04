"""Offline target feasibility only: exact decoder after area cleanup.

These are NOT predictions and NOT a deliverable weight package. They answer
whether increasing the component area alone can improve the current geometry.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_decisive_regions import modal_region_cleanup
from .build_b1_ordered_palette import bits
from .probe_b1_decoder_coverage import fragment
from .minimum_area_blocks import components
from .runtime import write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(s['model']);net.package['residual_encoding']=s['residual_encoding'];net.quantized=True
    pal=(np.array(s['palette'])*128+128).round().clip(0,255).astype(np.uint8);table=bits('cpu')
    pics={};records=[]
    for row in m.load_data()['validation']:
        if row['id'] not in ('val_stairs_090','val_plaza_090','val_plaza_210'):continue
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        nodes=net(x,all_nodes=True);v=nodes['up2_low_dw'][0,:15].numpy()>0
        ids=v[:7].sum(0)*25+v[7:11].sum(0)*5+v[11:].sum(0)
        images=[m.t.r.c.rgb_image(nodes['rgb_head'][0]/128)]
        for area in (256,1024,2304):
            cleaned,_=modal_region_cleanup(ids,pal,area//4,2)
            z=torch.zeros(1,16,*ids.shape);z[0,:15]=table[torch.tensor(cleaned)].permute(2,0,1)*127
            predicted=fragment(net,z,'up2_low_dw')['rgb_head'][0]/128
            image=m.t.r.c.rgb_image(predicted);image.save(out/(row['id']+f'_OFFLINE_area{area}.png'));images.append(image)
            sizes=np.bincount(components(cleaned).ravel())*4
            records.append(dict(id=row['id'],minimum_area=area,actual_minimum=int(sizes.min()),
                colour_regions=len(sizes),pixel_fraction_changed=float((cleaned!=ids).mean()),
                network_prediction=False,offline_injected_labels=True))
        pics[row['id']]=images
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(pic,title) in enumerate(zip(pics[key],('实际B1预测','离线目标≥256','离线目标≥1024','离线目标≥2304'))):
            board.paste(pic.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(out/'OFFLINE_target_feasibility.png')
    write_json(out/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),records=records,
        not_a_learned_result=True,no_deployable_area_guarantee=True))
    print('MODAL_AREA_TARGET_PROBE_COMPLETE',records,flush=True)


if __name__=='__main__':main()
