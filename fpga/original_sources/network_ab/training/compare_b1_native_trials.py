"""Same-image comparison of actual B1 coefficient reloads; no output edits."""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .fit_b1_sixteen_regions import palette_key, region_class_map
from .minimum_area_blocks import components, boundary_mask
from .train_b1_native_contrast_regions import boundary_contrast_loss
from .runtime import write_json, file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    dest=Path(args.out);dest.mkdir(exist_ok=False);data=m.load_data();pictures={};records=[];names=[]
    for spec in args.models:
        name,path=spec.split('=',1);names.append(name)
        state=torch.load(path,map_location='cpu',weights_only=True)
        net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
        net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
        blob=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu')
        palette=torch.tensor(state['palette']);pk=palette_key(palette)
        for row in data['validation']:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);ref=net(x/128,all_nodes=True)
            error=max(float((v-ref[k]).abs().max()) for k,v in nodes.items());assert error==0
            output=nodes['rgb_head']/128;coords=nodes['up2_high_dw'][0,:3].numpy()
            pred=(coords[0]*25+coords[1]*5+coords[2]).astype(np.int32)
            codes=region_class_map(row['target'],row['region'],pk)
            truth=palette[torch.tensor(codes.astype(np.int64))].permute(2,0,1)[None]
            lab=components(pred);sizes=np.bincount(lab.ravel())
            interior=cv2.dilate(boundary_mask(codes),np.ones((7,7),np.uint8))==0
            rec=dict(model=name,id=row['id'],all_node_error=error,
                small_area_fraction=float(sizes[sizes<256].sum()/sizes.sum()),
                interior_wrong_fraction=float((pred[interior]!=codes[interior]).mean()),
                boundary_contrast_error=float(boundary_contrast_loss(output,truth)),
                target_colour_mae=float((output-truth).abs().mean()*128))
            records.append(rec);im=m.t.r.c.rgb_image(output[0]);pictures[name,row['id']]=im
            if name==names[0]:pictures['target',row['id']]=m.t.r.c.rgb_image(truth[0])
        print('NATIVE_TRIAL_COMPARE',name,[r for r in records if r['model']==name],flush=True)
    width=420*(1+len(names));board=Image.new('RGB',(width,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,name in enumerate(['target']+names):
            board.paste(pictures[name,key].crop(box),(j*420,i*314+34))
            draw.text((j*420+4,i*314+5),label+' | '+('量化训练目标' if name=='target' else name),font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,
        models=[dict(label=s.split('=',1)[0],path=s.split('=',1)[1],sha256=file_hash(s.split('=',1)[1])) for s in args.models],
        actual_reloaded_integer=True,postfilter=False,visual_acceptance=False))


if __name__=='__main__':main()
