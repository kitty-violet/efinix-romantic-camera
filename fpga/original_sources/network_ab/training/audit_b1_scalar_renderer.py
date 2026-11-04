"""Separate a scalar-field decoder's limitations from encoder prediction.

The oracle injects label-derived latent colours. It is never an actual style
output, weight-package preview, or valid substitute for source-only inference.
"""
import argparse
import copy
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True)
    a=ap.parse_args();configure_workspace();set_reproducible(20261006127)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    net=m.t.r.initialize();state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['model']);net.quantized=True
    decoder=copy.deepcopy(net);nodes=decoder.plan['nodes']
    start=next(i for i,r in enumerate(nodes) if r['id']=='up1_low_dw')
    decoder.plan['nodes']=nodes[start:];decoder.plan['nodes'][0]['inputs']=['input']
    images={};records=[]
    for row in m.load_data()['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        full=net(x,all_nodes=True);normal=decoder(full['res3_add']/128)[0]
        error=float((normal-full['rgb_head']/128).abs().max());assert error==0
        z=F.avg_pool2d((target+1)*64,4);r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
        colour=torch.cat(((r+2*g+b)/4,(r-b)/2+64,(-r+2*g-b)/4+64),1)
        latent=torch.zeros_like(full['res3_add'])
        for j in range(24):latent[:,j]=colour[:,(j%12)//4]
        oracle=decoder(latent/128)[0]
        out=m.t.r.c.rgb_image(oracle[0]);out.save(dest/(row['id']+'_oracle_diagnostic.png'))
        images[row['id']]=[Image.fromarray(m.t.r.read(row['target'])),out,m.t.r.c.rgb_image(normal[0])]
        records.append(dict(id=row['id'],split_decoder_error=error,**m.t.r.metrics(oracle,target)))
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],['离线目标','诊断：给定正确区域颜色','实际：仅输入原片'])):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png');write_json(dest/'review.json',dict(checkpoint=a.checkpoint,
        checkpoint_sha256=file_hash(a.checkpoint),records=records,diagnostic_only=True,visual_accepted=False,
        middle_column_uses_target_latent=True))
    print('SCALAR_RENDERER_AUDIT',dest,flush=True)


if __name__=='__main__':main()
