"""Pointwise confidence decoding; no spatial or temporal filtering."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .train_luma_regions import ROOT,RUN,LumaNet,LEVELS,SLICES,display
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import configure_workspace,write_json,file_hash

def decode(q,width):
    parts=[]
    for s,levels in zip(SLICES,LEVELS):
        a=q[:,s];first=a.argmax(1);a2=a.clone();a2.scatter_(1,first[:,None],-32768);second=a2.argmax(1)
        hi=a.gather(1,first[:,None])[:,0];lo=a.gather(1,second[:,None])[:,0];diff=hi-lo
        table=q.new_tensor(levels);p=table[first];r=table[second]
        # Four fixed dyadic weights; only add/subtract/shift/mux in hardware.
        d=p-r
        value=torch.where(diff>=width*3,p,torch.where(diff>=width*2,(7*p+r+4).floor_divide(8),
              torch.where(diff>=width,(3*p+r+2).floor_divide(4),(p+r+1).floor_divide(2))))
        parts.append(value)
    y,co,cg=parts
    return torch.stack((y+co-cg,y+cg,y-co-cg),1).clamp(0,255)

def main():
    configure_workspace();root=ROOT/'runs'/RUN;out=root/'confidence-probe';out.mkdir(exist_ok=False)
    ck=root/'luma/qat_checkpoint_000500.pt';state=torch.load(ck,map_location='cpu',weights_only=True);net=LumaNet().cuda().eval();net.load_state_dict(state['model']);net.scales=state['scales']
    data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'));report=[]
    for row in data['validation']:
        x=as_tensor(read_image(row['source']))[None].cuda()
        with torch.no_grad():
            q=(net(x)/net.scales['rgb']).round();variants={'hard':to_rgb(display(q)[0])}
            for w in (1,2,4):variants[f'w{w}']=decode(q,w)[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
        for k,v in variants.items():Image.fromarray(v).save(out/(row['id']+'_'+k+'.png'))
        report.append(dict(id=row['id'],top1_top2_margin_mean=[float((v.topk(2,dim=1).values[:,0]-v.topk(2,dim=1).values[:,1]).mean()) for v in [q[:,s] for s in SLICES]]))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1280,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        x,y=xy
        for i,k in enumerate(('hard','w1','w2','w4')):
            d.text((320*i+5,298*j+5),k,font=font,fill='white');board.paste(Image.open(out/(key+'_'+k+'.png')).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(records=report,checkpoint_sha256=file_hash(ck),head_scale=net.scales['rgb'],spatial_filter=False,temporal_filter=False,RTL_verified=False))

if __name__=='__main__':main()
