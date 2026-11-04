"""Explicit ordered-class median experiment; integer labels, not RGB blur.

This is not in the current deployed graph and must not be reported as its
network output. A useful result would require its own streaming RTL proof.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .skip_filter_net import SkipFilterNet
from .coarse_skip_net import CoarseSkipNet
from .train_luma_regions import ROOT,RUN,SLICES,LEVELS
from .train_b1_distillation import as_tensor,read_image
from .runtime import write_json,file_hash


def decode(q):
    y,co,cg=[np.asarray(v,dtype=np.int16)[q[...,i]] for i,v in enumerate(LEVELS)]
    return np.stack([y+co-cg,y+cg,y-co-cg],-1).clip(0,255).astype(np.uint8)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--coarse',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);root=ROOT/'runs'/RUN;out=root/a.name;out.mkdir(exist_ok=False)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in s['binding']['code'].items())
    net=(CoarseSkipNet if a.coarse else SkipFilterNet)().eval();net.load_state_dict(s['model']);net.scales=s['scales'];data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'))
    configs=[(1,1),(3,1),(3,2),(5,1)]
    for r in data['validation']:
        pred=net(as_tensor(read_image(r['source']))[None]);q=np.stack([pred[0,c].argmax(0).numpy() for c in SLICES],-1).astype(np.uint8)
        for k,it in configs:
            z=q.copy()
            if k>1:
                for _ in range(it):z=cv2.medianBlur(z,k)
            Image.fromarray(decode(z)).save(out/f'{r["id"]}_{k}_{it}.png')
        print('CLASS_RANK',r['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1600,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        paths=[root/'intermediate-target-screen'/f'{key}_0.008.png']+[out/f'{key}_{k}_{it}.png' for k,it in configs]
        for i,(path,title) in enumerate(zip(paths,['区域目标','网络原输出','类别中值3','类别中值3两轮','类别中值5'])):
            d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(i*320,j*298+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='EXPLICIT_CLASS_RANK_DIAGNOSTIC_NOT_RTL',checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),configs=configs,postprocess=True,RGB_blur=False,temporal_blend=False,style_accepted=False))


if __name__=='__main__':main()
