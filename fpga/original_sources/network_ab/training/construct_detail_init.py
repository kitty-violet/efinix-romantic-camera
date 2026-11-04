"""Construct a continuous weak-detail shrinker inside the existing RGB graph.

The native path carries color and signed luminance contrasts. The low-resolution
path supplies a local luminance plane. ReLUs implement a soft dead-zone around
zero detail; final linear RGB recombination retains chroma. All operations are
already in DetailRGBNet. This is a weight initialization, not an output filter.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .detail_rgb_net import DetailRGBNet
from .region_detail_target import ROOT,RUN,manifest
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible
from .train_b1_distillation import as_tensor,to_rgb,read_image


@torch.no_grad()
def initialize(net,threshold=.12,gain=1.15):
    k=torch.tensor([[1,2,1],[2,4,2],[1,2,1]],device='cuda',dtype=torch.float32)/16
    lum=torch.tensor([.299,.587,.114],device='cuda')
    stem=net.conv('stem');stem.weight.zero_();stem.bias.zero_()
    for c in range(3):stem.weight[c,c,1,1]=1;stem.bias[c]=1
    stem.weight[3,:,1,1]=lum;stem.bias[3]=1
    stem.weight[4,:,1,1]=-lum;stem.bias[4]=1
    stem.weight[6,:,1,1]=lum;stem.bias[6]=1
    stem.weight[7,:,1,1]=-lum;stem.bias[7]=1
    for name in ('down.dw','decode.dw','up.dw'):
        c=net.conv(name);c.weight.copy_(k[None,None].expand_as(c.weight));c.bias.zero_()
    c=net.conv('down.pw');c.weight.zero_();c.bias.zero_()
    for j in range(16):c.weight[j,j%8,0,0]=1
    for block in (1,2):
        for layer in (1,2):
            c=net.conv(f'res{block}_conv{layer}.dw');c.weight.copy_(k[None,None].expand_as(c.weight));c.bias.zero_()
            c=net.conv(f'res{block}_conv{layer}.pw');c.weight.zero_();c.bias.zero_()
            if layer==1:c.weight[:,:,0,0].copy_(torch.eye(16,device='cuda'))
    c=net.conv('decode.pw');c.weight.zero_();c.bias.zero_()
    for j in range(8):c.weight[j,j,0,0]=1
    c=net.conv('mix.pw');c.weight.zero_();c.bias.zero_()
    c.weight[3,3,0,0]=-1;c.bias[3]=1
    c.weight[4,3,0,0]=1;c.bias[4]=-1
    c.weight[5,3,0,0]=1
    c=net.conv('finish.pw');c.weight.zero_();c.bias.zero_();c.weight[:,:,0,0].copy_(torch.eye(8,device='cuda'))
    c.bias[3]=-1-threshold;c.bias[4]=-1-threshold
    c=net.conv('finish.dw');c.weight.zero_();c.bias.zero_();c.weight[:,0,1,1]=1
    c=net.conv('rgb');c.weight.zero_();c.bias.fill_(-1)
    for j in range(3):
        c.weight[j,:3,0,0]=-lum;c.weight[j,j,0,0]+=1
        c.weight[j,3,0,0]=gain;c.weight[j,4,0,0]=-gain;c.weight[j,5,0,0]=1
    return net


def main():
    configure_workspace();contract_guard();set_reproducible(2026100423)
    out=ROOT/'runs'/RUN/'constructed';out.mkdir(exist_ok=False);m=manifest();records=[]
    for name,t,g in [('t06',.06,1.15),('t12',.12,1.15),('t20',.20,1.2)]:
        net=initialize(DetailRGBNet().cuda().eval(),t,g);folder=out/name;folder.mkdir()
        with torch.no_grad():
            for r in m['validation']:
                x=as_tensor(read_image(r['source']))[None].cuda();p=net(x)
                Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'))
            # Constants and large-step edges: avoid silently clipping color.
            x=torch.full((1,3,64,96),.2,device='cuda');constant=float((net(x)-x).abs().max());assert constant<1e-5
            x[:,:,:,48:]=.8;y=net(x);contrast=float(y[:,:,:,60:80].mean()-y[:,:,:,16:36].mean());assert contrast>.55
        torch.save(dict(model=net.state_dict(),threshold=t,gain=g,trained=False,graph='DetailRGBNet',
            inference_postprocess=False,code_sha256=file_hash(__file__)),folder/'initialized.pt')
        records.append(dict(name=name,threshold=t,gain=g,constant_error=constant,step_contrast=contrast))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for scene,key,xy in [('grass','val_stairs_090',(150,430)),('pillar','val_stairs_090',(424,320)),('people','val_plaza_090',(80,400))]:
        row=next(r for r in m['validation'] if r['id']==key);x,y=xy;board=Image.new('RGB',(1600,298),'#18202a');d=ImageDraw.Draw(board)
        paths=[row['source'],ROOT/'runs'/RUN/'detail_a'/(key+'.png')]+[out/n/(key+'.png') for n in ('t06','t12','t20')]
        for i,(p,title) in enumerate(zip(paths,['原片','区域目标','构造阈值0.06','构造阈值0.12','构造阈值0.20'])):
            d.text((320*i+5,5),title,font=font,fill='white');board.paste(Image.open(p).crop((x,y,x+320,y+264)),(320*i,30))
        board.save(out/(scene+'.png'))
    write_json(out/'checks.json',dict(status='CONSTRUCTED_NUMERIC_CHECKS_ONLY',records=records,trained=False))
    print('CONSTRUCTED_INITIALIZATIONS_DONE',flush=True)

if __name__=='__main__':main()
