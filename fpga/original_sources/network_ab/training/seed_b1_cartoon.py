"""Construct a realizable cartoon prior INSIDE the unmodified B1 weights.

Piecewise-linear five-level RGB surfaces and bounded luminance edges are
implemented by existing residual PW/ReLU weights. This is a transparent
hand-designed initialization, not a trained teacher or postprocessor.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch import nn
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .train_b1_cartoon_budget import RUN
from .train_b1_distillation import as_tensor,to_rgb,read_image


def seed(width=.28,ink=2.):
    net=build_fp32_model().eval()
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer,nn.Conv2d):
                layer.weight.zero_()
                if layer.bias is not None:layer.bias.zero_()
            elif isinstance(layer,nn.BatchNorm2d):
                layer.running_mean.zero_();layer.running_var.fill_(1-layer.eps);layer.num_batches_tracked.zero_();layer.weight.fill_(1);layer.bias.zero_()
        def op(name):
            seq=net.operators[net.names[name]]
            return next(m for m in seq if isinstance(m,nn.Conv2d)),next((m for m in seq if isinstance(m,nn.BatchNorm2d)),None)
        kernel=torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16
        c,b=op('stem')
        for i in range(3):c.weight[i,i,1,1]=1
        for i in (3,4):c.weight[i,:,1,1]=torch.tensor([.299,.587,.114])
        b.bias[:5]=1
        for name in ('down1.dw','down2.dw'):
            c,b=op(name);c.weight[:5,0]=kernel
        for name in ('down1.pw','down2.pw'):
            c,b=op(name)
            for i in range(5):c.weight[i,i,0,0]=1
        # Existing 32->32 residual block realizes a smoothed five-level surface.
        c,b=op('res1_conv1.dw');c.weight[:5,0,1,1]=1
        c,b=op('res1_conv1.pw')
        for channel in range(3):
            base=channel*9;c.weight[base,channel,0,0]=1
            for j,threshold in enumerate((.25,.75,1.25,1.75)):
                for side,sign in enumerate((-1,1)):
                    k=base+1+2*j+side;c.weight[k,channel,0,0]=1;b.bias[k]=-(threshold+sign*width/2)
        c,b=op('res1_conv2.dw');c.weight[:27,0,1,1]=1
        c,b=op('res1_conv2.pw')
        for channel in range(3):
            base=channel*9;c.weight[channel,base,0,0]=-1
            for j in range(4):
                c.weight[channel,base+1+2*j,0,0]=.5/width;c.weight[channel,base+2+2*j,0,0]=-.5/width
        # Original luminance channels3/4 survive the residual. Bound edge darkening.
        lap=torch.full((3,3),-1/8);lap[1,1]=1
        c,b=op('res2_conv1.dw');c.weight[3,0]=lap;c.weight[4,0]=-lap
        c,b=op('res2_conv1.pw');c.weight[0,3:5,0,0]=1;c.weight[1,3:5,0,0]=1;b.bias[1]=-.18
        c,b=op('res2_conv2.dw');c.weight[:2,0,1,1]=1
        c,b=op('res2_conv2.pw');c.weight[:3,0,0,0]=-ink;c.weight[:3,1,0,0]=ink
        # Residual3 is initially an exact identity; it remains available for learning.
        for name in ('up1_low_dw','up2_low_dw'):
            c,b=op(name);c.weight[:3,0,1,1]=1
        for name in ('up1_low_pw','up2_low_pw'):
            c,b=op(name)
            for i in range(3):c.weight[i,i,0,0]=1
        for name in ('up1_high_dw','up2_high_dw'):
            c,b=op(name);c.weight[:3,0]=kernel
        c,b=op('rgb_head')
        for i in range(3):c.weight[i,i,0,0]=1
        c.bias.fill_(-1)
    return net


def main():
    root=configure_workspace();contract_guard();torch.set_num_threads(2);run=root/'runs'/RUN
    out=run/'constructed_prior';out.mkdir(exist_ok=False)
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));cases=[('soft',.38,1.2),('medium',.28,2.),('bold',.18,3.)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);records=[]
    for name,width,ink in cases:
        folder=out/name;folder.mkdir();net=seed(width,ink)
        weight=folder/'initial_fp32.pt';torch.save(dict(model=net.state_dict(),hand_constructed_initialization=True,trained=False,width=width,ink=ink),weight)
        for row in manifest['validation']:
            with torch.inference_mode():rgb=to_rgb(net(as_tensor(read_image(row['source']))[None])[0])
            Image.fromarray(rgb).save(folder/(row['id']+'.png'))
        records.append(dict(case=name,width=width,ink=ink,sha256=file_hash(weight),parameters=sum(p.numel() for p in net.parameters())))
    canvas=Image.new('RGB',(1600,962),'#151c25');d=ImageDraw.Draw(canvas)
    for i,l in enumerate(['原片','教师','B1 动画初值 / 轻','B1 动画初值 / 中','B1 动画初值 / 强']):d.text((i*320+4,7),l,font=font,fill='white')
    for j,(key,xy) in enumerate([('val_stairs_090',(424,320)),('val_stairs_090',(150,430)),('val_plaza_090',(80,400))]):
        row=next(r for r in manifest['validation'] if r['id']==key);x,y=xy
        paths=[Path(row['source']),Path(row['target'])]+[out/name/(key+'.png') for name,_,_ in cases]
        for i,p in enumerate(paths):canvas.paste(Image.open(p).crop((x,y,x+320,y+290)),(i*320,38+j*297))
    d.text((6,932),'仅作未训练初始化对照：所有变化来自B1真实权重，未经过输出滤镜。',font=font,fill='#a9cce9');canvas.save(out/'comparison.png')
    write_json(out/'manifest.json',dict(cases=records,trained=False,inference_ops_added=False,hardware_files_changed=False,
        note='Explicit realizable prior. Not a style acceptance or trained result.'))
    print('CONSTRUCTED_B1_INITIALIZATIONS',records,flush=True)


if __name__=='__main__':main()
