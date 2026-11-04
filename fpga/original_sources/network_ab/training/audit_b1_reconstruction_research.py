"""Native-size evidence, phase tests and source-isolated final evaluation."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_reconstruction as r
from . import research_b1_reconstruction as s
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.inference_mode()
def phase_tests(net,rows):
    entries=[]
    for row in (rows[2],rows[4],rows[5]):
        x=r.tensor(r.read(row['path']))[None];y=net(x)[0]
        for shift in [(0,1),(1,0),(1,1),(2,2)]:
            z=net(torch.roll(x,shift,(-2,-1)))[0]
            aligned=torch.roll(z,tuple(-v for v in shift),(-2,-1))
            entries.append(dict(id=row['id'],shift=list(shift),
                translation_error=float((aligned-y)[...,80:-80,80:-80].abs().mean()*128)))
    return entries


@torch.inference_mode()
def frequencies(net):
    records=[]
    for period in (2,3,4,6,8,12,16,24,32):
        for phase in (0.,.7):
            theta=torch.arange(256,device='cuda')*(2*math.pi/period)+phase
            wave=.5*torch.cos(theta)
            x=wave[None,None,None].expand(1,3,256,256)
            p=net(x)[0]
            u=x[0,0,80:176,80:176];v=p[0].mean(0)[80:176,80:176]
            carrier=torch.exp(-1j*theta[80:176])[None]
            gain=float(((v-v.mean())*carrier).mean().abs()/(((u-u.mean())*carrier).mean().abs()+1e-9))
            records.append(dict(period=period,phase=phase,gain=gain,
                output_std=float(v.std()),source_std=float(u.std()),mae=float((p-x).abs()[...,80:176,80:176].mean()*128)))
    return records


@torch.inference_mode()
def activation_summary(net,rows):
    result=[]
    for row in (rows[2],rows[5]):
        nodes=net(r.tensor(r.read(row['path']))[None],all_nodes=True)
        for key in ['stem','down1.dw','down1.pw','down2.dw','down2.pw','res3_add','up2_low_pw']:
            z=nodes[key][0,:,10:-10:4,10:-10:4].flatten(1)
            centered=z-z.mean(1,keepdim=True);sv=torch.linalg.svdvals(centered)
            energy=sv.square().cumsum(0)/sv.square().sum().clamp_min(1e-12)
            result.append(dict(id=row['id'],node=key,zero_fraction=float((z==0).float().mean()),
                saturation_fraction=float((z.abs()>=127).float().mean()),
                rank99=int(torch.searchsorted(energy,.99))+1,channels=len(z),
                per_channel_std=z.std(1).cpu().tolist()))
    return result


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--test',action='store_true')
    ap.add_argument('--diagnostics',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(2026100559);contract_guard()
    _,val=r.data();dest=Path(a.out);dest.mkdir(parents=True,exist_ok=False)
    rows=val
    if a.test:
        rows=rows+json.loads((s.OUT/'holdout/manifest.json').read_text())['records']
    pictures={z['id']:[Image.fromarray(r.read(z['path']))] for z in val}
    results={};labels=['原图720p']
    for spec in a.models:
        label,path=spec.split('=',1);state=torch.load(path,map_location='cpu',weights_only=True)
        net=r.initialize();net.load_state_dict(state['model'])
        net.quantized=bool(state.get('qat',False) or state.get('args',{}).get('qat',False))
        e=s.evaluate(net,rows)
        item=dict(checkpoint=path,sha256=file_hash(path),step=state['step'],quantized=net.quantized,evaluation=e)
        if a.diagnostics:
            item['phase']=phase_tests(net,val);item['frequencies']=frequencies(net)
            item['activations']=activation_summary(net,val)
        results[label]=item;labels.append(label)
        for row in val:
            im=r.c.rgb_image(net(r.tensor(r.read(row['path']))[None])[0][0])
            im.save(dest/(label+'_'+row['id']+'.png'))
            pictures[row['id']].append(im)
        del net;torch.cuda.empty_cache()
        print('AUDIT_MODEL',label,e['pooled_psnr'],e['mean_ssim'],e['mean_edge'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    board=Image.new('RGB',(420*len(labels),314*len(r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,scene) in enumerate(r.c.CROPS):
        for j,(im,title) in enumerate(zip(pictures[key],labels)):
            crop=im.crop(box);assert crop.size==(420,280)
            board.paste(crop,(j*420,i*314+34))
            draw.text((j*420+5,i*314+6),scene+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'results.json',dict(models=results,contract_sha256=r.CONTRACT_SHA,
        test_included=a.test,original_native_crops=True,external_enhancement=False))


if __name__=='__main__':main()
