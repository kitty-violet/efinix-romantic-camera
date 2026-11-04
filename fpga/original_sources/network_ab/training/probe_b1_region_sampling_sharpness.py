"""Isolate upstream averaging from late reconstruction in fixed B1.

Positive normalized kernels only: no unsharp overshoot or external filter.
The common RGB selector is unchanged; these are designed coefficients.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_centered_quadrant import install as base_install
from .design_b1_quadrant_consensus import regression
from .design_b1_edge_stopping_diffusion import KERNEL
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,mode):
    base_install(net,2,2)
    centre=torch.zeros(3,3);centre[1,1]=1
    if mode=='reduced_prefilter':
        for k in ('down1__dw','down2__dw'):
            net.weight[k].mul_(.5)
            for c in range(5 if k=='down1__dw' else 16):net.weight[k][c,0,1,1]+=.5
    if mode in ('centre_windows','reduced_prefilter'):
        for c in range(16):net.weight['res1_conv1__dw'][c].mul_(.5);net.weight['res1_conv1__dw'][c,0,1,1]+=.5
    # All real coefficients are re-quantized; no float-only preview.
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert (w/s).abs().max()<=127.001
        w.copy_(net.round(w/s)*s)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for mode in ('base','centre_windows','reduced_prefilter'):
        net=copy.deepcopy(base);install(net,mode);checks=regression(net,2);sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im
            records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('SAMPLING_SHARPNESS',mode,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('base','centre_windows','reduced_prefilter')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
