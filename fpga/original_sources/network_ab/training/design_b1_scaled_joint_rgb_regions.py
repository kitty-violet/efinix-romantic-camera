"""Joint-RGB diffusion in a lower-precision colour domain, fixed B1 only.

Existing integer T8 common conductance is reused with explicitly scaled RGB.
The output is expanded BEFORE spatial reconstruction, so interpolation has
ordinary7-bit precision. No independent luminance/chroma decisions remain.
This is a coefficient design experiment, not an accepted learned model.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_joint_rgb_diffusion import install as base_install,regression
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,compression=4):
    base_install(net,8)
    clear(net,'stem')
    for c in range(3):net.weight['stem'][c,c,1,1]=.5/compression;net.bias['stem'][c]=64/compression
    # The common RGB diffusion part itself is unchanged and independently
    # tested; its colour-domain threshold scales with the stated precision.
    clear(net,'up1_low_pw',1/16)
    for c in range(3):net.weight['up1_low_pw'][c,24+c,0,0]=compression
    kernel=KERNEL*.25;kernel[1,1]+=.75
    identity_dw(net,'up1_high_dw',range(3),kernel)
    identity_dw(net,'up2_low_dw',range(3))
    clear(net,'up2_low_pw')
    for c in range(3):net.weight['up2_low_pw'][c,c,0,0]=1
    identity_dw(net,'up2_high_dw',range(3),KERNEL)
    clear(net,'rgb_head',1/32)
    luma=torch.tensor([.25,.5,.25]);matrix=1.08*(1.15*torch.eye(3)-.15*luma[None])
    net.weight['rgb_head'][:,:3,0,0]=matrix*2
    net.bias['rgb_head'].copy_(torch.tensor([-140.,-138.,-136.]))
    # Quantize the intended affine colour transform to the existing lattice.
    net.weight['rgb_head'].copy_(net.round(net.weight['rgb_head']*32)/32)
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert torch.equal(w,net.round(w/s)*s) and (w/s).abs().max()<=127,(k,'lattice')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for compression in (2,4):
        net=copy.deepcopy(base);install(net,compression);checks=regression(net,8)
        name='rgb'+str(128//compression);sub=out/name;sub.mkdir()
        raw=net.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw)
        rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[name,row['id']]=im
            records.append(dict(variant=name,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
            contract_sha256=state['contract_sha256'],step=0,qat=False,designed_not_trained=True,
            visual_acceptance=False,colour_domain_compression=compression,extra_inference_ops=0),sub/'designed.pt')
        print('SCALED_RGB_DESIGN',name,checks,flush=True)
    rows={r['id']:r for r in data['validation']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('原片',Image.open(rows[rid]['path'])),('固定目标',Image.open(rows[rid]['target']))]+[(key,images[key,rid]) for key in ('rgb64','rgb32')]
        for j,(title,im) in enumerate(ims):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,code_sha256=file_hash(__file__),
        designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0))


if __name__=='__main__':main()
