"""Carry both candidate RGB regions to720p before selecting the side.

Unlike old late-luminance decoding, the common RGB region decision itself is
delayed. All calculations are compiled into the unchanged24-convolution B1.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_quadrant_consensus import install as base_install
from .design_b1_edge_stopping_diffusion import clear,KERNEL,identity_dw
from .probe_b1_bounded_pair720 import install as decode,regression
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,cost_gain=4):
    base_install(net,2,1.5)
    identity_dw(net,'res3_conv1__dw',range(10))
    k='res3_conv1__pw';clear(net,k,.5)
    y=torch.tensor([.25,.5,.25])
    for side in range(2):
        for j in range(12):
            c=14*side+j;net.weight[k][c,3*side:3*side+3,0,0]=32*y;net.bias[k][c]=63.5-32*(9+10*j)
        for j,coef in ((12,torch.tensor([.5,0.,-.5])),(13,torch.tensor([-.25,.5,-.25]))):
            c=14*side+j;getattr(net,k+'_m')[:,c]=2048;net.weight[k][c,3*side:3*side+3,0,0]=coef;net.bias[k][c]=64
    net.weight[k][28,6,0,0]=cost_gain;net.weight[k][28,7,0,0]=-cost_gain;net.bias[k][28]=64
    identity_dw(net,'res3_conv2__dw',range(29))
    k='res3_conv2__pw';clear(net,k,1032/131072)
    for side in range(2):
        for c in range(3):
            out=3*side+c;net.weight[k][out,14*side:14*side+12,0,0]=10*1032/131072
        net.weight[k][3*side:3*side+3,14*side+12:14*side+14,0,0]=torch.tensor([[1.,-1.],[0.,1.],[-1.,-1.]])*127*1032/131072
        net.bias[k][3*side:3*side+3]=torch.tensor([4.,-60.,132.])
    net.res3_conv2__pw_m[:,6]=2048;net.weight[k][6,28,0,0]=1
    identity_dw(net,'up1_low_dw',range(7));clear(net,'up1_low_pw')
    for c in range(7):net.weight['up1_low_pw'][c,c,0,0]=1
    identity_dw(net,'up1_high_dw',range(7),KERNEL)
    decode(net,4)
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];assert (w/s).abs().max()<=127.001,(k,float((w/s).abs().max()))
        w.copy_(net.round(w/s)*s)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for gain in (2,4,8):
        mode=f'g{gain}';net=copy.deepcopy(base);install(net,gain);checks=regression(net,4);sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im
            records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('LATE_RGB_REGIONS',mode,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('g2','g4','g8')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
