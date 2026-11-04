"""Keep quadrant costs but take colour from its interior, not blended centre.

The extra12 existing down2 channels carry whole RGB corner samples. Jensen
cost uses the same moment/mean fields, stored with a bounded scale. This is
a controlled coefficient design, not a learned semantic interior predictor.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_joint_colour_levels import install as levels
from .design_b1_quadrant_consensus import regression
from .design_b1_edge_stopping_diffusion import clear
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,mixture=0):
    levels(net,2);q=net.weight['down1__pw'][3,:7,0,0].clone();y=torch.tensor([.25,.5,.25])
    for group,(dy,dx) in enumerate(((0,0),(0,2),(2,0),(2,2))):
        for c in range(3):
            net.weight['down2__pw'][16+3*group+c,c,0,0]=1
            net.weight['res1_conv1__dw'][16+3*group+c,0,dy,dx]=1-mixture
            net.weight['res1_conv1__dw'][16+3*group+c,0,1,1]+=mixture
    k='res1_conv1__pw';clear(net,k)
    for group in range(4):
        for c in range(3):net.weight[k][3*group+c,16+3*group+c,0,0]=1
        net.res1_conv1__pw_m[:,12+group]=512
        net.weight[k][12+group,4*group+3,0,0]=.25
        net.weight[k][12+group,4*group:4*group+3,0,0]=-.25*q[:3]
        for j,t in enumerate((25,51,76,102)):
            net.weight[k][16+4*group+j,4*group:4*group+3,0,0]=y;net.bias[k][16+4*group+j]=-t
    cost=torch.zeros(4,32)
    for group in range(4):cost[group,12+group]=1;cost[group,16+4*group:20+4*group]=-.25*q[3:]
    k='res1_conv2__pw';clear(net,k)
    for j in range(12):net.weight[k][j,j,0,0]=1
    net.weight[k][12:16,:,0,0]=cost;net.bias[k][12:16]=-24
    for pair in range(2):
        d=cost[2*pair+1]-cost[2*pair]
        net.weight[k][16+2*pair,:,0,0]=d;net.bias[k][16+2*pair]=63
        net.weight[k][17+2*pair,:,0,0]=-d;net.bias[k][17+2*pair]=63
    net.bias['res2_conv2__pw'][8:10]=63
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];assert (w/s).abs().max()<=127.001
        w.copy_(net.round(w/s)*s)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for mode,mix in (('mean',None),('corner',0),('half_corner',.5)):
        net=copy.deepcopy(base)
        if mix is None:levels(net,2)
        else:install(net,mix)
        checks=regression(net,8 if mix is None else 1);sub=out/mode;sub.mkdir();raw=net.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im;records.append(dict(mode=mode,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt');print('INTERIOR_REGION_COLOUR',mode,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('mean','corner','half_corner')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
