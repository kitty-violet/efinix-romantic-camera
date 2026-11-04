"""Preserve narrow foregrounds by including centre fidelity in window cost.

Two PWL moment hinges free eight existing rows for four exact absolute
mean-minus-centre luminance differences. No semantic or external filter.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_quadrant_consensus import install as base_install,regression
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,fidelity=1.,margin=2):
    base_install(net,margin,1.5);y=torch.tensor([.25,.5,.25])
    net.weight['stem'][3:].zero_();net.bias['stem'][3:].zero_()
    for j,t in enumerate((42,85)):net.weight['stem'][3+j,:,1,1]=y*.5;net.bias['stem'][3+j]=64-t
    q=torch.cat((3*(42/128-1)*y,torch.tensor([255.,258.])/128))
    q=net.round(q*32)/32
    clear(net,'down1__pw',1/32)
    for group in range(4):
        for c in range(3):net.weight['down1__pw'][4*group+c,c,0,0]=1
        net.weight['down1__pw'][4*group+3,:5,0,0]=q;net.bias['down1__pw'][4*group+3]=96
    for c in range(3):net.weight['down2__pw'][16+c,c,0,0]=1
    for c in range(16,19):net.weight['res1_conv1__dw'][c,0,1,1]=1
    clear(net,'res1_conv1__pw')
    for group in range(4):
        for c in range(3):net.weight['res1_conv1__pw'][3*group+c,4*group+c,0,0]=1
        net.weight['res1_conv1__pw'][12+group,4*group+3,0,0]=1
        for j,t in enumerate((42,85)):
            net.weight['res1_conv1__pw'][16+2*group+j,4*group:4*group+3,0,0]=y;net.bias['res1_conv1__pw'][16+2*group+j]=-t
        for side,sgn in enumerate((1.,-1.)):
            net.weight['res1_conv1__pw'][24+2*group+side,4*group:4*group+3,0,0]=sgn*y
            net.weight['res1_conv1__pw'][24+2*group+side,16:19,0,0]=-sgn*y
    cost=torch.zeros(4,32)
    for group in range(4):
        cost[group,12+group]=.5;cost[group,3*group:3*group+3]=-.5*q[:3]
        cost[group,16+2*group:18+2*group]=-.5*q[3:]
        cost[group,24+2*group:26+2*group]=.5*fidelity
    k='res1_conv2__pw';clear(net,k,1/64)
    for j in range(12):net.weight[k][j,j,0,0]=1
    net.weight[k][12:16,:,0,0]=cost;net.bias[k][12:16]=-48
    for pair in range(2):
        d=cost[2*pair+1]-cost[2*pair]
        net.weight[k][16+2*pair,:,0,0]=d;net.bias[k][16+2*pair]=64-margin
        net.weight[k][17+2*pair,:,0,0]=-d;net.bias[k][17+2*pair]=64-margin
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];assert (w/s).abs().max()<=127.001,(k,float((w/s).abs().max()))
        w.copy_(net.round(w/s)*s)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for fidelity in (.5,1.,2.):
        net=copy.deepcopy(base);install(net,fidelity);checks=regression(net,2);name=f'f{fidelity:g}';sub=out/name;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            # Explicitly verify that the added absolute centre deviations are
            # the same values as the equivalent integer pointwise operation.
            f=nodes['res1_conv1.dw'];g=nodes['res1_conv1.pw']
            for group in range(4):
                d=((f[:,4*group:4*group+3]-f[:,16:19])*torch.tensor([.25,.5,.25])[None,:,None,None]).sum(1)
                assert torch.equal(g[:,24+2*group],net.round(d).clamp(0,127))
                assert torch.equal(g[:,25+2*group],net.round(-d).clamp(0,127))
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[name,row['id']]=im
            records.append(dict(mode=name,id=row['id'],all_nodes_error=0,centre_absolute_difference_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('CENTERED_QUADRANT',name,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('f0.5','f1','f2')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
