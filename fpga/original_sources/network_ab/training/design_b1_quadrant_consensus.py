"""Variance-margin consensus: select one RGB or their shared mean on ties.

No fractional binary mask is used. Three exact hard cases select A, B, or
the common RGB average, avoiding the black holes of a soft clipping mask.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_quadrant_geometry720 import install as base_install
from .design_b1_quadrant_variance import hard
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,margin=4,gain=1.5):
    base_install(net,'late720')
    k='res1_conv2__pw'
    difference=net.weight[k][16:18].clone()
    net.weight[k][16:].zero_();net.bias[k][16:].zero_()
    for pair in range(2):
        # A when VB-VA>margin, B when VA-VB>margin, else common mean.
        net.weight[k][16+2*pair]=-difference[pair];net.bias[k][16+2*pair]=64-margin
        net.weight[k][17+2*pair]=difference[pair];net.bias[k][17+2*pair]=64-margin
    identity_dw(net,'res2_conv1__dw',range(20));hard(net,'res2_conv1__dw',range(16,20))
    k='res2_conv1__pw';clear(net,k)
    for pair in range(2):
        ga,gb=16+2*pair,17+2*pair
        for channel in range(4):
            ia=6*pair+channel if channel<3 else 12+2*pair
            ib=ia+3 if channel<3 else ia+1
            out=12*pair+3*channel
            net.weight[k][out,ia,0,0]=1;net.weight[k][out,ga,0,0]=1;net.bias[k][out]=-127
            net.weight[k][out+1,ib,0,0]=1;net.weight[k][out+1,gb,0,0]=1;net.bias[k][out+1]=-127
            net.weight[k][out+2,ia,0,0]=.5;net.weight[k][out+2,ib,0,0]=.5
            net.weight[k][out+2,ga,0,0]=-1;net.weight[k][out+2,gb,0,0]=-1
    identity_dw(net,'res2_conv2__dw',range(24));k='res2_conv2__pw';clear(net,k)
    for pair in range(2):
        for c in range(3):net.weight[k][3*pair+c,12*pair+3*c:12*pair+3*c+3,0,0]=1
        net.weight[k][6+pair,12*pair+9:12*pair+12,0,0]=1
    net.weight[k][8,21:24,0,0]=1;net.weight[k][8,9:12,0,0]=-1;net.bias[k][8]=64-margin
    net.weight[k][9,9:12,0,0]=1;net.weight[k][9,21:24,0,0]=-1;net.bias[k][9]=64-margin
    identity_dw(net,'res3_conv1__dw',range(10));hard(net,'res3_conv1__dw',(8,9))
    k='res3_conv1__pw';clear(net,k)
    for c in range(3):
        net.weight[k][3*c,c,0,0]=1;net.weight[k][3*c,8,0,0]=1;net.bias[k][3*c]=-127
        net.weight[k][3*c+1,c+3,0,0]=1;net.weight[k][3*c+1,9,0,0]=1;net.bias[k][3*c+1]=-127
        net.weight[k][3*c+2,c,0,0]=.5;net.weight[k][3*c+2,c+3,0,0]=.5
        net.weight[k][3*c+2,8:10,0,0]=-1
    identity_dw(net,'res3_conv2__dw',range(9));k='res3_conv2__pw';clear(net,k)
    for c in range(3):net.weight[k][c,3*c:3*c+3,0,0]=1
    for j in range(6):net.weight['up2_high_dw'][j,0]=KERNEL*gain;net.bias['up2_high_dw'][j]=64*(1-gain)


def decide(a,b,va,vb,margin):
    average=m.t.r.c.Round()((a+b)/2)
    return torch.where((vb-va>margin),a,torch.where(va-vb>margin,b,average))

@torch.no_grad()
def regression(net,margin):
    torch.manual_seed(820);rgb=torch.randint(0,128,(64,4,3,7,8)).float();var=torch.randint(0,40,(64,4,1,7,8)).float()
    z=torch.zeros(64,32,7,8);z[:,:12]=rgb.flatten(1,2);z[:,12:16]=var.squeeze(2)
    for p in range(2):z[:,16+2*p]=64-margin+(var[:,2*p+1]-var[:,2*p]).squeeze(1);z[:,17+2*p]=64-margin+(var[:,2*p]-var[:,2*p+1]).squeeze(1)
    got=fragment(net,z,'res1_add')['res3_add'][:,:3]
    ra=decide(rgb[:,0],rgb[:,1],var[:,0],var[:,1],margin);rb=decide(rgb[:,2],rgb[:,3],var[:,2],var[:,3],margin)
    va=decide(var[:,0],var[:,1],var[:,0],var[:,1],margin);vb=decide(var[:,2],var[:,3],var[:,2],var[:,3],margin)
    want=decide(ra,rb,va,vb,margin);assert torch.equal(got,want),float((got-want).abs().max())
    assert ((got>=rgb.amin(1))&(got<=rgb.amax(1))).all()
    return dict(actual_three_case_selector_error=0,component_extrema_error=0,rounded_common_rgb_means=True)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[]
    for margin in (2,4,8):
        net=copy.deepcopy(base);install(net,margin);checks=regression(net,margin);name=f'm{margin}';sub=out/name;sub.mkdir();raw=net.export();assert len(raw)==20480
        (sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[name,row['id']]=im
            records.append(dict(mode=name,id=row['id'],all_nodes_error=0))
        torch.save(dict(model=net.state_dict(),residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,visual_acceptance=False,extra_inference_ops=0),sub/'designed.pt')
        print('QUADRANT_CONSENSUS',name,checks,flush=True)
    rows={r['id']:r for r in data['validation']};board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(rows[rid]['target']))]+[(k,images[k,rid]) for k in ('m2','m4','m8')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),checks=checks,records=records,source_sha256=file_hash(__file__),visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
