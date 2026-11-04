"""Target-only check of a shared luminance decoder with final720 colour levels.

Controlled reuse of the old19-level idea: six bands share ONE luminance field;
chroma has9x9 states, and all final decisions use the last existing720 DW.
This removes independent luminance votes, not the hardware sampling stages.
No source-prediction claim; the target is injected at the180p interface.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import joint_region_loss
from .audit_b1_white_rims_grid import measures
from .runtime import write_json,file_hash


def coordinates(rgb):
    r,g,b=rgb.unbind(1);y=(r+2*g+b)/4;co=(r-b)/2;cg=(-r+2*g-b)/4
    return torch.stack(((y-8)*127/234,(co+64)*127/128,(cg+48)*127/96),1).clamp(0,127)


@torch.no_grad()
def install(net):
    tail=('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
    for k in tail:net.weight[k].zero_();net.bias[k].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    net.up1_high_dw_m.fill_(512);net.weight['up1_high_dw'][:3,0]=kernel/16
    net.up2_low_dw_m.fill_(2048);net.weight['up2_low_dw'][:3,0,1,1]=1
    net.up2_low_pw_m.fill_(8192)
    for j in range(6):net.weight['up2_low_pw'][j,0,0,0]=6;net.bias['up2_low_pw'][j]=-127*j
    net.weight['up2_low_pw'][6,1,0,0]=1;net.weight['up2_low_pw'][7,2,0,0]=1
    for j in range(8):
        states=3 if j<6 else 8;mult=round(131072*states/(127*16))
        net.up2_high_dw_m[:,j]=mult;net.weight['up2_high_dw'][j,0]=kernel*mult/131072
    net.rgb_head_m.fill_(32768);net.weight['rgb_head'][:,:6,0,0]=13
    net.weight['rgb_head'][:,6,0,0]=torch.tensor([16,0,-16.])
    net.weight['rgb_head'][:,7,0,0]=torch.tensor([-12,12,-12.])
    net.bias['rgb_head'].copy_(torch.tensor([-136,-168,-8.]))


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);install(net)
    codes=torch.cartesian_prod(torch.arange(19),torch.arange(9),torch.arange(9)).float()
    z=F.pad(net.round(codes*torch.tensor([127/18,127/8,127/8]))[:,:,None,None],(0,0,0,0,0,13))
    got=fragment(net,z,'up1_low_pw')['rgb_head'][:,:,0,0]+128
    yy,co,cg=codes.unbind(1);y=8+13*yy;co=16*co-64;cg=12*cg-48
    expected=torch.stack((y+co-cg,y+cg,y-co-cg),1).clamp(0,255)
    assert torch.equal(got,expected),float((got-expected).abs().max())
    data=m.load_data(a.data);pictures={};records=[]
    for row in data['validation']:
        src=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()
        target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()
        # Native coverage through the real180p sampling interface. This is a
        # label injection, never a source image passed through a trained net.
        fields=F.pad(net.round(F.avg_pool2d(coordinates(target),4)),(0,0,0,0,0,13))
        p=fragment(net,fields,'up1_low_pw')['rgb_head']/128
        lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        _,v=joint_region_loss(p,target/128-1,lab);r,_=measures(p*128+128,target,src)
        pictures[row['id']]=m.t.r.c.rgb_image(p[0]);pictures[row['id']].save(dest/(row['id']+'_target_injection.png'))
        records.append(dict(id=row['id'],**{k:float(z) for k,z in v.items()},**r))
        print('SHARED_LUMA720_TARGET_ONLY',row['id'],r['bright_target_excess_fraction'],float(v['sharp_boundary']),flush=True)
    # The full graph is also byte-consistent, but its previous readout has NOT
    # been fitted for these new coordinates and must not be called a stylizer.
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
    x=torch.randint(-128,128,(1,3,64,80)).float();na=net(x/128,all_nodes=True);nb,_=actual(x,True)
    assert all(torch.equal(v,nb[k]) for k,v in na.items())
    byid={r['id']:r for r in data['validation']};board=Image.new('RGB',(840,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,im) in enumerate([('区域目标',Image.open(byid[rid]['target'])),('已知目标注入·不是源预测',pictures[rid])]):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png')
    state.update(model=net.state_dict(),ternary_packed720=False,shared_luma720=True,
        source_readout_not_fitted=True,visual_acceptance=False,step=0)
    torch.save(state,dest/'decoder_template_not_trained.pt')
    write_json(dest/'review.json',dict(checkpoint_sha256=file_hash(a.checkpoint),records=records,
        constant_colours=1539,constant_colour_error=0,byte_node_error=0,byte_size=len(net.export()),
        source_prediction=False,target_injection=True,accepted=False))


if __name__=='__main__':main()
