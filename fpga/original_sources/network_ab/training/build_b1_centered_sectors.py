"""Contour-preserving sector selection: texture cost plus centre agreement.

The cost penalizes selecting a smooth region with a different centre colour.
It is a fixed local method expressed solely by original B1 coefficients.
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import install as install_palette,GROUPS
from .build_b1_median_cartoon import LEVELS
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash


@torch.no_grad()
def build(net,penalty,head):
    for k,w in net.weight.items():w.zero_();net.bias[k].zero_()
    install_palette(net,LEVELS)
    ker=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    lap=ker.new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])/4;luma=ker.new_tensor([.25,.5,.25])
    for c in range(3):net.weight['stem'][c,c,1,1]=.5
    net.bias['stem'][:3]=64;net.weight['stem'][3]=lap[None]*luma[:,None,None];net.weight['stem'][4]=-net.weight['stem'][3]
    net.weight['down1__dw'][:5,0]=ker
    for s in range(4):
        for c in range(3):net.weight['down1__pw'][s*3+c,c,0,0]=1
        q=ker.new_zeros(3,3);y,x=s//2,s%2;q[y:y+2,x:x+2]=.25
        net.weight['down2__dw'][3*s:3*s+3,0]=q;net.weight['res1_conv1__dw'][3*s:3*s+3,0]=q
        net.weight['down2__pw'][15+s,15,0,0]=1;net.weight['res1_conv1__dw'][15+s,0]=q
    for c in range(3):net.weight['down1__pw'][12+c,c,0,0]=1
    net.weight['down1__pw'][15,3:5,0,0]=1
    net.weight['down2__dw'][12:15,0,1,1]=1;net.weight['down2__dw'][15,0]=ker
    for c in range(15):net.weight['down2__pw'][c,c,0,0]=1
    net.weight['res1_conv1__dw'][12:15,0,1,1]=1
    k='res1_conv1__pw'
    for c in range(12):net.weight[k][c,c,0,0]=1
    for s in range(4):
        net.weight[k][12+s,15+s,0,0]=1
        net.weight[k][16+s,3*s:3*s+3,0,0]=luma;net.weight[k][16+s,12:15,0,0]=-luma
        net.weight[k][20+s,3*s:3*s+3,0,0]=-luma;net.weight[k][20+s,12:15,0,0]=luma
    net.weight['res1_conv2__dw'][:24,0,1,1]=1;k='res1_conv2__pw'
    for c in range(12):net.weight[k][c,c,0,0]=1
    for s in range(4):
        net.weight[k][12+s,12+s,0,0]=1;net.weight[k][12+s,16+s,0,0]=penalty;net.weight[k][12+s,20+s,0,0]=penalty
    net.weight['res2_conv1__dw'][:16,0,1,1]=1;k='res2_conv1__pw'
    for c in range(12):net.weight[k][c,c,0,0]=1
    for p,(a,b) in enumerate(((12,13),(14,15))):
        net.weight[k][12+p,b,0,0]=127;net.weight[k][12+p,a,0,0]=-127;net.bias[k][12+p]=127
        net.weight[k][14+2*p,b,0,0]=1;net.weight[k][14+2*p,a,0,0]=-1;net.weight[k][15+2*p,b,0,0]=1
    net.weight['res2_conv2__dw'][:18,0,1,1]=1
    for j in (12,13):net.weight['res2_conv2__dw'][j,0]=1;net.bias['res2_conv2__dw'][j]=-4*127
    k='res2_conv2__pw'
    for p in range(2):
        for c in range(3):
            j=p*6+c;net.weight[k][j,j,0,0]=1;net.weight[k][j,12+p,0,0]=1;net.bias[k][j]=-127
            net.weight[k][j+3,j+3,0,0]=1;net.weight[k][j+3,12+p,0,0]=-1
        net.weight[k][12+p,15+2*p,0,0]=1;net.weight[k][12+p,14+2*p,0,0]=-1
    net.weight['res3_conv1__dw'][:14,0,1,1]=1;k='res3_conv1__pw'
    for p in range(2):
        for c in range(3):net.weight[k][p*3+c,p*6+c,0,0]=1;net.weight[k][p*3+c,p*6+c+3,0,0]=1
    net.weight[k][6,13,0,0]=127;net.weight[k][6,12,0,0]=-127;net.bias[k][6]=127
    net.weight['res3_conv2__dw'][:7,0,1,1]=1;net.weight['res3_conv2__dw'][6,0]=1;net.bias['res3_conv2__dw'][6]=-4*127
    k='res3_conv2__pw'
    for c in range(3):
        net.weight[k][c,c,0,0]=1;net.weight[k][c,6,0,0]=1;net.bias[k][c]=-127
        net.weight[k][c+3,c+3,0,0]=1;net.weight[k][c+3,6,0,0]=-1
    net.weight['up1_low_dw'][:6,0,1,1]=1;k='up1_low_pw';net.weight[k].zero_();net.bias[k].zero_()
    matrix=np.array([[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]])
    for g,(start,end) in enumerate(GROUPS):
        v=np.array(LEVELS[g],float);cuts=(v[:-1]+v[1:])/2
        raw=((cuts+8)/1.06/2 if g==0 else (cuts+8)/1.20/2 if g==1 else cuts/1.15/2)
        for j,cut in enumerate(raw):
            net.weight[k][start+j,:3,0,0]=net.weight[k].new_tensor(matrix[g]);net.weight[k][start+j,3:6,0,0]=net.weight[k].new_tensor(matrix[g]);net.bias[k][start+j]=64-cut
    for b in range(1,4):net.package['residual_encoding'][f'res{b}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    preserved={'up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'}
    for k,w in net.weight.items():
        if k in preserved:continue
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    net.weight['rgb_head'].copy_(head['weight.rgb_head']);net.bias['rgb_head'].copy_(head['bias.rgb_head']);net.rgb_head_m.copy_(head['rgb_head_m'])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);head=torch.load(a.warm,map_location='cpu',weights_only=True)['model']
    rows=m.load_data()['validation'];results=[]
    for penalty in (0,.5,2):
        net=m.t.r.initialize();net.quantized=True;build(net,penalty,head);sub=dest/str(penalty).replace('.','p');sub.mkdir()
        for row in rows:
            n=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True);candidates=n['res1_conv1.dw'][:,:12].reshape(1,4,3,180,320)
            picked=n['up1_low_dw'][:,:3]+n['up1_low_dw'][:,3:6]
            assert (candidates==picked[:,None]).all(2).any(1).all(),'Colour mixing/hole in selector'
        export(net,rows,sub/'package');m.t.preview(net,rows,sub/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            ordered_regions=True,centered_sectors=True,centre_penalty=penalty,trained=False),sub/'initial.pt')
        results.append(dict(penalty=penalty,one_candidate_selection=True));print('CENTERED_SECTOR_COMPLETE',penalty,flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name);write_json(dest/'review.json',dict(results=results,head_parent_sha256=file_hash(a.warm),
        added_operators=0,postfilter=False,minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
