"""An exact8-way winner built only from B1's existing PW/ReLU/residuals.

Scores are integer0..15, encoded as8*score+priority. Three compare/reduce
stages compute their max. An existing clipped PW selects exactly one colour
direction; final INT8 levels simplify brightness. No runtime argmax or LUT.
This is an engineering probe, not an accepted style or minimum-area proof.
"""
import argparse
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .seed_b1_binary_regions import build as base
from .runtime import configure_workspace,write_json
from .engine import set_reproducible
from .publish_b1_style_recovery import export


@torch.no_grad()
def build(states=16,aa=.5,shape=False,soft_gate=False,bounded_scores=False):
    net=base()
    for key in net.weight:net.weight[key].zero_();net.bias[key].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    def identity(key):net.weight[key][:,0,1,1]=1
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    for c in range(3):net.weight['down1__dw'][c,0]=kernel
    centres=kernel.new_tensor([[0,0],[-7,-3],[3,20],[15,2],[24,-12],[-24,-10],[10,-22],[16,16]])
    chroma=kernel.new_tensor([[.5,0,-.5],[-.25,.5,-.25]])
    score_scale=.003 if shape else .006
    net.weight['down1__pw'][:8,:3,0,0]=score_scale*centres@chroma
    net.bias['down1__pw'][:8]=8-score_scale*.5*centres.square().sum(1)
    net.weight['down1__pw'][8,:3,0,0]=kernel.new_tensor([.25,.5,.25])
    for c in range(9):net.weight['down2__dw'][c,0]=kernel
    for c in range(8):
        for j in (c,c+8):net.weight['down2__pw'][j,c,0,0]=8;net.bias['down2__pw'][j]=c
    net.weight['down2__pw'][16,8,0,0]=1
    for stage in (1,2,3):
        net.package['residual_encoding'][f'res{stage}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
        identity(f'res{stage}_conv1__dw');identity(f'res{stage}_conv2__dw')
        w=net.weight[f'res{stage}_conv1__pw'];v=net.weight[f'res{stage}_conv2__pw']
        for j in range(8):
            partner=j^(1<<(stage-1))
            w[j,j,0,0]=1;w[j,partner,0,0]=-1
            w[j+8,partner,0,0]=1;w[j+16,j+8,0,0]=1
            v[j,j,0,0]=1;v[j,j+8,0,0]=1;v[j+8,j+16,0,0]=1
        w[24,16,0,0]=1;v[16,24,0,0]=1
    identity('up1_low_dw')
    for j in range(8):
        for k in (j,j+8):
            net.weight['up1_low_pw'][k,j+8,0,0]=127
            net.weight['up1_low_pw'][k,0,0,0]=-127
        net.weight['up1_low_pw'][j,16,0,0]=1
        net.bias['up1_low_pw'][j+8]=127
    identity('up1_high_dw');identity('up2_low_dw')
    for j in range(8):net.weight['up2_low_pw'][j,j,0,0]=1
    for j in range(8):
        net.weight['up2_high_dw'][j,0]=kernel*aa*states/127
        net.weight['up2_high_dw'][j,0,1,1]+=(1-aa)*states/127
    if shape:
        # Average raw scores on the180p grid, re-encode priority BEFORE max.
        # Dividing an encoded score without removing priority would corrupt
        # the exact winner guarantee at quantization ties.
        k='res1_conv1__dw';net.weight[k].zero_();net.bias[k].zero_()
        for j in range(8):
            for c in (j,j+8):net.weight[k][c,0]=kernel/8;net.bias[k][c]=-j/8
        net.weight[k][16,0]=kernel
        k='res1_conv1__pw';net.weight[k].zero_();net.bias[k].zero_()
        for j in range(8):
            q=j^1
            net.weight[k][j,j,0,0]=8;net.weight[k][j,q,0,0]=-8;net.bias[k][j]=j-q
            net.weight[k][j+8,q,0,0]=8;net.bias[k][j+8]=q
            net.weight[k][j+16,j+8,0,0]=8;net.bias[k][j+16]=j
        net.weight[k][24,16,0,0]=1
        for key in ('up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw'):
            net.weight[key].zero_();net.bias[key].zero_()
        for j in range(8):
            # A binary mask is interpolated and sharpened at360p before it
            # gates luminance. Partial coverage may darken a boundary; it
            # cannot add two full colour directions into a white halo.
            net.weight['up1_low_pw'][j,j+8,0,0]=127
            net.weight['up1_low_pw'][j,0,0,0]=-127
            net.bias['up1_low_pw'][j]=127
            net.weight['up1_low_pw'][j+8,16,0,0]=1
            net.weight['up1_high_dw'][j,0]=kernel
            net.weight['up1_high_dw'][j+8,0]=kernel
            net.weight['up2_low_dw'][j,0]=3*kernel;net.bias['up2_low_dw'][j]=-127
            net.weight['up2_low_dw'][j+8,0,1,1]=1
            net.weight['up2_low_pw'][j,j,0,0]=1
            net.weight['up2_low_pw'][j,j+8,0,0]=1
            net.bias['up2_low_pw'][j]=-127
        if soft_gate:
            for j in range(8):
                net.weight['up1_low_pw'][j+8].copy_(net.weight['up1_low_pw'][j])
                net.weight['up1_low_pw'][j+8,16,0,0]=1
                net.bias['up1_low_pw'][j+8]=0
                net.weight['up2_low_pw'][j].zero_()
                net.weight['up2_low_pw'][j,j,0,0]=.25*64/127
                net.weight['up2_low_pw'][j,j+8,0,0]=.75
                net.bias['up2_low_pw'][j]=0
    colours=kernel.new_tensor([[235,239,250],[165,193,232],[156,220,88],[249,220,177],
                              [250,130,133],[113,165,254],[221,154,247],[228,241,96]])
    net.weight['rgb_head'][:,:,0,0]=colours.T/states
    net.bias['rgb_head'].fill_(-128)
    if bounded_scores:
        # Down1's ordinary saturation bounds scores to127; a positive
        # normalized down2 kernel divides them by8.5, giving0..15 after
        # rounding for EVERY input, including after encoder training.
        net.weight['down1__pw'][:8]*=8.5;net.bias['down1__pw'][:8]*=8.5
        net.weight['down2__dw'][:8]/=8.5
    for key,w in net.weight.items():
        scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--shape',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(20261006283);torch.set_num_threads(2)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False);net=build(states=12 if a.shape else 16,shape=a.shape);d=m.load_data();checks=[]
    for row in d['validation']:
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
        score=nodes['down2.dw'][:,:8]
        assert score.min()>=0 and score.max()<=15,('score_range',float(score.min()),float(score.max()))
        enc=nodes['down2.pw'][:,:8];expected=score*8+torch.arange(8,device='cuda')[None,:,None,None]
        assert torch.equal(enc,expected),'score embedding'
        if a.shape:expected=nodes['res3_add'][:,8:16]
        mx=nodes['res3_add'][:,:8];assert torch.equal(mx,expected.max(1,keepdim=True).values.expand_as(mx)),'exact max'
        masks=nodes['up1_low_pw'][:,:8] if a.shape else nodes['up1_low_pw'][:,8:]
        assert torch.equal(masks.sum(1),torch.full_like(masks[:,0],127)),'one winner'
        assert set(masks.unique().tolist())<={0.,127.}
        checks.append(dict(id=row['id'],one_winner=True,score_min=float(score.min()),score_max=float(score.max())))
    export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
                    coverage_layout='8exclusive_winner_rays',designed_only=True,training_steps=0),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(checks=checks,postfilter=False,added_operators=0,minimum_area_guaranteed=False,visual_acceptance=False))
    print('EXCLUSIVE_WINNER_COMPLETE',checks,flush=True)


if __name__=='__main__':main()
