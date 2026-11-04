"""Four-sector local-gradient selection, compiled only into fixed B1 weights.

Inspired by Kuwahara's least-variable-neighbourhood selection. This uses mean
absolute luminance Laplacian, NOT variance or the anisotropic Kuwahara method.
No input/target masks, connected-component processing or external postfilter.
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import install as install_palette, GROUPS
from .build_b1_median_cartoon import LEVELS
from .probe_b1_decoder_coverage import fragment
from .publish_b1_style_recovery import export
from .runtime import configure_workspace, write_json


@torch.no_grad()
def build(net,wide=False):
    for k,w in net.weight.items():w.zero_();net.bias[k].zero_()
    install_palette(net,LEVELS)
    ker=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(3):net.weight['stem'][c,c,1,1]=.5
    net.bias['stem'][:3]=64
    lap=net.weight['stem'].new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])/4
    luma=net.weight['stem'].new_tensor([.25,.5,.25])
    net.weight['stem'][3]=luma[:,None,None]*lap
    net.weight['stem'][4]=-net.weight['stem'][3]
    net.weight['down1__dw'][:5,0]=ker
    for s in range(4):
        for c in range(3):net.weight['down1__pw'][s*4+c,c,0,0]=1
        net.weight['down1__pw'][s*4+3,3:5,0,0]=1
        q=ker.new_zeros(3,3);y,x=s//2,s%2
        q[y:y+2,x:x+2]=.25
        net.weight['down2__dw'][s*4:s*4+4,0]=q
        net.weight['res1_conv1__dw'][s*4:s*4+4,0]=q if wide else ker
    for c in range(16):net.weight['down2__pw'][c,c,0,0]=1
    # Pair comparisons. Colours are packed separately from comparison state.
    k='res1_conv1__pw'
    for s in range(4):
        for c in range(3):net.weight[k][s*3+c,s*4+c,0,0]=1
    for p,(a,b) in enumerate(((3,7),(11,15))):
        net.weight[k][12+p,b,0,0]=127;net.weight[k][12+p,a,0,0]=-127;net.bias[k][12+p]=127
        net.weight[k][14+2*p,b,0,0]=1;net.weight[k][14+2*p,a,0,0]=-1
        net.weight[k][15+2*p,b,0,0]=1
    k='res1_conv2__dw';net.weight[k][:18,0,1,1]=1
    k='res1_conv2__pw'
    for p in range(2):
        for c in range(3):
            j=p*6+c
            net.weight[k][j,j,0,0]=1;net.weight[k][j,12+p,0,0]=1;net.bias[k][j]=-127
            net.weight[k][j+3,j+3,0,0]=1;net.weight[k][j+3,12+p,0,0]=-1
        net.weight[k][12+p,15+2*p,0,0]=1;net.weight[k][12+p,14+2*p,0,0]=-1
    net.weight['res2_conv1__dw'][:14,0,1,1]=1
    k='res2_conv1__pw'
    for p in range(2):
        for c in range(3):
            net.weight[k][p*3+c,p*6+c,0,0]=1;net.weight[k][p*3+c,p*6+c+3,0,0]=1
    net.weight[k][6,13,0,0]=127;net.weight[k][6,12,0,0]=-127;net.bias[k][6]=127
    net.weight['res2_conv2__dw'][:7,0,1,1]=1
    k='res2_conv2__pw'
    for c in range(3):
        net.weight[k][c,c,0,0]=1;net.weight[k][c,6,0,0]=1;net.bias[k][c]=-127
        net.weight[k][c+3,c+3,0,0]=1;net.weight[k][c+3,6,0,0]=-1
    net.weight['res3_conv1__dw'][:6,0,1,1]=1
    # Half-range RGB to Y/2, Co/2, Cg/2; apply fixed cool/contrast mapping.
    matrix=np.array([[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]])
    k='res3_conv1__pw'
    for g,(start,end) in enumerate(GROUPS):
        v=np.array(LEVELS[g],float);cuts=(v[:-1]+v[1:])/2
        raw=((cuts+8)/1.06/2 if g==0 else (cuts+8)/1.20/2 if g==1 else cuts/1.15/2)
        for j,cut in enumerate(raw):
            net.weight[k][start+j,:3,0,0]=net.weight[k].new_tensor(matrix[g])
            net.weight[k][start+j,3:6,0,0]=net.weight[k].new_tensor(matrix[g])
            net.bias[k][start+j]=64-cut
    for k in ('res3_conv2__dw','up1_low_dw'):net.weight[k][:15,0,1,1]=1
    for k in ('res3_conv2__pw','up1_low_pw'):
        net.weight[k].zero_();net.bias[k].zero_()
        for c in range(15):net.weight[k][c,c,0,0]=1
    for b in range(1,4):net.package['residual_encoding'][f'res{b}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    # Preserve palette's calibrated fractions. All hand-coded integer +/-127
    # comparison rows use the maximum Q17 multiplier, as the FPGA requires.
    preserved={'up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'}
    for k,w in net.weight.items():
        if k in preserved:continue
        maximum=w.abs().flatten(1).amax(1)
        scale=torch.pow(2,torch.ceil(torch.log2(maximum.clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def verify(net):
    gen=torch.Generator(device='cuda').manual_seed(610069)
    z=torch.zeros(256,32,8,8,device='cuda');z[:,:16]=torch.randint(0,128,(256,16,8,8),generator=gen,device='cuda')
    z[0,3:16:4]=7;z[1,3:16:4]=0;z[2,3:16:4]=127
    # conv2 PW has no ReLU. The existing following DW applies that clamp,
    # so selection is complete at res3_conv1.dw, not at the residual add.
    part=fragment(net,z,'res1_conv1.dw');out=part['res3_conv1.dw'];rgb=out[:,:3]+out[:,3:6]
    costs=z[:,3:16:4];winner=costs.argmin(1)
    colours=torch.stack([z[:,4*s:4*s+3] for s in range(4)],2)
    expected=colours.gather(2,winner[:,None,None].expand(-1,3,1,-1,-1)).squeeze(2)
    error=float((rgb-expected).abs().max());assert error==0,error
    return dict(random_and_tie_cases=16384,selection_error=error,tie_break='first_sector')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);rows=m.load_data()['validation'];results=[]
    for wide in (False,True):
        net=m.t.r.initialize();net.quantized=True;build(net,wide);check=verify(net)
        sub=dest/('wide' if wide else 'base');sub.mkdir()
        export(net,rows,sub/'package');m.t.preview(net,rows,sub/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            manual_sector_selection=True,ordered_regions=True,coverage_layout='15ordered360_rgb200',trained=False),sub/'initial.pt')
        results.append(dict(wide=wide,check=check));print('SECTOR_SELECTION_DONE',results[-1],flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(results=results,method='4sector_minimum_mean_absolute_luma_laplacian_then_ordered_colours',
        reference='https://www.kyprianidis.com/p/pg2009/jkyprian-pg2009.pdf',not_full_kuwahara=True,trained=False,
        added_operators=0,postfilter=False,guaranteed_minimum_area=False,visual_acceptance=False))


if __name__=='__main__':main()
