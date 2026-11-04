"""B1 ordered colour decisions at360p:8 luminance x5 x5 chroma levels.

Fifteen thresholds fit the existing16-channel stage. They are combined before
the existing8-channel stage, so complete colour blocks need not mean16 colours
for the entire image. All palette arithmetic is original convolution weights.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .build_b1_binary_palette import calibrate_rows
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json
from .engine import set_reproducible

GROUPS=((0,7),(7,11),(11,15))


def levels(device):return torch.cartesian_prod(torch.arange(8,device=device),torch.arange(5,device=device),torch.arange(5,device=device))


def bits(device):
    idx=levels(device)
    return torch.cat([(idx[:,g,None]>torch.arange(end-start,device=device)[None]).float() for g,(start,end) in enumerate(GROUPS)],1)


def offsets(device):
    return torch.cat([63.5-(torch.arange(end-start,device=device)+.5)*127/(end-start) for start,end in GROUPS])


@torch.no_grad()
def project(net):
    k='up1_low_pw';off=offsets(net.weight[k].device)
    for start,end in GROUPS:
        w=net.weight[k][start:end].mean(0);b=(net.bias[k][start:end]-off[start:end]).mean()
        net.weight[k][start:end]=w;net.bias[k][start:end]=b+off[start:end]


@torch.no_grad()
def actual_palette(net):
    z=torch.zeros(200,16,1,1,device='cuda');z[:,:15,0,0]=bits(z.device)*127
    return fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]/128


@torch.no_grad()
def install(net,colour_levels=None):
    device=net.weight['stem'].device
    keys=['up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head']
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.bias['up1_low_pw'][:15]=63.5+offsets(device)
    net.weight['up1_high_dw'][:15,0]=kernel
    scale=131071/131072;net.weight['up2_low_dw'][:15,0,1,1]=127*scale;net.bias['up2_low_dw'][:15]=127*(1-64)*scale
    colour_levels=colour_levels or [list(range(16,241,32)),[-48,-24,0,24,48],[-40,-20,0,20,40]]
    for g,(start,end) in enumerate(GROUPS):
        v=torch.tensor(colour_levels[g],device=device);assert len(v)==end-start+1 and torch.all(v[1:]>v[:-1])
        net.weight['up2_low_pw'][g,start:end,0,0]=torch.diff(v)/(2*127)
    net.weight['up2_high_dw'][:3,0]=.5*kernel;net.weight['up2_high_dw'][:3,0,1,1]+=.5
    net.weight['rgb_head'][:,:3,0,0]=net.weight['rgb_head'].new_tensor([[2,2,-2],[2,0,2],[2,-2,-2]])
    y,co,cg=[v[0] for v in colour_levels]
    net.bias['rgb_head'][:]=net.bias['rgb_head'].new_tensor([y+co-cg-128,y+cg-128,y-co-cg-128])
    calibrate_rows(net,keys);net.up2_low_dw_m[:,:15]=131071
    net.rgb_head_m.fill_(4096);net.up2_high_dw_m.fill_(1024);net.up1_high_dw_m.fill_(256)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);ap.add_argument('--levels',default='');a=ap.parse_args()
    configure_workspace();set_reproducible(2026100683);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
    colour_levels=json.loads(Path(a.levels).read_text())['levels'] if a.levels else None
    install(net,colour_levels);pal=actual_palette(net);pk=palette_key(pal);d=m.load_data();rows=d['training'][::4];xs=[];ys=[]
    table=levels('cuda').float()*torch.tensor([127/7,127/4,127/4],device='cuda')
    with torch.no_grad():
        for i,row in enumerate(rows):
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];f=net(x,all_nodes=True)['up1_low_dw']
            ids=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64)).cuda()[None]
            target=F.avg_pool2d(table[ids].permute(0,3,1,2),4)
            xs.append(f[0,:,::3,::3].flatten(1).T);ys.append(target[0,:,::3,::3].flatten(1).T)
            if (i+1)%10==0:print('ORDERED_FEATURES',i+1,len(rows),flush=True)
        x=torch.cat(xs).double();y=torch.cat(ys).double();mu=x.mean(0);sd=x.std(0).clamp_min(2);xn=(x-mu)/sd
        design=torch.cat((xn,torch.ones(len(xn),1,device='cuda',dtype=torch.float64)),1)
        ridge=torch.eye(33,device='cuda',dtype=torch.float64)*len(x)*.002;ridge[-1,-1]=0
        theta=torch.linalg.solve(design.T@design+ridge,design.T@y);w=(theta[:32].T/sd[None]).float();b=(theta[-1]-mu@(theta[:32]/sd[:,None])).float()
        off=offsets('cuda')
        for g,(start,end) in enumerate(GROUPS):net.weight['up1_low_pw'][start:end,:,0,0]=w[g];net.bias['up1_low_pw'][start:end]=b[g]+off[start:end]
        calibrate_rows(net,['up1_low_pw']);project(net)
        # Shared rows/scales enforce monotonic ordered decisions for any input.
        for start,end in GROUPS:
            assert torch.equal(net.weight['up1_low_pw'][start:end],net.weight['up1_low_pw'][start:start+1].expand(end-start,-1,-1,-1))
            assert torch.equal(net.up1_low_pw_m[:,start:end],net.up1_low_pw_m[:,start:start+1].expand(-1,end-start,-1,-1))
        export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            ordered_regions=True,coverage_layout='15ordered360_rgb200',parent=a.warm,palette=pal.cpu().tolist(),levels_source=a.levels),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(training_ids=[r['id'] for r in rows],integer_palette=(pal*128+128).cpu().tolist(),
        possible_codes=200,levels_source=a.levels,decision_resolution='640x360',added_operators=0,postfilter=False,visual_acceptance=False))
    print('ORDERED_BUILD_COMPLETE',flush=True)


if __name__=='__main__':main()
