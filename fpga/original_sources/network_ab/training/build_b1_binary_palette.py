"""Four learned region decisions at360p, encoded by original B1 operations.

Unlike the16-way comparator, this retains full 0..127 score precision and all
three180p residual blocks for learned shape features. Eight existing decoder
channels implement a fitted16-colour dictionary; no LUT is added.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def bits(device):return ((torch.arange(16,device=device)[:,None]>>torch.arange(4,device=device))&1).float()


@torch.no_grad()
def calibrate_rows(net,keys):
    for k in keys:
        w=net.weight[k];s=w.abs().flatten(1).amax(1).clamp_min(1e-5)/120
        getattr(net,k+'_m').copy_((s*131072).ceil().clamp(1,131071).double()[None,:,None,None])


def fit_dictionary(palette):
    device=palette.device;restarts=64;x=bits(device)
    # Nearby binary codes must also denote nearby colours. Choosing a random
    # code permutation solely for decoder fit can create false bright/dark
    # colours while two learned decision boundaries are slightly misaligned.
    pal=palette.detach().cpu().numpy();distance=((pal[:,None]-pal[None])**2).sum(2)
    edges=np.array([(i,i^(1<<k)) for i in range(16) for k in range(4) if i<(i^(1<<k))])
    rng=np.random.default_rng(2026100681);orders=np.stack([rng.permutation(16) for _ in range(64)])
    def energy(order):return distance[order[:,edges[:,0]],order[:,edges[:,1]]].sum(1)
    value=energy(orders);best_order=orders.copy();best_value=value.copy();rr=np.arange(64)
    for step in range(8000):
        trial=orders.copy();ij=rng.integers(0,16,(2,64));trial[rr,ij[0]],trial[rr,ij[1]]=orders[rr,ij[1]],orders[rr,ij[0]]
        proposed=energy(trial);temp=10000*(10/10000)**(step/7999)
        accept=(proposed<value)|(rng.random(64)<np.exp(np.minimum(0,(value-proposed)/temp)))
        orders[accept]=trial[accept];value[accept]=proposed[accept]
        improve=value<best_value;best_order[improve]=orders[improve];best_value[improve]=value[improve]
    chosen=best_order[best_value.argmin()]
    print('COLOUR_CODE_EDGE_ENERGY',float(best_value.min()),chosen.tolist(),flush=True)
    perms=torch.tensor(chosen,device=device)[None].expand(restarts,-1).clone()
    target=palette[perms]/255
    w=torch.nn.Parameter(torch.randn(restarts,4,8,device=device)*.7)
    b=torch.nn.Parameter(torch.randn(restarts,1,8,device=device)*.2)
    v=torch.nn.Parameter(torch.randn(restarts,8,3,device=device)*.2)
    c=torch.nn.Parameter(target.mean(1,keepdim=True).clone());opt=torch.optim.Adam([w,b,v,c],lr=.02)
    best=(float('inf'),None)
    for step in range(1600):
        u=(x@w+b).clamp(0,1);p=(u@v+c).clamp(0,1)
        loss=(p-target).square().mean()+1e-7*(w.square().mean()+v.square().mean())
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%400==0:print('BINARY_DICTIONARY',step+1,float(((p-target).square().mean((1,2))).min().detach()),flush=True)
    with torch.no_grad():
        err=(p-target).square().mean((1,2));j=int(err.argmin())
        return w[j].T.detach(),b[j,0].detach(),v[j].T.detach(),c[j,0].detach(),perms[j],float(err[j])


@torch.no_grad()
def install(net,fit):
    w,b,v,c,perm,loss=fit;keys=['up1_low_pw','up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head']
    for key in keys:net.weight[key].zero_();net.bias[key].zero_()
    net.bias['up1_low_pw'][:4]=64
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['up1_high_dw'][:4,0]=kernel
    # Integer score>=64 ->127, otherwise0. All scores0..127 can participate.
    net.weight['up2_low_dw'][:4,0,1,1]=127;net.bias['up2_low_dw'][:4]=127*(1-64)
    net.weight['up2_low_pw'][:,:4,0,0]=w;net.bias['up2_low_pw'][:]=b*127
    net.weight['up2_high_dw'][:,0]=.5*kernel;net.weight['up2_high_dw'][:,0,1,1]+=.5
    net.weight['rgb_head'][:,:,0,0]=v*(255/127);net.bias['rgb_head'][:]=c*255-128
    calibrate_rows(net,keys)
    # Exact multipliers for threshold and masks. Recalibration must not soften this switch.
    net.up2_low_dw_m[:,:4]=131071
    net.weight['up2_low_dw'][:4,0,1,1]=127*(131071/131072)
    net.bias['up2_low_dw'][:4]=127*(1-64)*(131071/131072)


@torch.no_grad()
def actual_palette(net):
    z=torch.zeros(16,16,1,1,device='cuda');z[:,:4,0,0]=bits(z.device)*127
    return fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]/128


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--palette',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100675);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
    palette=torch.tensor(json.loads(Path(a.palette).read_text())['palette'],device='cuda',dtype=torch.float32)
    fit=fit_dictionary(palette);install(net,fit);pal=actual_palette(net);pk=palette_key(pal)
    # Explicit exhaustive integer threshold check, including all128 input codes.
    z=torch.zeros(128,16,1,1,device='cuda');z[:,:4]=torch.arange(128,device='cuda')[:,None,None,None]
    threshold=fragment(net,z,'up1_high_dw')['up2_low_dw'][:,:4,0,0]
    expected=(torch.arange(128,device='cuda')[:,None]>=64).float().expand(-1,4)*127
    assert torch.equal(threshold,expected)
    data=m.load_data();rows=data['training'][::4];xs=[];ys=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];f=net(x,all_nodes=True)['up1_low_dw']
            codes=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype('int64')).cuda()[None]
            target=F.avg_pool2d(bits('cuda')[codes].permute(0,3,1,2),4)
            xs.append(f[0,:,::3,::3].flatten(1).T);ys.append(target[0,:,::3,::3].flatten(1).T)
            if (i+1)%10==0:print('BINARY_FEATURES',i+1,len(rows),flush=True)
        x=torch.cat(xs);y=torch.cat(ys);mu=x.mean(0);sd=x.std(0).clamp_min(2);xn=(x-mu)/sd
    theta=torch.nn.Parameter(torch.zeros(4,32,device='cuda'));offset=torch.nn.Parameter(torch.zeros(4,device='cuda'))
    opt=torch.optim.LBFGS([theta,offset],lr=1,max_iter=90,history_size=10,line_search_fn='strong_wolfe');hist=[]
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(xn@theta.T+offset,y)+.0003*theta.square().mean()
        loss.backward();hist.append(float(loss.detach()));return loss
    opt.step(closure)
    with torch.no_grad():
        w=theta*9/sd[None];b=offset*9+63.5-w@mu;k='up1_low_pw'
        net.weight[k][:4,:,0,0]=w;net.bias[k][:4]=b;calibrate_rows(net,[k])
        export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            coverage_layout='4binary360_rgb16',binary_regions=True,parent=a.warm,palette=pal.cpu().tolist()),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(float_dictionary_mse=fit[-1],dictionary_rgb_mae=float((pal*128+128-palette[fit[4]]).abs().mean()),
        integer_palette=(pal*128+128).cpu().tolist(),source_palette_order=fit[4].cpu().tolist(),threshold_exhaustive_codes=128,
        classifier_history=hist,training_ids=[r['id'] for r in rows],data_sha256=file_hash(m.OUT/'data.json'),
        actual_byte_verified=True,postfilter=False,added_operators=0,decision_resolution='640x360',visual_acceptance=False))
    print('BINARY_BUILD_COMPLETE',flush=True)


if __name__=='__main__':main()
