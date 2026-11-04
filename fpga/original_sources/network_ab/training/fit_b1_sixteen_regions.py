"""Initialize16 class scores from actual train features, then compile B1.

Region IDs assign one palette entry to an entire training area. Anti-aliased
RGB pixels at its boundary cannot create an unintended third colour class.
"""
import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .probe_b1_sixteen_decoder import build,palette_actual
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@lru_cache(maxsize=96)
def region_class_map(target_path,region_path,palette_tuple):
    rgb=m.t.r.read(target_path).astype(np.float64);labels=m.load_region(region_path).astype(np.int64)
    n=int(labels.max())+1;count=np.bincount(labels.ravel(),minlength=n).clip(1)
    colours=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel(),minlength=n)/count for c in range(3)],1)
    palette=np.array(palette_tuple)
    ids=((colours[:,None]-palette[None])**2).sum(2).argmin(1)
    return ids[labels].astype(np.uint8)


def palette_key(palette):return tuple(tuple(float(x) for x in row) for row in (palette.detach().cpu().numpy()*128+128))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--palette',required=True);ap.add_argument('--name',required=True);ap.add_argument('--exact-tournament',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(2026100663);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    p=torch.tensor(json.loads(Path(a.palette).read_text())['palette'],device='cuda');build(net,p);net.requires_grad_(False)
    tournament=None
    if a.exact_tournament:
        from .compile_b1_tournament import compile_tournament,verify
        compile_tournament(net);tournament=verify(net);print('FULL_PRECISION_TOURNAMENT',tournament,flush=True)
    palette=palette_actual(net);pk=palette_key(palette);d=m.load_data();rows=d['training'][::4];xs=[];ys=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];f=net(x,all_nodes=True)['res1_conv1.dw']
            ids=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64)).cuda()[None]
            truth=F.avg_pool2d(F.one_hot(ids,16).permute(0,3,1,2).float(),4)
            xs.append(f[0,:,::3,::3].flatten(1).T);ys.append(truth[0,:,::3,::3].flatten(1).T)
            if (i+1)%10==0:print('SIXTEEN_FEATURES',i+1,len(rows),flush=True)
        x=torch.cat(xs);truth=torch.cat(ys);mu=x.mean(0);sd=x.std(0).clamp_min(2);xn=(x-mu)/sd
    theta=torch.nn.Parameter(torch.zeros(16,32,device='cuda'));offset=torch.nn.Parameter(torch.zeros(16,device='cuda'))
    optimizer=torch.optim.LBFGS([theta,offset],lr=1,max_iter=90,history_size=10,line_search_fn='strong_wolfe');history=[]
    frequency=truth.mean(0);weight=frequency.clamp_min(.001).rsqrt();weight=(weight/weight.mean()).clamp(.3,3)
    def closure():
        optimizer.zero_grad();logits=xn@theta.T+offset
        ce=-(truth*weight[None]*F.log_softmax(logits,1)).sum(1).mean()
        loss=ce+.0004*theta.square().mean();loss.backward();history.append(float(loss.detach()))
        if len(history)%15==0:print('SIXTEEN_SCORE_FIT',len(history),history[-1],flush=True)
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        # Class scores span0..7 after /18. The priority is compensated in
        # initial biases; exact rounding still defines the actual decision.
        w=theta*9/sd[None];b=offset*9+64-w@mu
        if not a.exact_tournament:b-=18*torch.arange(16,device='cuda')/16
        k='res1_conv1__pw';net.weight[k].zero_();net.bias[k].zero_();net.weight[k][:16,:,0,0]=w;net.bias[k][:16]=b
        s=net.weight[k].abs().flatten(1).amax(1).clamp_min(1e-5)/120
        getattr(net,k+'_m').copy_((s*131072).ceil().clamp(1,131071).double()[None,:,None,None])
        export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            coverage_layout='16exclusive_rgb',sixteen_winner=True,exact_tournament=a.exact_tournament,parent=a.warm,palette_source=a.palette),dest/'initial.pt')
    for f in (Path(__file__),Path(__file__).with_name('probe_b1_sixteen_decoder.py')):shutil.copy2(f,dest/f.name)
    write_json(dest/'review.json',dict(history=history,samples=len(x),training_ids=[r['id'] for r in rows],data_sha256=file_hash(m.OUT/'data.json'),
        tournament=tournament,actual_byte_verified=True,postfilter=False,visual_acceptance=False))
    print('SIXTEEN_FIT_COMPLETE',flush=True)


if __name__=='__main__':main()
