"""Fit16 independent region decisions on actual B1 train features.

This removes the handcrafted assumption that every boundary is a threshold
on only one of three tied coordinates. The fitted classifiers occupy the
existing32x32 pointwise layer; morphology/colour formation remain in B1.
"""
import argparse
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def palette_of(net):
    aa=round(float(net.weight['up2_high_dw'][0,0].sum()*127))
    indices=torch.cartesian_prod(torch.arange(9,device='cuda'),torch.arange(5,device='cuda'),torch.arange(5,device='cuda'))
    features=torch.stack([(indices[:,c]-2*g).clamp(0,2)*(aa/2) for c,groups in enumerate((4,2,2)) for g in range(groups)],1)
    scale=(net.rgb_head_m/131072).float().flatten();w=net.round(net.weight['rgb_head'][:,:,0,0]/scale[:,None]).clamp(-127,127);b=net.round(net.bias['rgb_head']/scale)
    return net.round((features@w.T+b)*scale).clamp(-128,127)/128,indices


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006189);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];net.eval().requires_grad_(False);data=m.load_data();rows=data['training'][::4]
    with torch.no_grad():
        net.weight['res3_conv1__dw'].zero_();net.weight['res3_conv1__dw'][:,0,1,1]=1;net.bias['res3_conv1__dw'].zero_();net.res3_conv1__dw_m.fill_(2048)
        palette,indices=palette_of(net);xs=[];ys=[]
        for i,row in enumerate(rows):
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
            node=net(x,all_nodes=True)['res3_conv1.dw'];feature=node[0,:,::3,::3].flatten(1).T
            colour=F.avg_pool2d(t,4)[0,:,::3,::3].flatten(1).T
            distance=colour.square().sum(1,keepdim=True)+palette.square().sum(1)[None]-2*colour@palette.T
            idx=indices[distance.argmin(1)]
            truth=torch.cat([(idx[:,c:c+1]>=torch.arange(1,n+1,device='cuda')[None]).float() for c,n in enumerate((8,4,4))],1)
            xs.append(feature);ys.append(truth)
            if (i+1)%10==0:print('REGION_CLASS_FEATURES',i+1,len(rows),flush=True)
        x=torch.cat(xs);truth=torch.cat(ys);mu=x.mean(0);sd=x.std(0).clamp_min(2);xn=(x-mu)/sd
        key='res3_conv1__pw';s=(net.res3_conv1__pw_m/131072).float().flatten()
        w=net.round(net.weight[key][:16,:,0,0]/s[:16,None]).clamp(-127,127)*s[:16,None]
        b=net.round(net.bias[key][:16]/s[:16])*s[:16]
        theta=torch.nn.Parameter(w*sd[None]/24);offset=torch.nn.Parameter((w@mu+b-63.5)/24)
        positive=truth.sum(0).clamp_min(1);pos_weight=((len(truth)-positive)/positive).sqrt().clamp(.5,3)
    optimizer=torch.optim.LBFGS([theta,offset],lr=1,max_iter=100,history_size=12,line_search_fn='strong_wolfe')
    history=[]
    def closure():
        optimizer.zero_grad();logits=xn@theta.T+offset
        loss=F.binary_cross_entropy_with_logits(logits,truth,pos_weight=pos_weight)+.0003*theta.square().mean()
        loss.backward();history.append(float(loss.detach()))
        if len(history)%10==0:print('REGION_CLASS_FIT',len(history),history[-1],flush=True)
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        w=theta*24/sd[None];b=offset*24+63.5-w@mu
        net.weight[key].zero_();net.bias[key].zero_();net.weight[key][:16,:,0,0]=w;net.bias[key][:16]=b
        scale=(net.weight[key].abs().flatten(1).amax(1).clamp_min(1e-5)/112)
        net.res3_conv1__pw_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
        export(net,data['validation'],dest/'package');m.t.preview(net,data['validation'],dest/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=state['residual_encoding'],
            field_node='res2_add',fixed_sampling=state.get('fixed_sampling',False),classifier_mode=True,parent=a.warm),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),training_ids=[r['id'] for r in rows],
        sampled_pixels=len(x),history=history,added_operators=0,source_inference=True,visual_acceptance=False))
    print('REGION_CLASSIFIERS_COMPLETE',flush=True)


if __name__=='__main__':main()
