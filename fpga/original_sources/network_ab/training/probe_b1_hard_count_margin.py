"""Actual sign-count margins, invariant to arbitrary internal bit order.

The r-th largest logit must be positive iff the target count exceeds r.
This directly supervises hard deployed counts, while permitting any channel
assignment. Sorting is training-only. No new operator or LUT is exported.
"""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from .build_b1_ordered_palette import GROUPS
from .train_b1_count_regions import count_targets,log_count_probability
from . import train_b1_minimum_area as m
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .runtime import write_json,file_hash


def hard_count_margin(logits,targets,margin=1.5):
    losses=[]
    for (start,end),truth in zip(GROUPS,targets):
        ranked=logits[:,start:end].sort(dim=1,descending=True,stable=True).values
        positive=1-truth.cumsum(1)[:,:-1]
        losses.append((positive*F.relu(margin-ranked)+(1-positive)*F.relu(margin+ranked)).mean())
    return torch.stack(losses).mean()


def regression():
    torch.manual_seed(719);codes=torch.randint(0,200,(2,4,5));targets=count_targets(codes,1)
    logits=torch.randn(2,15,4,5,requires_grad=True)
    loss=hard_count_margin(logits,targets)
    permutation=[3,6,0,4,2,5,1,9,7,10,8,14,11,13,12]
    assert torch.allclose(loss,hard_count_margin(logits[:,permutation],targets),atol=1e-7)
    levels=(codes//25,(codes//5)%5,codes%5)
    exact=torch.cat([((lev[:,None]>torch.arange(z-a)[None,:,None,None])*2-1).float()*3
        for lev,(a,z) in zip(levels,GROUPS)],1)
    assert hard_count_margin(exact,targets)==0
    for g,(a,z) in enumerate(GROUPS):
        wrong=exact.clone();wrong[:,a]=-wrong[:,a]
        assert hard_count_margin(wrong,targets)>0
    # The equal-probability counterexample has hard count0 instead of3.
    z=torch.full((1,15,1,1),float(torch.logit(torch.tensor(3/7))),requires_grad=True)
    target=count_targets(torch.tensor([[[3*25+2*5+2]]]),1)
    loss=hard_count_margin(z,target);loss.backward()
    assert z.grad.norm()>0
    assert (z.grad[:,:3]<0).all() and (z.grad[:,3:7]>0).all()
    print('HARD_COUNT_MARGIN_REGRESSION permutation / exact / wrong / symmetric_escape PASS',flush=True)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    # Regression requires gradients despite the read-only audit forward below.
    with torch.enable_grad():regression()
    torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    data=m.load_data();pk=palette_key(torch.tensor(state['palette']));records=[]
    for suffix,(xx,yy) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix));assert row['split']=='train'
        rgb=m.t.r.read(row['path'])[yy:yy+320,xx:xx+320]
        x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        nodes=net(x,all_nodes=True);logits=(nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
        codes=region_class_map(row['target'],row['region'],pk)[yy:yy+320,xx:xx+320]
        truth=[v[...,20:-20,20:-20] for v in count_targets(torch.tensor(codes.astype('int64'))[None])]
        for g,((start,end),desired) in enumerate(zip(GROUPS,truth)):
            logp=log_count_probability(logits[:,start:end]);mode=logp.argmax(1)
            hard=(logits[:,start:end]>0).sum(1);gt=desired.argmax(1);pure=desired.amax(1)>.999
            wrong=(hard!=gt)&pure
            records.append(dict(id=row['id'],group=g,pure_pixels=int(pure.sum()),wrong_hard_pixels=int(wrong.sum()),
                probability_mode_correct_but_hard_wrong=int(((mode==gt)&wrong).sum()),
                wrong_fraction_explained_by_decision_mismatch=float(((mode==gt)&wrong).sum()/wrong.sum().clamp_min(1)),
                probability_mode_accuracy=float((mode[pure]==gt[pure]).float().mean()),
                actual_hard_accuracy=float((hard[pure]==gt[pure]).float().mean())))
    write_json(a.out,dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,
        optimized_weights=False,training_crops_only=True));print(records,flush=True)


if __name__=='__main__':main()
