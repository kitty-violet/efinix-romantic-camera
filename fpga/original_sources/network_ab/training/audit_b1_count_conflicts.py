"""CPU audit of weighted count, self-area and RGB-stability gradient conflicts."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_count_regions import CountRegionTraining
from .train_b1_count_regions import count_targets
from .train_b1_decisive_regions import modal_region_cleanup
from .train_b1_ordered_regions import OrderedGradientB1
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .train_b1_sixteen_regions import colour_target
from .runtime import write_json,file_hash


def audit(path,truth_area=False):
    torch.manual_seed(731)
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.__class__=OrderedGradientB1;net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding']
    if truth_area:
        from .train_b1_truth_area_counts import TruthAreaCountTraining
        cls=TruthAreaCountTraining
    else:cls=CountRegionTraining
    tr=cls.__new__(cls);tr.net=net;tr.step=0
    tr.palette=torch.tensor(state['palette']);pk=palette_key(tr.palette);xs=[];codes=[];ids=[]
    for suffix,(x,y) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in m.load_data()['training'] if Path(r['video']).stem.endswith(suffix))
        assert row['split']=='train';ids.append(row['id'])
        source=m.t.r.read(row['path'])[y:y+320,x:x+320]
        xs.append(torch.tensor(source.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        codes.append(torch.tensor(region_class_map(row['target'],row['region'],pk)[y:y+320,x:x+320].astype(np.int64)))
    x=torch.stack(xs);tr.class_labels=torch.stack(codes);tr.nodes=net(x,all_nodes=True)
    pred=tr.nodes['rgb_head']/128;cut=lambda a:a[...,40:-40,40:-40]
    _,values=tr.additional_loss(cut(pred),cut(colour_target(tr.class_labels,tr.palette)),False)
    tiny,_=tr.input_stability_loss(x,pred,False)
    shifted=torch.roll(net(torch.roll(x[:1],(1,1),(-2,-1)))[0],(-1,-1),(-2,-1))
    area_key='truth_anchored_area_correction' if truth_area else 'actual_count_area_correction'
    terms=dict(supervised=values['actual_palette_count_ce'],area_correction=.3*values[area_key],
        tiny_input=tiny,phase=.3*F.l1_loss(cut(pred[:1]),cut(shifted)))
    keys=[k for k in net.weight if k not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')]
    parameters=[p for k in keys for p in (net.weight[k],net.bias[k])];grads={};records={}
    for name,value in terms.items():
        gs=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
        g=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(parameters,gs)])
        grads[name]=g;records[name]=dict(loss=float(value.detach()),norm=float(g.norm()))
    main=grads['supervised'];total_aux=sum(g for k,g in grads.items() if k!='supervised')
    for name,g in dict(**grads,total_aux=total_aux).items():
        if name=='supervised':continue
        records.setdefault(name,{})
        records[name].update(norm_ratio=float(g.norm()/main.norm()),cosine=float(F.cosine_similarity(main,g,dim=0)),
            opposing_fraction_of_main=float(-(main*g).sum()/main.square().sum()))
    palette=(tr.palette.numpy()*128+128).round().clip(0,255).astype(np.uint8)
    corrections=[]
    for i,item in enumerate(tr.nodes['up2_low_dw'][:,:15]):
        code=tr.predicted_codes(item-63.5).detach().numpy();clean,_=modal_region_cleanup(code,palette,64,2)
        full=tr.class_labels[i:i+1];levels=count_targets(full)
        desired=(levels[0].argmax(1)*25+levels[1].argmax(1)*5+levels[2].argmax(1))[0].numpy()
        pure=np.logical_and.reduce([v.max(1).values[0].numpy()>.999 for v in levels])
        code,clean,desired,pure=[v[20:-20,20:-20] for v in (code,clean,desired,pure)]
        changed=(code!=clean)&pure;n=max(1,int(changed.sum()))
        corrections.append(dict(id=ids[i],changed_pure_pixels=int(changed.sum()),
            reaches_target=float(((clean==desired)&changed).sum()/n),
            destroys_previously_correct=float(((code==desired)&changed).sum()/n),
            remains_wrong=float(((clean!=desired)&changed).sum()/n)))
    return dict(checkpoint=str(path),sha256=file_hash(path),train_ids=ids,records=records,
        truth_anchored_area=truth_area,old_modal_label_audit=corrections,
        no_temporal_pair_in_this_diagnostic=True,gradient='surrogate derivative in effective parameter coordinates')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--truth-area',action='store_true')
    a=ap.parse_args();torch.set_num_threads(2);r=audit(a.checkpoint,a.truth_area);write_json(a.out,r);print(r,flush=True)
