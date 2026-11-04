"""Measure projection/colour gradient balance on training crops, using CPU."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_projected_regions import ProjectedRegionTraining,coarse_appearance
from .train_b1_anchored_projection import AnchoredProjectionTraining
from .train_b1_ordered_regions import OrderedGradientB1
from .build_b1_ordered_palette import bits
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .train_b1_sixteen_regions import colour_target
from .runtime import write_json,file_hash


def audit(path,anchored):
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.__class__=OrderedGradientB1;net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding']
    cls=AnchoredProjectionTraining if anchored else ProjectedRegionTraining
    tr=cls.__new__(cls);tr.net=net;tr.step=0;tr.args=argparse.Namespace(minimum_area=256)
    tr.palette=torch.tensor(state['palette']);tr.bit_table=bits('cpu');pk=palette_key(tr.palette)
    xs=[];cs=[];ls=[];ids=[];data=m.load_data()
    for suffix,(x,y) in [('121509',(768,288)),('121831',(416,224))]:
        row=next(r for r in data['training'] if Path(r['video']).stem.endswith(suffix))
        assert row['split']=='train';ids.append(row['id'])
        src=m.t.r.read(row['path'])[y:y+320,x:x+320]
        xs.append(torch.tensor(src.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        cs.append(torch.from_numpy(region_class_map(row['target'],row['region'],pk)[y:y+320,x:x+320].astype(np.int64)))
        ls.append(torch.from_numpy(m.load_region(row['region'])[y:y+320,x:x+320].astype(np.int64)))
    tr.class_labels=torch.stack(cs);tr.labels=torch.stack(ls);tr.nodes=net(torch.stack(xs),all_nodes=True)
    p=tr.nodes['rgb_head'][...,40:-40,40:-40]/128
    target=colour_target(tr.class_labels,tr.palette)[...,40:-40,40:-40]
    total,values=tr.additional_loss(p,target,False)
    rgb=2*values['regional_colour_anchor']+values['regional_contrast_anchor']+coarse_appearance(p,target,'')[0]
    correction=(2*values['changed_only_projection']) if anchored else 5*values['primary_region_projection']
    terms={'RGB_colour_anchor':rgb,'projection':correction}
    if anchored:terms['predecision_colour_anchor']=2*values['decision_colour_anchor']
    keys=[k for k in net.weight if k not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')]
    weights=[net.weight[k] for k in keys];grads={};records={}
    for name,value in terms.items():
        gradient=torch.autograd.grad(value,weights,retain_graph=True,allow_unused=True)
        flat=torch.cat([(torch.zeros_like(w) if g is None else g).flatten() for w,g in zip(weights,gradient)])
        grads[name]=flat;records[name]=dict(loss=float(value.detach()),gradient_norm=float(flat.norm()))
    comparisons={}
    for name,g in grads.items():
        if name=='projection':continue
        h=grads['projection'];comparisons[name]=dict(projection_to_colour_norm=float(h.norm()/g.norm().clamp_min(1e-12)),
            cosine=float(torch.nn.functional.cosine_similarity(g,h,dim=0)))
    field=(tr.nodes['up1_high_dw'][:,:15,20:-20,20:-20]-63.5)/4
    truth=torch.nn.functional.avg_pool2d(tr.bit_table[tr.class_labels].permute(0,3,1,2),2)[...,20:-20,20:-20]
    pure=(truth<.01)|(truth>.99);wrong=((field>=0)!=(truth>=.5))&pure
    return dict(checkpoint=str(path),sha256=file_hash(path),training_ids=ids,anchored=anchored,
        weighted_terms=records,comparisons=comparisons,
        wrong_pure_bit_fraction=float(wrong.sum()/pure.sum()),
        confident_wrong_bit_fraction=float((wrong&(field.abs()>6)).sum()/pure.sum()),
        changed_fraction=float(values['projected_changed_fraction'].detach()),
        omitted=['input stability','phase','temporal'],gradient='surrogate training derivative, not derivative of discrete inference')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--anchored',action='store_true');a=ap.parse_args();torch.set_num_threads(2)
    result=audit(a.checkpoint,a.anchored);write_json(a.out,result);print('PROJECTION_BALANCE',result,flush=True)


if __name__=='__main__':main()
