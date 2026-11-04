"""CPU audit of opposing supervised/projection gradients on train crops only."""
import argparse
from pathlib import Path
import torch
import numpy as np
from . import train_b1_minimum_area as m
from .train_b1_native_distance_boundaries import NativeDistanceTraining
from .train_b1_decisive_regions import DecisiveRegionTraining
from .train_b1_shared_boundaries import SharedBoundaryTraining
from .train_b1_active_colour_boundaries import ActiveColourBoundaryTraining
from .train_b1_region_features import RegionFeatureTraining
from .train_b1_region_sets import coarse_colour
from .train_b1_ordered_regions import OrderedGradientB1
from .train_b1_balanced_region_boundaries import colour_appearance
from .build_b1_ordered_palette import bits
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .train_b1_sixteen_regions import colour_target
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--frame-index',type=int,default=0);ap.add_argument('--indoor',action='store_true')
    ap.add_argument('--decisive',action='store_true');ap.add_argument('--shared',action='store_true');ap.add_argument('--active',action='store_true')
    ap.add_argument('--features',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin');net.__class__=OrderedGradientB1
    net.load_state_dict(s['model']);net.quantized=True;net.package['residual_encoding']=s['residual_encoding']
    # Build only the loss context, deliberately without the CUDA trainer/Adam.
    cls=RegionFeatureTraining if a.features else ActiveColourBoundaryTraining if a.active else SharedBoundaryTraining if a.shared else DecisiveRegionTraining if a.decisive else NativeDistanceTraining
    tr=cls.__new__(cls);tr.net=net;tr.step=0;tr.args=argparse.Namespace(minimum_area=256)
    tr.palette=torch.tensor(s['palette']);tr.bit_table=bits('cpu');pk=palette_key(tr.palette)
    xs=[];cs=[];ls=[];ids=[]
    crops=[('070939',(672,256)),('075618',(640,240))] if a.indoor else [('121509',(768,288)),('121831',(416,224))]
    data=m.load_data()
    for suffix,(x,y) in crops:
        rows=[r for r in data['training'] if Path(r['video']).stem.endswith(suffix)]
        row=rows[a.frame_index%len(rows)]
        assert row['split']=='train';ids.append(row['id'])
        src=m.t.r.read(row['path'])[y:y+320,x:x+320]
        xs.append(torch.tensor(src.transpose(2,0,1).copy(),dtype=torch.float32)/128-1)
        cs.append(torch.from_numpy(region_class_map(row['target'],row['region'],pk)[y:y+320,x:x+320].astype(np.int64)))
        ls.append(torch.from_numpy(m.load_region(row['region'])[y:y+320,x:x+320].astype(np.int64)))
    tr.class_labels=torch.stack(cs);tr.labels=torch.stack(ls);tr.nodes=net(torch.stack(xs),all_nodes=True)
    p=tr.nodes['rgb_head'][...,40:-40,40:-40]/128;target=colour_target(tr.class_labels,tr.palette)[...,40:-40,40:-40]
    area,values=tr.additional_loss(p,target,False)
    appearance,_=(coarse_colour if a.features else colour_appearance)(p,target,'direct')
    correction=2*(values['region_feature_pull']+.25*values['region_feature_worst']+values['region_feature_separation']) if a.features else values['active_boundary_direction'] if a.active else 2*values['shared_boundary_disagreement'] if a.shared else 3*values['modal_area_correction'] if a.decisive else .5*values['actual_island_correction']
    supervised=area+appearance-correction
    keys=[k for k,w in net.weight.items() if k not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')]
    weights=[net.weight[k] for k in keys]
    grad1=torch.autograd.grad(supervised,weights,retain_graph=True,allow_unused=True)
    grad2=torch.autograd.grad(correction,weights,allow_unused=True)
    records=[];v1=[];v2=[]
    for k,w,g,h in zip(keys,weights,grad1,grad2):
        g=torch.zeros_like(w) if g is None else g;h=torch.zeros_like(w) if h is None else h
        v1.append(g.flatten());v2.append(h.flatten())
        records.append(dict(layer=k,supervised_norm=float(g.norm()),weighted_correction_norm=float(h.norm()),
            cosine=float(torch.nn.functional.cosine_similarity(g.flatten(),h.flatten(),dim=0))))
    g,h=torch.cat(v1),torch.cat(v2)
    result=dict(checkpoint_sha256=file_hash(a.checkpoint),training_ids=ids,records=records,
        supervised_loss=float(supervised.detach()),weighted_correction_loss=float(correction.detach()),
        changed_fraction=float(values['consistent_area_fraction' if a.features else 'modal_area_changed_fraction' if a.decisive or a.shared or a.active else 'actual_island_changed_fraction'].detach()),
        global_cosine=float(torch.nn.functional.cosine_similarity(g,h,dim=0)),
        correction_to_supervision_gradient_norm=float(h.norm()/g.norm()),
        omitted_losses=['source phase','tiny input perturbation','video temporal'],
        gradient_type='QAT straight-through/sigmoid training proxy; not derivative of discrete byte inference',device='cpu',decisive=a.decisive,shared=a.shared,
        compared_correction='feature regions' if a.features else 'active boundary' if a.active else 'shared boundary' if a.shared else 'modal area' if a.decisive else 'original area',
        shared_pair_coverage=float(values['shared_boundary_pair_coverage'].detach()) if a.shared else None)
    write_json(a.out,result);print('REGION_GRADIENT_AUDIT',{k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':main()
