"""Audit revised objective on fixed TRAIN crops before another fit."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .b1_all_region_means import all_region_colour_objective,regression as mean_regression
from .b1_boundary_palette_distribution import palette_distribution,regression
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .audit_b1_rim_tolerance import rim_terms
from .train_b1_contour_coherence import contour_coherence
from .b1_geometry_first_objective import geometry_first
from .probe_b1_encoder_gauge import load
from .probe_b1_decision_slope_current import harden
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def primary(p,t,labels):
    base,v=all_region_colour_objective(p,t,labels)
    pr,_=complete_rgb_profiles(p,t,labels);rim,_=rim_terms(p,t,labels,2)
    return base+4*pr+rim+4*contour_coherence(p,t)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);mean_regression();regression();_,base=load(a.checkpoint)
    nets={'current':base}
    for name,gain in (('half_slope',.5),('double_slope',2)):
        n=copy.deepcopy(base);harden(n,gain);nets[name]=n
    rows={r['id']:r for r in m.load_data(a.data)['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];records=[]
    for item in (mined[1],mined[5],mined[6]):
        row=rows[item['row']['id']];xx,yy=item['xy'];w,h=item['wh'];cut=lambda z:z[...,40:-40,40:-40]
        tensor=lambda k:torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        x=tensor('path');t=cut(tensor('target'));labels=cut(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64))[None])
        with torch.no_grad():
            imgs={}
            for name,n in nets.items():
                nodes=n(x,all_nodes=True);actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(n.export()),device='cpu')(n.round(x*128),True)
                assert all(torch.equal(v,actual[k]) for k,v in nodes.items());imgs[name]=cut(actual['rgb_head']/128)
            imgs['blur5']=F.avg_pool2d(F.pad(imgs['current'],(2,2,2,2),mode='replicate'),5,1);imgs['target']=t
        p=imgs['current'].clone().requires_grad_();ref=primary(p,t,labels);part,_=palette_distribution(p,t,labels)
        _,terms=all_region_colour_objective(p,t,labels)
        weights={k:(8 if k=='chroma_pixel' else 4) for k in terms}
        terms['complete_profile']=complete_rgb_profiles(p,t,labels)[0];weights['complete_profile']=4
        terms['rim']=rim_terms(p,t,labels,2)[0];weights['rim']=1
        terms['coherence']=contour_coherence(p,t);weights['coherence']=4
        decomposition={}
        for key,value in terms.items():
            gradient=torch.autograd.grad(value*weights[key],p,retain_graph=True)[0]
            decomposition[key]=dict(value=float(value.detach()),weighted_gradient_norm=float(gradient.norm()),
                **{name:float((gradient*(imgs[name]-p.detach())).sum()) for name in ('half_slope','double_slope','blur5','target')})
        gr=torch.autograd.grad(ref,p,retain_graph=True)[0];gs=torch.autograd.grad(part,p)[0]
        revised=geometry_first(p,t,labels)[0]+.5*contour_coherence(p,t)
        revised_gradient=torch.autograd.grad(revised,p)[0]
        comparisons={}
        for name,img in imgs.items():
            if name=='current':continue
            delta=img-p.detach();values={'main_dot':float((gr*delta).sum()),'palette_dot':float((gs*delta).sum())}
            values['geometry_first_dot']=float((revised_gradient*delta).sum())
            for fraction in (1,2,4):
                coeff=(fraction*gr.norm()/gs.norm().clamp_min(1e-12)).clamp_max(32)
                values['combined_'+str(fraction)]=float(((gr+coeff*gs)*delta).sum())
            comparisons[name]=values
        records.append(dict(id=row['id'],directions_negative_prefers_change=comparisons,main_norm=float(gr.norm()),palette_norm=float(gs.norm()),
            decomposition=decomposition,distribution_scores={k:float(palette_distribution(v,t,labels)[0]) for k,v in imgs.items()},actual_nodes_error=0))
        print('PALETTE_DIRECTION',records[-1],flush=True)
    out=Path(a.out);out.mkdir(exist_ok=False)
    for f in ('audit_b1_palette_objective_direction.py','b1_all_region_means.py','b1_boundary_palette_distribution.py','b1_geometry_first_objective.py'):shutil.copy2(Path(__file__).with_name(f),out/f)
    write_json(out/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),scope='Fixed TRAIN RGB-space direction only; excludes phase/flow/noise/optimizer.'))


if __name__=='__main__':main()
