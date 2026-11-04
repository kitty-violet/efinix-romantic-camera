"""Check critic preference and actual RGB-space directions before increasing it.

This crop diagnostic excludes temporal/phase/noise gradients and does not
claim to reproduce a complete optimizer update. All B1 images are true byte
replays; blur/target cases are explicitly discriminator-only interventions.
"""
import argparse,copy,json,shutil
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_decision_slope_current import harden
from .train_b1_style_recovery import ConditionalCritic
from .train_b1_complete_rgb_geometry import CompleteRGBGeometryTraining
from .train_b1_contour_coherence import contour_coherence
from .b1_local_colour_commitment import colour_commitment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint)
    critic=ConditionalCritic(True);critic.load_state_dict(state['critic']);critic.eval().requires_grad_(False)
    networks={'current':base}
    for name,gain in (('half_slope',.5),('double_slope',2)):
        n=copy.deepcopy(base);harden(n,gain);networks[name]=n
    data=m.load_data(a.data);rows={r['id']:r for r in data['training']};mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];records=[]
    cut=lambda v:v[...,40:-40,40:-40]
    for item in (mined[1],mined[5],mined[6]):
        row=rows[item['row']['id']];xx,yy=item['xy'];w,h=item['wh']
        tensor=lambda key:torch.from_numpy(m.t.r.read(row[key])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        x= tensor('path');t=cut(tensor('target'));hard=cut(tensor('hard_region_target'))
        labels=torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64))[None];images={}
        with torch.no_grad():
            for name,n in networks.items():
                nodes=n(x,all_nodes=True);actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(n.export()),device='cpu')(n.round(x*128),True)
                assert all(torch.equal(v,actual[k]) for k,v in nodes.items());images[name]=cut(actual['rgb_head']/128)
            images.update(target=t,target_blur5=F.avg_pool2d(F.pad(t,(2,2,2,2),mode='replicate'),5,1),
                current_blur5=F.avg_pool2d(F.pad(images['current'],(2,2,2,2),mode='replicate'),5,1))
            score={name:float(torch.stack([v.mean() for v in critic(z,cut(x))]).mean()) for name,z in images.items()}
        p=images['current'].clone().requires_grad_();obj=object.__new__(CompleteRGBGeometryTraining);obj.labels=labels
        primary,_=obj.additional_loss(p,t,False);primary=primary+4*contour_coherence(p,t)
        part,_=colour_commitment(p,t,hard,cut(labels))
        g0=torch.autograd.grad(primary,p,retain_graph=True)[0];gs=torch.autograd.grad(part,p,retain_graph=True)[0]
        scale=(.75*g0.norm()/gs.norm().clamp_min(1e-12)).detach().clamp_max(16)
        reference=primary+scale*part;adv=-torch.stack([v.mean() for v in critic(p,cut(x))]).mean()
        gr=torch.autograd.grad(reference,p,retain_graph=True)[0];ga=torch.autograd.grad(adv,p)[0]
        directions={}
        for name in ('half_slope','double_slope','current_blur5','target'):
            delta=images[name]-p.detach();v={'main_dot_delta':float((gr*delta).sum()),'unscaled_critic_dot_delta':float((ga*delta).sum()),'delta_rgb_mae':float(delta.abs().mean()*128)}
            for amount in (.5,2):
                coefficient=(amount*gr.norm()/ga.norm().clamp_min(1e-12)).clamp_max(4)
                v['combined_direction_at_'+str(amount)]=float(((gr+coefficient*ga)*delta).sum())
            directions[name]=v
        records.append(dict(id=row['id'],critic_scores_higher_is_more_target_like=score,directions_negative_prefers_change=directions,
            main_output_gradient_norm=float(gr.norm()),critic_output_gradient_norm=float(ga.norm()),rgb_commitment_coefficient=float(scale),all_nodes_error=0))
        print('CRITIC_DIRECTION',records[-1],flush=True)
    out=Path(a.out);out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),training_crops_only=True,
        includes='final RGB/profile/rim/coherence/local-RGB and critic',excludes='phase/flow/shift/noise/synthetic and optimizer mapping',not_a_complete_training_step=True))


if __name__=='__main__':main()
