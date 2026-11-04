"""Preserve source sampling phases inside features, constrain output geometry.

Auxiliary linear heads are discarded; the actual output is supervised only
against stylized RGB. No photographic RGB reconstruction loss enters output.
This is a hypothesis test, not a claim that phase readout explains all blur.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .train_b1_fine_rgb_native import FineRGBNativeTraining,evaluate
from .train_b1_contour_coherence import contour_coherence
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


class PhaseGeometryNativeTraining(FineRGBNativeTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        default_specs={'phase2':(16,4,1),'phase4':(32,16,1)}
        self.phase_head_specs=getattr(self,'phase_head_specs',default_specs)
        for key,(ci,co,kernel) in self.phase_head_specs.items():
            self.heads[key]=nn.Conv2d(ci,co,kernel,padding=kernel//2).cuda()
        # Small coefficients provide a nonzero feature gradient without large
        # random auxiliary outputs. All RNG is covered by checkpoint recovery.
        for head in self.heads.values():
            nn.init.normal_(head.weight,0,.001);nn.init.zeros_(head.bias)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.phase_heads_reused=bool((warm.get('source_phase_auxiliary_heads') or warm.get('stylized_phase_auxiliary_heads'))
                                    and warm.get('phase_head_specs',default_specs)==self.phase_head_specs)
        if self.phase_heads_reused:self.heads.load_state_dict(warm['heads'])
        self.opt.add_param_group(dict(params=list(self.heads.parameters()),lr=.001))
        self.base_lrs=[g['lr'] for g in self.opt.param_groups]
        def remember(module,inputs,result):
            if isinstance(result,dict):self.current_nodes=result
        self.net.register_forward_hook(remember)

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        y=(self.last_source[:,:1]+2*self.last_source[:,1:2]+self.last_source[:,2:3])/4
        phase_losses=[]
        for scale,node,key in ((2,'down1.pw','phase2'),(4,'down2.pw','phase4')):
            target=F.pixel_unshuffle(y,scale);target=target-target.mean(1,keepdim=True)
            q=self.heads[key](self.current_nodes[node]/64-1);q=q-q.mean(1,keepdim=True)
            margin=40//scale
            # Exclude replicated image/crop borders also for native frames.
            q=q[...,margin:-margin,margin:-margin];target=target[...,margin:-margin,margin:-margin]
            part=F.smooth_l1_loss(q,target,beta=2/128)
            phase_losses.append(part);values[f'feature_phase{scale}_mae_byte']=(q-target).abs().mean()*128
        aux=torch.stack(phase_losses).mean()
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],t[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,t)
        values.update(source_phase_feature_aux=aux,short_contour_turn_excess=curve)
        return loss+2*aux+4*curve,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(source_phase_auxiliary_heads=True,phase_auxiliary_weight=2,
            phase_auxiliary_targets='within-cell source luminance residual only at down1.pw/down2.pw',
            output_target='stylized RGB only; no output photo reconstruction',
            contour_coherence_weight=4,auxiliary_heads_exported=False,
            phase_head_specs=self.phase_head_specs,phase_heads_reused=self.phase_heads_reused)
        torch.save(state,path)

    def preflight_extras(self):
        result=super().preflight_extras()
        result['auxiliary_head_gradients']={k:float(h.weight.grad.norm()) for k,h in self.heads.items()}
        assert min(result['auxiliary_head_gradients'].values())>0
        assert all(torch.isfinite(h.weight.grad).all() for h in self.heads.values())
        for k in ('down1__pw','down2__pw'):
            assert self.net.weight[k].grad is not None and torch.isfinite(self.net.weight[k].grad).all()
        if self.critic is not None:
            self.step=400
            row=self.update(False)
            assert row['values']['adversarial_factor']>0 and row['temporal_pair']
            assert all(torch.isfinite(v).all() for v in self.critic.state_dict().values())
            result['full_adversarial_strength_update']=row
        return result


if __name__=='__main__':
    torch.set_num_threads(2)
    driver(training_factory=PhaseGeometryNativeTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
