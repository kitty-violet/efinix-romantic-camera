"""Learn native contour distance in30 existing ordered fields, same B1.

Area-g7 gives .307px straight-edge RMS and2.874px10--90 width in the
controlled probe. Native SDF8/g11 gives .155px and3.231px. This measured
tradeoff, not a claim of learned success, motivates the representation.
Distance transforms and labels exist only in the trainer. Deployment is the
same single fixed B1 graph and20KiB coefficients, without any postfilter.
"""
from pathlib import Path
import copy
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_radix_reachable_ordinal_band import RadixReachableBandTraining, OrdinalBandB1
from .train_b1_ordinal_transition_band import driver, preview, rgb_evaluate
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_region_tail_flatness import region_tail_flatness
from .train_b1_packed720_local_rims import local_rim_loss
from .train_b1_boundary_profile import boundary_profile
from .train_b1_contour_coherence import contour_coherence
from .train_b1_ordered_boundary_fields import rank_loss
from .probe_b1_ordered_distance_geometry import distance_fields, set_ramp
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits, GROUPS30


def distance_objective(field, truth):
    p=field[:,:30]/127; t=truth/127
    terms=F.smooth_l1_loss(p,t,beta=.08,reduction='none')
    band=(F.max_pool2d(t,5,1,2)+F.max_pool2d(-t,5,1,2))>.02
    full=terms.mean(); edge=(terms*band).sum()/band.sum().clamp_min(1)
    return full,edge


class OrderedDistanceRimTraining(RadixReachableBandTraining):
    boundary_profile_loss=staticmethod(boundary_profile)
    def configure_distance_decoder(self):
        set_ramp(self.net,11)

    def __init__(self,args,data):
        super().__init__(args,data)
        self.configure_distance_decoder()
        self.fixed['up2_low_dw']=(self.net.weight['up2_low_dw'].detach().clone(),self.net.bias['up2_low_dw'].detach().clone())
        self.setup=copy.deepcopy(self.setup);self.setup['gain']=11

    def additional_loss(self,p,t,native):
        with torch.no_grad():
            truth=distance_fields(self.class_labels,8)
            self.supervision_distance_truth=truth
            rendered=fragment(self.net,F.pad(self.net.round(truth),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head']/128
        field=self.nodes['up1_low_dw'];labels=self.labels
        if not native:
            field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
            labels=labels[...,40:-40,40:-40];rendered=rendered[...,40:-40,40:-40]
        assert rendered.shape==p.shape
        region,v=joint_region_loss(p,rendered,labels);tail=region_tail_flatness(p,labels)
        rim=local_rim_loss(p,rendered,labels);profile,terms=self.boundary_profile_loss(p,rendered,labels)
        if native:
            yy=self.rng.randrange(p.shape[-2]-384+1);xx=self.rng.randrange(p.shape[-1]-384+1)
            curve=contour_coherence(p[...,yy:yy+384,xx:xx+384],rendered[...,yy:yy+384,xx:xx+384])
        else:curve=contour_coherence(p,rendered)
        full,band=distance_objective(field,truth);rank=rank_loss(field)
        rgb=.7*region+4*v['region_flatness']+v['adjacent_region_contrast']+4*tail+4*curve+4*rim+4*profile
        v.update(terms);v.update(ordered_distance_full=full,ordered_distance_band=band,
            ordinal_rank_violation=rank,region_tail_flatness=tail,mean_centred_local_bright_rim=rim,
            short_contour_turn_excess=curve,motion_aligned_target_fraction=self.motion_fraction,
            reachable_vs_original_rgb_mae=((rendered-t).abs().mean()*128).detach())
        return rgb+self.args.classification*(8*full+4*band)+10*rank,v

    def preflight_extras(self):
        while self.step%4:self.update(True)
        out=self.update(False)
        assert out['temporal_pair'] and 'tiny_input_stability' in out['values'] and 'ordered_distance_band' in out['values']
        return out

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(ordinal_target='native720 signed distance, gain8 then avg4, clip0..127',
            rgb_training_target='same fixed radix g11 decoder of known ordered distance',
            ordered_distance_gain=8,decoder_ramp_gain=11,area_supervision_replaced=True,
            signed_distance_training_only=True,extra_inference_operations=0,visual_acceptance=False)
        torch.save(state,path)


def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    state,net=load(checkpoint);net.__class__=OrdinalBandB1;set_ramp(net,11)
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    palette=fragment(net,z,'up1_low_dw')['rgb_head']
    setup=state['binary_coverage720'];lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    wanted=(lev@lev.new_tensor(setup['matrix']).T+lev.new_tensor(setup['bias'])).clamp(-128,127)
    assert torch.equal(palette[:,:,0,0],wanted)
    codes=torch.zeros(1,64,80,dtype=torch.long);codes[:,:,40:]=1214
    truth=distance_fields(codes,8);assert sum(distance_objective(truth,truth))==0
    wrong=(127-truth).requires_grad_();loss=sum(distance_objective(wrong,truth));loss.backward()
    assert torch.isfinite(wrong.grad).all() and wrong.grad[...,0].mean()>0 and wrong.grad[...,-1].mean()<0
    for lo,hi in GROUPS30:assert (truth[:,lo:hi-1]>=truth[:,lo+1:hi]).all()
    x=torch.rand(1,3,64,80)*1.9-1;nodes=net(x,all_nodes=True);actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
    other,_=actual(net.round(x*128),True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
    assert len(net.export())==20480
    nodes['rgb_head'].square().mean().backward()
    for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw'):
        grad=net.weight[key].grad;assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
    print('ORDERED_DISTANCE_RIM_PASS 1215constant/target0/order/directional_gradient/actual_nodes0/20KiB/four_layer_gradient',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=OrderedDistanceRimTraining,extra_sources=[Path(__file__)],evaluation=rgb_evaluate,preview=preview)
