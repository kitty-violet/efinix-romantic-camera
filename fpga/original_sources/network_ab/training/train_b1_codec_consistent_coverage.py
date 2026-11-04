"""One consistent target for the fixed decoder's physical coverage output.

Do not ask the pre-gain score to equal a post-gain coverage fraction. Do not
also force average colour/ordinal occupancy before the spatial decoder at mixed
boundaries. Supervise the actual360p clipped coverage against2x2 native areas;
only deep homogeneous cells receive early ordinal supervision. The real720 RGB
shape/profile/flatness and source-motion losses remain independent requirements.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_ordered_boundary_fields import OrderedBoundaryTraining,rank_loss,driver,preview
from .train_b1_complete_boundary_gradients import CompleteBoundaryGradientB1
from .train_b1_symmetric_binary_coverage import evaluate as rgb_evaluate
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_region_tail_flatness import region_tail_flatness
from .train_b1_packed720_local_rims import local_rim_loss
from .train_b1_boundary_profile import boundary_profile
from .train_b1_contour_coherence import contour_coherence
from .build_b1_thirty_binary_regions import code_bits
from .b1_binary_layout import fields_for_layout
from .train_b1_ternary_regions import region_codes
from . import train_b1_minimum_area as m

def reachable_coverage(gain,device):
    assert gain%2==1 and 1<=gain<=127
    raw=(torch.arange(128,device=device,dtype=torch.float64)*gain+127*(1-gain)//2)*131071/131072
    return (raw.sign()*torch.floor(raw.abs()+.5)).clamp(0,127).float()

def coverage_loss(field,truth,gain=7):
    error=(field[:,:15]/127-truth).abs()
    # Integer pre-gain scores cannot produce every output byte. For gain7,
    # half-coverage63.5 has equally near legal outputs60/67, not64. Avoid
    # gradients which keep pushing between these equally good representations.
    quarter=torch.arange(5,device=field.device)*127/4
    tolerance=(quarter[:,None]-reachable_coverage(gain,field.device)[None]).abs().amin(1)/127
    index=(truth*4).long();assert torch.equal(index.float(),truth*4)
    residual=F.relu(error-tolerance[index]-1e-7)
    terms=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.05,reduction='none')
    mixed=(truth>0)&(truth<1)
    return terms.mean(),(terms*mixed).sum()/mixed.sum().clamp_min(1)

def core_ordinal_loss(hinge,code):
    low=-F.max_pool2d(-code[:,None].float(),4,4);high=F.max_pool2d(code[:,None].float(),4,4)
    # A full3x3 of180p cells must have the same code, not only the central4x4.
    local_low=-F.max_pool2d(-low,3,1,1);local_high=F.max_pool2d(high,3,1,1)
    core=(local_low==local_high).float()
    target=F.avg_pool2d(code_bits(code),4);error=F.smooth_l1_loss(hinge[:,:30]/127,target,beta=.1,reduction='none')
    return (error*core).sum()/(30*core.sum()).clamp_min(1)

class CodecCoverageTraining(OrderedBoundaryTraining):
    def __init__(self,args,data):
        super().__init__(args,data);self.net.__class__=CompleteBoundaryGradientB1
        gain=self.setup['gain'];key='up2_low_dw';expected=self.net.weight[key].new_zeros(15,1,3,3)
        expected[:,0,1,1]=gain*131071/131072
        assert torch.equal(self.net.weight[key][:15],expected)
        assert (getattr(self.net,key+'_m')[:,:15]==131071).all()
        assert (self.net.bias[key][:15]==127*(1-gain)//2*131071/131072).all()

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        region,v=joint_region_loss(p,t,labels);tail=region_tail_flatness(p,labels)
        rim=local_rim_loss(p,t,labels);profile,terms=boundary_profile(p,t,labels)
        if native:
            y=self.rng.randrange(p.shape[-2]-384+1);x=self.rng.randrange(p.shape[-1]-384+1)
            curve=contour_coherence(p[...,y:y+384,x:x+384],t[...,y:y+384,x:x+384])
        else:curve=contour_coherence(p,t)
        code=self.class_labels;field=self.nodes['up2_low_dw'];hinge=self.nodes['res3_conv1.pw']
        truth=F.avg_pool2d(fields_for_layout(code,self.setup),2)
        if not native:
            field=field[...,20:-20,20:-20];truth=truth[...,20:-20,20:-20]
            code=code[...,40:-40,40:-40];hinge=hinge[...,10:-10,10:-10]
        full,mixed=coverage_loss(field,truth,self.setup['gain']);core=core_ordinal_loss(hinge,code);rank=rank_loss(hinge)
        v.update(terms);v.update(region_tail_flatness=tail,mean_centred_local_bright_rim=rim,
          short_contour_turn_excess=curve,decoded_coverage=full,mixed_decoded_coverage=mixed,
          core_ordinal=core,ordinal_rank_violation=rank,motion_aligned_target_fraction=self.motion_fraction)
        rgb=.7*region+4*v['region_flatness']+v['adjacent_region_contrast']+4*tail+4*curve+4*rim+4*profile
        return rgb+self.args.classification*(4*full+2*mixed+2*core)+10*rank,v

    def preflight_extras(self):
        while self.step%4:self.update(True)
        result=self.update(False);assert result['temporal_pair'] and 'tiny_input_stability' in result['values']
        assert 'mixed_decoded_coverage' in result['values'];return result

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        for key in ('predecision_region_supervision','coordinate_target_weight','coordinate_target_scope',
                    'post_vote_ordinal_occupancy_weight','mixed_boundary_coverage_weight','predecision_units'):
            s.pop(key,None)
        s.update(codec_consistent_coverage=True,coverage_supervision_node='up2_low_dw after gain and clamp',
          no_mixed_predecoder_target=True,early_ordinal_scope='homogeneous3x3 of4x4 native cells',
          coverage_tolerance='nearest reachable actual integer ramp value, equal minima not penalized',
          boundary_occupancy_training=False,exact_complement_training=False,
          complete_boundary_gradient_proxy=['res3_conv1.pw','up1_low_dw','up2_low_dw'],visual_acceptance=False)
        torch.save(s,path)

@torch.inference_mode()
def evaluate(net,rows):
    result=rgb_evaluate(net,rows)
    for rec,row in zip(result['records'],rows):
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
        code=torch.from_numpy(region_codes(row['target'],row['region'],net.confidence_palette_key).astype('int64'))[None].cuda()
        truth=F.avg_pool2d(fields_for_layout(code,net.binary_layout_setup),2)
        full,mixed=coverage_loss(nodes['up2_low_dw'],truth,net.binary_layout_setup['gain'])
        rec.update(decoded_coverage=float(full),mixed_decoded_coverage=float(mixed),ordinal_rank_violation=float(rank_loss(nodes['res3_conv1.pw'])))
    for k in ('decoded_coverage','mixed_decoded_coverage','ordinal_rank_violation'):result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    result['mean']['score']+=4*result['mean']['decoded_coverage']+2*result['mean']['mixed_decoded_coverage']+10*result['mean']['ordinal_rank_violation']
    return result

def regression():
    target=torch.tensor([0.,.25,.5,.75,1.])[None,None,None].expand(1,15,1,5)
    reachable=reachable_coverage(7,'cpu');index=(reachable[:,None]-(target[0,0,0]*127)[None]).abs().argmin(0)
    legal=reachable[index][None,None,None].expand_as(target);a,b=coverage_loss(legal,target);assert a==0 and b==0
    alternate=legal.clone();alternate[...,2]=67;a,b=coverage_loss(alternate,target);assert a==0 and b==0
    wrongly_applied_pre=(target*127-63.5)*7+63.5
    wrong=wrongly_applied_pre.clamp(0,127);a,b=coverage_loss(wrong,target);assert b>.1
    inverse=63.5+(target*127-63.5)/7;decoded=reachable[inverse.round().long()]
    a,b=coverage_loss(decoded,target);assert a==0 and b==0
    code=torch.zeros(1,32,48,dtype=torch.long);code[...,20:]=8*81
    truth=F.avg_pool2d(code_bits(code),4)*127;assert core_ordinal_loss(truth,code)==0
    broken=truth.clone();broken[...,4:6]=42;assert core_ordinal_loss(broken,code)==0
    leaf=wrong.detach().requires_grad_();a,b=coverage_loss(leaf,target);(a+b).backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad[...,1].mean()<0 and leaf.grad[...,3].mean()>0
    print('CODEC_COVERAGE_REGRESSION_PASS actual_reachable0/equal_minima0/wrong_pre_target_rejected/inverse_ramp_correct/core_scope/directional_gradient',flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:driver.main(training_factory=CodecCoverageTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
