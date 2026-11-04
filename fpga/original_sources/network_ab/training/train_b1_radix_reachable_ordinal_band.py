"""Train sharp existing radix decoder from coherent ordinal-area targets.

Keep its narrower measured boundary ramp; learn the formerly fixed spatial
layers with representable quantization steps. The RGB training target is the
same fixed decoder applied to known ordinal coverage, so internal supervision
and output supervision describe one physical rendering. Validation stays on
the unchanged original stylized target. No target labels at inference.
"""
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from .train_b1_ordinal_transition_band import TransitionBandTraining,regression,driver,preview,rgb_evaluate
from .train_b1_direct_ordinal_pairs import DirectPairB1
from .train_b1_recover_boundary_gradients import recover_gradient
from .train_b1_quantum_regions import RowScaledAdam
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits

class OrdinalBandB1(DirectPairB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name=='up2_low_dw':return recover_gradient(value,pre)
        return super().training_activation_proxy(name,value,pre,r)

class RadixReachableBandTraining(TransitionBandTraining):
    def __init__(self,args,data):
        m.AreaTraining.__init__(self,args,data);assert args.qat and not args.freeze_head
        state=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert state.get('trainable_ordinal_precision') and not state.get('monotone_pair_decoder')
        self.setup=copy.deepcopy(state['binary_coverage720']);self.net.__class__=OrdinalBandB1;self.fixed={};groups=[]
        for key,w in self.net.weight.items():
            fixed=key.startswith(('up1_low_pw','up1_high','up2_','rgb_head')) or key in ('down1__dw','down2__dw')
            b=self.net.bias[key];w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),b.detach().clone())
            else:
                scale=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(scale*256)[:,None,None,None]),dict(params=[b],lr=.008)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda')).float()
        self.palette=(levels@levels.new_tensor(self.setup['matrix']).T+levels.new_tensor(self.setup['bias'])).clamp(-128,127)/128
        self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def additional_loss(self,p,t,native):
        with torch.no_grad():
            truth=F.pad(self.net.round(F.avg_pool2d(code_bits(self.class_labels)*127,4)),(0,0,0,0,0,2))
            rendered=fragment(self.net,truth,'up1_low_dw')['rgb_head']/128
            if not native:rendered=rendered[...,40:-40,40:-40]
            assert rendered.shape==p.shape
        loss,v=super().additional_loss(p,rendered,native)
        v['reachable_vs_original_rgb_mae']=((rendered-t).abs().mean()*128).detach()
        return loss,v

    def save(self,path):
        m.AreaTraining.save(self,path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(binary_coverage720=self.setup,trainable_ordinal_precision=True,monotone_pair_decoder=False,
          trainable_spatial_prediction=['res3_conv2.dw','res3_conv2.pw','up1_low_dw'],
          ordinal_target='4x4 native area at up1_low_dw',rgb_training_target='same fixed decoder of known ordinal area',
          validation_target='unchanged original hierarchical-targets-v2',ordinal_transition_band_native=20,
          fractional_cell_loss_replaced=True,ordinal_area_class_rebalancing=False,
          extra_inference_operations=0,external_filter=False,visual_acceptance=False)
        torch.save(s,path)

def check(checkpoint):
    from .probe_b1_encoder_gauge import load
    regression();state,net=load(checkpoint);x=torch.rand(1,3,64,80)*1.9-1
    with torch.no_grad():before=net(x,all_nodes=True)
    raw=net.export();net.__class__=OrdinalBandB1;after=net(x,all_nodes=True)
    assert all(torch.equal(v,after[k]) for k,v in before.items()) and net.export()==raw
    after['rgb_head'].square().mean().backward()
    for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw'):
        g=net.weight[key].grad;assert g is not None and g.norm()>0 and torch.isfinite(g).all()
    with torch.no_grad():
        part=fragment(net,after['up1_low_dw'].detach(),'up1_low_dw')
        assert torch.equal(part['rgb_head'],after['rgb_head'])
    print('RADIX_REACHABLE_BAND_PASS unchanged_nodes/bytes/four_layer_gradient/exact_target_decoder',flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:check(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=RadixReachableBandTraining,extra_sources=[Path(__file__)],evaluation=rgb_evaluate,preview=preview)
