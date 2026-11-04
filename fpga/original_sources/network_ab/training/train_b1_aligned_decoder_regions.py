"""Align the training derivative and count target with the fixed decoder kernel.

The earlier centre-only sigmoid proxy was correct for the centre-only decoder,
but skipped the neighbourhood used by the later half-Gaussian coefficient
change. Forward values remain the original exact integer B1 values.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_native_contrast_regions import NativeContrastTraining
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_learned_colour import ColourB1
from .train_b1_ordered_regions import evaluate, preview
from . import train_b1_late_region_supervision as driver


def decoder_logits(net,nodes):
    key='up2_low_dw';s=(getattr(net,key+'_m')/131072).float().flatten()
    w=(net.weight[key]/s[:,None,None,None]).round().clamp(-127,127)*s[:,None,None,None]
    b=(net.bias[key]/s).round()*s
    gain=w[:15].sum((1,2,3));assert torch.all(gain>0)
    x=nodes['up1_high_dw']
    pre=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),w,b,groups=x.shape[1])
    return (pre[:,:15]-63.5)/(gain[None,:,None,None]*4)


class DecoderAlignedB1(ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up2_low_dw':return value
        proxy=F.pad(127*decoder_logits(self,nodes).sigmoid(),(0,0,0,0,0,1))
        return ExactValueProxyGradient.apply(value,proxy)


class DecoderAlignedTraining(NativeContrastTraining):
    def __init__(self,args,data):
        super().__init__(args,data);self.net.__class__=DecoderAlignedB1

    def count_logits(self):return decoder_logits(self.net,self.nodes)

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(decoder_neighbourhood_aligned_gradient=True,
                     count_logits_at_actual_decoder_decision=True,
                     forward_values_unchanged_by_gradient_proxy=True)
        torch.save(state,path)


def regression(checkpoint):
    from . import train_b1_minimum_area as m
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    source=next(iter(m.load_data()['training']))
    rgb=m.t.r.read(source['path'])[:64,:96]
    x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
    with torch.no_grad():reference=net(x,all_nodes=True)
    original_bytes=net.export();net.__class__=DecoderAlignedB1
    replay=net(x,all_nodes=True)
    assert all(torch.equal(v,replay[k]) for k,v in reference.items()) and net.export()==original_bytes
    f=torch.full((1,16,9,9),63.5,requires_grad=True)
    z=decoder_logits(net,{'up1_high_dw':f})
    (127*z.sigmoid())[0,0,4,4].backward()
    grad=f.grad[0,0,3:6,3:6];weight=net.weight['up2_low_dw'][0,0].detach()
    assert torch.allclose(grad/grad.sum(),weight/weight.sum(),atol=1e-5)
    old=f.detach().clone().requires_grad_();(127*((old-63.5)/4).sigmoid())[0,0,4,4].backward()
    assert old.grad[0,0,3,4]==0 and f.grad[0,0,3,4]>0
    print('DECODER_ALIGNED_GRADIENT_PASS',dict(all_node_forward_error=0,bytes_unchanged=True,
        neighbour_gradient=float(f.grad[0,0,3,4]),old_neighbour_gradient=0),flush=True)


if __name__=='__main__':
    # Full GPU preflight in driver follows; independent CPU regression can
    # also be called before starting a run, using its explicit warm checkpoint.
    here=Path(__file__).parent
    names=['train_b1_native_contrast_regions.py','train_b1_pure_footprint_regions.py',
        'train_b1_actual_island_coverage.py','train_b1_actual_island_regions.py','train_b1_native_coverage_counts.py',
        'train_b1_count_regions.py','train_b1_truth_area_counts.py','train_b1_temporal_count_regions.py',
        'probe_b1_native_region_objective.py','minimum_area_blocks.py']
    driver.main(training_factory=DecoderAlignedTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
                evaluation=evaluate,preview=preview)
