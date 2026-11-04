"""Align14 actual luminance decisions without radix carries, fixed B1.

Training-only supervision uses the decoder's own known-interface pre-ramp
values. Unlike historical pair/15bit KL, each Y predicate has its own row and
every contribution to luminance is positive. Co/Cg remain continuous.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_luma_chroma_interfaces import LumaChromaInterfaceTraining,evaluate,driver,preview
from .probe_b1_decoder_coverage import fragment


def luma_decision_loss(prediction,target):
    assert prediction.shape==target.shape and prediction.shape[1]==14
    pp=(prediction-63.5)/3;tt=(target.detach()-63.5)/3
    terms=(F.softplus(pp,threshold=64)-F.softplus(tt,threshold=64)-(pp-tt)*torch.sigmoid(tt)).clamp_min(0)*(3/127)
    band=(target-63.5).abs()<24
    return terms.mean(),(terms*band).sum()/band.sum().clamp_min(1)


class DirectLumaInterfaceTraining(LumaChromaInterfaceTraining):
    def __init__(self,args,data):
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('direct_luminance_planes') and warm.get('nonnegative_luma_decoder')
        super().__init__(args,data)

    def additional_loss(self,p,t,native):
        value,v=super().additional_loss(p,t,native)
        truth=self.supervision_distance_truth
        with torch.no_grad():
            z=F.pad(self.net.round(truth),(0,0,0,0,0,18))
            want=fragment(self.net,z,'up1_low_dw')['up1_high_dw'][:,:14]
        pred=self.nodes['up1_high_dw'][:,:14]
        if not native:pred=pred[...,20:-20,20:-20];want=want[...,20:-20,20:-20]
        full,band=luma_decision_loss(pred,want)
        # Replace the old16full+8band SDF emphasis, not stack an unbounded
        # list of competing losses. Chroma/rank/source consistency unchanged.
        value+=self.args.classification*(-6*v['ordered_distance_full']-3*v['ordered_distance_band']+8*full+4*band)
        v.update(luminance_decision_kl=full,luminance_decision_band_kl=band)
        return value,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(direct_luminance_planes=True,nonnegative_luma_decoder=True,hard_luminance_decisions=14,
            independent_chroma_decisions=0,luminance_render_supervision='proper Bernoulli KL at14 up1_high_dw rows',
            rendering_decision_temperature=3.,subtractive_luminance_radix=False,extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)


def regression():
    t=torch.linspace(0,127,128,dtype=torch.float64)[None,None,None].expand(1,14,3,128).clone()
    p=t.clone().requires_grad_();loss=sum(luma_decision_loss(p,t));assert loss==0
    loss.backward();assert p.grad.abs().max()<1e-15
    wrong=(127-t).clone().requires_grad_();loss=sum(luma_decision_loss(wrong,t));loss.backward()
    assert loss>.1 and wrong.grad[...,0].min()>0 and wrong.grad[...,-1].max()<0 and torch.isfinite(wrong.grad).all()
    p=(t[:,:,:1,::16]+1.3).clone().requires_grad_();target=t[:,:,:1,::16]
    assert torch.autograd.gradcheck(lambda z:sum(luma_decision_loss(z,target)),(p,),atol=1e-5,rtol=1e-3)
    print('DIRECT_LUMA_DECISION_PASS target0/zero_gradient/wrong_direction/gradcheck',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=DirectLumaInterfaceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
