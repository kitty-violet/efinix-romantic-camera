"""Supervise actual decoded luminance in addition to its14 latent decisions.

The reference passes known TRAIN geometry through the unchanged fixed tail.
Thus output supervision does not ask for an impossible zero-width edge or
confuse a pre-threshold sigmoid with the actual integer rendered brightness.
No reference/label/filter exists in deployment.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_revived_luma_readout import RevivedLumaReadoutTraining,driver,evaluate,preview
from .probe_b1_decoder_coverage import fragment
from .train_b1_reference_boundary_profile import reference_boundary_profile


def rendered_luma_loss(p,t,labels):
    pixel=F.smooth_l1_loss(p,t,beta=2/128)
    profile,detail=reference_boundary_profile(p.expand(-1,3,-1,-1),t.expand(-1,3,-1,-1),labels)
    return 4*pixel+4*profile,dict(rendered_luma_pixel=pixel,rendered_luma_profile_shape=detail['rgb_profile_shape'],
        rendered_luma_profile_contrast=detail['rgb_profile_contrast'])


class RenderedLumaReadoutTraining(RevivedLumaReadoutTraining):
    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        with torch.no_grad():
            # Tail is frozen; neutral chroma cannot alter this explicit Y node.
            z=F.pad(self.net.round(self.supervision_distance_truth),(0,0,0,0,0,2))
            target=fragment(self.net,z,'up1_low_pw')['up2_high_dw'][:,:1]/64
        predicted=self.nodes['up2_high_dw'][:,:1]/64
        labels=self.labels
        if not native:
            predicted=predicted[...,40:-40,40:-40];target=target[...,40:-40,40:-40];labels=labels[...,40:-40,40:-40]
        rendered,detail=rendered_luma_loss(predicted,target,labels);values.update(detail)
        return loss+rendered,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(final_rendered_luma_supervision=True,rendered_luma_reference='known shared14Y through frozen actual integer tail',
            rendered_luma_node='up2_high_dw:0',rendered_luma_weights={'pixel':4,'profile':4},
            extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)

    def preflight_extras(self):
        result=super().preflight_extras()
        assert all(k in result['values'] for k in ('rendered_luma_pixel','rendered_luma_profile_shape','rendered_luma_profile_contrast'))
        return result


def regression():
    labels=torch.zeros(1,64,96,dtype=torch.long);labels[...,48:]=1
    t=torch.zeros(1,1,64,96);t[...,48:]=1
    assert rendered_luma_loss(t,t,labels)[0]==0
    blur=F.avg_pool2d(F.pad(t,(6,6,6,6),mode='replicate'),13,1)
    rim=t.clone();rim[...,44:48]=.35
    for name,p in (('blur',blur),('rim',rim),('constant',t*0)):
        p=p.detach().requires_grad_();loss,_=rendered_luma_loss(p,t,labels);loss.backward()
        assert loss>.05 and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
    print('RENDERED_LUMA_PASS exact0/blur/rim/constant/finite_gradient',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=RenderedLumaReadoutTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
