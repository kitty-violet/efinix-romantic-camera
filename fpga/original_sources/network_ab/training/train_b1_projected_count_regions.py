"""Focus learning on small erroneous connected regions in the actual codes.

Reuses earlier training-only area projection, with the current exact1215-code
ternary representation. A colour guard keeps corrections consistent with the
original target. Inference is still one unmodified B1, with no area algorithm.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_real_region_texture import RealRegionTextureTraining
from .train_b1_region_tail_flatness import evaluate as base_evaluate,preview
from .train_b1_ternary_regions import log_states,field_pre,count_distribution
from .probe_b1_projected_ternary_geometry import clean_codes
from . import train_b1_late_region_supervision as driver
from .minimum_area_blocks import components,boundary_mode
from . import train_b1_minimum_area as m


def selected_count_loss(pre,desired,mask):
    terms=[];lp=log_states(pre)
    for (lo,hi),target in zip(((0,7),(7,11),(11,15)),(desired//81,desired//9%9,desired%9)):
        probability=count_distribution(lp[:,lo:hi])
        loss=-probability.gather(1,target[:,None])[:,0]
        terms.append((loss*mask).sum()/mask.sum().clamp_min(1))
    return torch.stack(terms).mean()


def region_geometry(codes):
    lab=components(codes);areas=np.bincount(lab.ravel())
    return dict(regions360=len(areas),small_region_pixels=float(areas[areas<64].sum()/areas.sum()),
                categorical_vote_changes=float(np.mean(boundary_mode(lab,1)!=lab)))


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        z=net(x,all_nodes=True)['up2_low_dw'][0,:15]
        codes=(81*z[:7].sum(0)+9*z[7:11].sum(0)+z[11:15].sum(0)).long().cpu().numpy()
        record.update(region_geometry(codes))
    for key in ('regions360','small_region_pixels','categorical_vote_changes'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


class ProjectedCountTraining(RealRegionTextureTraining):
    def preflight_extras(self):
        # The common preflight's native update is step0, whereas the new loss
        # runs on native step3mod4. Exercise its real CUDA path explicitly.
        while self.step%4!=3:self.update(True)
        result=self.update(True)
        assert result['values']['projected_guarded_fraction']>0
        assert result['values']['projected_small_region_ce']>0
        return result

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        if not native or self.step%4!=3:return loss,values
        z=self.nodes['up2_low_dw'][:1,:15]
        h,w=z.shape[-2:];side=192
        yy=self.rng.randrange(12,h-side-12+1);xx=self.rng.randrange(12,w-side-12+1)
        crop=z[...,yy:yy+side,xx:xx+side]
        with torch.no_grad():
            ids=(crop[:,:7].sum(1)*81+crop[:,7:11].sum(1)*9+crop[:,11:15].sum(1)).long()
            pal=np.asarray(self.pk,dtype=np.float64)
            cleaned,report=clean_codes(ids[0].cpu().numpy(),pal,64)
            desired=torch.from_numpy(cleaned)[None].to(p.device)
            target=F.avg_pool2d(t[:1],2)[...,yy:yy+side,xx:xx+side].permute(0,2,3,1)
            old_error=(self.palette[ids]-target).abs().mean(-1)
            new_error=(self.palette[desired]-target).abs().mean(-1)
            changed=desired!=ids
            valid=changed&(new_error<=old_error+4/128)
            # A component truncated by the diagnostic crop must not teach a
            # false small-area label at the crop boundary.
            valid[:,:16]=False;valid[:,-16:]=False;valid[:,:,:16]=False;valid[:,:,-16:]=False
        logits=field_pre(self.net,self.nodes)[:1,...,yy:yy+side,xx:xx+side]
        projected=selected_count_loss(logits,desired,valid.float())
        values.update(projected_small_region_ce=projected,projected_change_fraction=changed.float().mean(),
                      projected_guarded_fraction=valid.float().mean(),
                      projected_original_regions=p.new_tensor(report['original_regions']),
                      projected_clean_regions=p.new_tensor(report['projected_regions']))
        return loss+.75*projected,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(projected_count_region_training=dict(minimum_area360=64,colour_guard_rgb=4,weight=.75),
                     projection_training_only=True,minimum_area_inference_guarantee=False,visual_acceptance=False)
        torch.save(state,path)


def regression():
    from .probe_b1_ternary_field_targets import canonical_states
    ids=torch.full((1,64,64),608,dtype=torch.long)
    pre=canonical_states(ids).clone();pre[:,:,20:24,20:24]=2;pre.requires_grad_()
    mask=torch.zeros(1,64,64);mask[:,20:24,20:24]=1
    loss=selected_count_loss(pre,ids,mask);loss.backward()
    assert torch.isfinite(pre.grad).all() and pre.grad[:,:,20:24,20:24].abs().sum()>0
    assert pre.grad[:,:,:18].abs().sum()==0
    empty=selected_count_loss(pre.detach(),ids,mask*0);assert empty==0
    palette=np.zeros((1215,3),np.float64);palette[:]=[60,70,80];palette[1214]=[220,230,240]
    test=np.full((64,64),608,np.int64);test[20:24,20:24]=1214
    cleaned,report=clean_codes(test,palette,64)
    assert report['original_regions']==2 and report['projected_regions']==1 and np.unique(cleaned).size==1
    print('PROJECTED_COUNT_REGRESSION_PASS',float(loss.detach()),report,flush=True)


if __name__=='__main__':
    import sys
    regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=ProjectedCountTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
