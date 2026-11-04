"""Ablate target-label-guided input contrast changes in source-to-contour fit.

Same centered optimizer, target distance, sampling and reliable source temporal
loss. Inputs stay the real source pixels; do not paint known target boundaries
into training inputs. This is a controlled distribution hypothesis, not proof
that the previous augmentation caused all halos.
"""
from pathlib import Path
import numpy as np
import torch
from .train_b1_centered_distance import CenteredDistanceTraining,evaluate,driver,preview
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_ternary_regions import region_codes
from .train_b1_real_region_texture import align_pair
from . import train_b1_minimum_area as m


class RawSourceDistanceTraining(CenteredDistanceTraining):
    def consume_contrast_draws(self,labels):
        # Keep the same sampler RNG progression as the augmentation control.
        # Discard only the contrast operation, not subsequent source choices.
        for lab in labels:
            for _ in range(int(lab.unique().numel())):self.rng.uniform(0,1.6)

    def batch_style(self,native):
        x,t,specs,flow,valid=m.AreaTraining.batch_style(self,native);codes=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
            a=region_codes(row['target'],row['region'],self.pk);xx,yy=spec['xy'];w,h=spec['wh']
            a=a[yy:yy+h,xx:xx+w]
            if spec['flip']:a=a[:,::-1]
            codes.append(torch.from_numpy(np.ascontiguousarray(a.astype(np.int64))).to(x.device))
            # Explicitly check the ablation: the integer input is the saved
            # source crop/flip, not a contrast intervention using target labels.
            source=m.t.r.read(row['path'])[yy:yy+h,xx:xx+w]
            if spec['flip']:source=source[:,::-1]
            wanted=torch.from_numpy(np.ascontiguousarray(source.transpose(2,0,1))).to(x.device).float()/128-1
            assert torch.equal(x[len(codes)-1],wanted)
        self.class_labels=torch.stack(codes);self.motion_fraction=x.new_zeros(())
        if flow is not None:t,self.labels,self.class_labels,self.motion_fraction=align_pair(x,t,self.labels,self.class_labels,flow,valid)
        elif self.step%4 in (1,2):self.consume_contrast_draws(self.labels)
        return x,t,specs,flow,valid

    def input_stability_loss(self,x,pred,native):
        loss,values=CleanRegionTraining.input_stability_loss(self,x,pred,native)
        if native and self.step%8==3:self.consume_contrast_draws(self.labels[:1])
        return loss,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(real_region_texture_interventions=False,label_guided_input_changes=False,
            texture_invariance_loss=False,real_source_input_assertion=True,
            augmentation_rng_draws_consumed_for_control=True,
            source_motion_aligned_pair_targets=True,visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    torch.set_num_threads(2)
    driver.main(training_factory=RawSourceDistanceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
