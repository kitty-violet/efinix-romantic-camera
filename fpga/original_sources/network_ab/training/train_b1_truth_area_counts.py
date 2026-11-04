"""Correct actual fragmented regions toward audited labels, never self colours.

The former modal self-projection changed correct colours into wrong colours.
Cleanup is now only a location detector: supervision always uses the immutable
region label, and already-correct pixels receive no corrective term. It adds
no deployed operator and does not assert a hard inference-area guarantee.
"""
from pathlib import Path
import numpy as np
import torch
from .train_b1_temporal_count_regions import TemporalCountTraining
from .train_b1_count_regions import count_targets,count_loss,regression as count_regression
from .train_b1_decisive_regions import modal_region_cleanup
from .train_b1_ordered_regions import evaluate,preview
from . import train_b1_late_region_supervision as driver


def correction_mask(actual,clean,truth):
    pure=torch.stack([v.amax(1)>.999 for v in truth]).all(0)
    wanted=truth[0].argmax(1)*25+truth[1].argmax(1)*5+truth[2].argmax(1)
    return ((actual!=clean)&(actual!=wanted)&pure)[:,None].float()


class TruthAreaCountTraining(TemporalCountTraining):
    def additional_loss(self,p,target,native):
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        truth=count_targets(self.class_labels)
        if not native:
            logits=logits[...,20:-20,20:-20];truth=[v[...,20:-20,20:-20] for v in truth]
        supervised=count_loss(logits,truth)
        values=dict(actual_palette_count_ce=supervised)
        if native:return supervised,values
        with torch.no_grad():
            palette=(self.palette.detach().cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
            actual=[];cleaned=[]
            for item in self.nodes['up2_low_dw'][:,:15]:
                code=self.predicted_codes(item-63.5).cpu().numpy()
                clean,_=modal_region_cleanup(code,palette,64,2)
                actual.append(torch.tensor(code[20:-20,20:-20],device=p.device))
                cleaned.append(torch.tensor(clean[20:-20,20:-20],device=p.device))
            actual=torch.stack(actual);cleaned=torch.stack(cleaned)
            mask=correction_mask(actual,cleaned,truth)
        correction=count_loss(logits,truth,mask,balanced=False)
        values.update(truth_anchored_area_correction=correction,corrected_wrong_fraction=mask.mean(),
            protected_changed_fraction=((actual!=cleaned)[:,None].float()*(1-mask)).mean())
        return supervised+.3*correction,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(self_colour_pseudolabels=False,area_correction_labels='audited region truth only',
            correct_colour_protected=True,objective='count CE1 + wrong-fragment truth count CE.3 + existing stability',
            minimum_area_inference_guarantee=False)
        torch.save(s,path)


def regression():
    # The old cleanup changes both positions to colour26. First position was
    # correct already; the second is wrong and must learn truth52, not26.
    actual=torch.tensor([[[0,0]]]);clean=torch.tensor([[[26,26]]])
    truth=count_targets(torch.tensor([[[0,52]]]),1)
    mask=correction_mask(actual,clean,truth)
    assert torch.equal(mask,torch.tensor([[[[0.,1.]]]]))
    z=torch.zeros(1,15,1,2,requires_grad=True)
    loss=count_loss(z,truth,mask,balanced=False);loss.backward()
    assert torch.count_nonzero(z.grad[:,:,:,0])==0
    assert torch.count_nonzero(z.grad[:,:,:,1])>0
    # All-correct / no-correction cases remain finite and exactly inert.
    for a,b in [(torch.tensor([[[0,52]]]),clean),(actual,actual)]:
        z=torch.randn(1,15,1,2,requires_grad=True)
        loss=count_loss(z,truth,correction_mask(a,b,truth),balanced=False)
        assert float(loss.detach())==0;loss.backward();assert torch.count_nonzero(z.grad)==0
    # Mixed boundary footprints must not receive a forced hard-area correction.
    mixed=count_targets(torch.tensor([[[0,52],[0,52]]]),2)
    assert not correction_mask(torch.zeros(1,1,1,dtype=torch.long),torch.ones(1,1,1,dtype=torch.long),mixed).any()
    print('TRUTH_AREA_REGRESSION correct protected / wrong learns truth / empty finite / mixed excluded PASS',flush=True)


if __name__=='__main__':
    count_regression();regression();here=Path(__file__).parent
    names=['train_b1_temporal_count_regions.py','train_b1_count_regions.py','prepare_b1_temporal_classes.py',
        'train_b1_native_distance_boundaries.py','train_b1_decisive_regions.py','probe_b1_independent_decoder.py',
        'train_b1_independent_region_boundaries.py','train_b1_balanced_region_boundaries.py','train_b1_binary_regions.py',
        'train_b1_ordered_regions.py','build_b1_ordered_palette.py','minimum_area_blocks.py',
        'train_b1_quantum_regions.py','train_b1_region_formation.py']
    driver.main(training_factory=TruthAreaCountTraining,extra_sources=[Path(__file__)]+[here/n for n in names],
        evaluation=evaluate,preview=preview)
