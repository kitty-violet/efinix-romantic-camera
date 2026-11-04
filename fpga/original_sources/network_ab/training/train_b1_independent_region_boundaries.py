"""Independent, aligned colour-region boundaries in the existing15 rows.

The3-coordinate ordered head forces thresholds to be parallel level sets.
Here every binary level learns a signed distance to its own target region.
Several levels may therefore cross the same object boundary simultaneously.
No distance transform, class lookup or extra layer is exported.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_binary_regions import BinaryTraining
from .train_b1_ordered_regions import OrderedGradientB1,evaluate,preview
from .build_b1_ordered_palette import actual_palette,bits,GROUPS
from .train_b1_flat_colour_fields import appearance
from . import train_b1_region_formation as formation
from . import train_b1_late_region_supervision as driver


def distances(mask):
    """Signed distance in180p samples; clip to a stable interior margin."""
    result=[]
    for item in mask.detach().cpu().numpy():
        fields=[]
        for channel in item:
            bit=(channel>=.5).astype(np.uint8)
            if bit.all():s=np.full(bit.shape,4,np.float32)
            elif not bit.any():s=np.full(bit.shape,-4,np.float32)
            else:s=(cv2.distanceTransform(bit,cv2.DIST_L2,5)-cv2.distanceTransform(1-bit,cv2.DIST_L2,5)).clip(-4,4)
            fields.append(s)
        result.append(np.stack(fields))
    return torch.from_numpy(np.stack(result)).to(mask.device)


class IndependentBoundaryTraining(BinaryTraining):
    checkpoint_flag='ordered_regions'
    network_class=OrderedGradientB1
    palette_function=staticmethod(actual_palette)
    bit_function=staticmethod(bits)
    num_bits=15
    appearance_loss=staticmethod(appearance)

    def __init__(self,args,data):
        super().__init__(args,data)
        # Counting arbitrary on-bits represents the same palette only because
        # the chosen balanced decoder uses equal increments within each group.
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        packed=warm.get('packed_native_count_decoder')
        if packed:
            assert packed in ('box','gauss')
            # Packed720 distributes each group's bits over several channels;
            # equal rows in the former three-channel decoder no longer apply.
            # Check the actual invariant: every permutation of the15 bits
            # must give exactly the palette colour of its three group counts.
            from .probe_b1_decoder_coverage import fragment
            was_quantized=self.net.quantized
            self.net.quantized=True
            try:
                with torch.no_grad():
                    device=self.palette.device
                    patterns=((torch.arange(32768,device=device)[:,None]>>torch.arange(15,device=device)[None])&1).float()
                    value=torch.zeros(32768,16,1,1,device=device);value[:,:15,0,0]=patterns*127
                    counts=torch.stack([patterns[:,a:z].long().sum(1) for a,z in GROUPS],1)
                    codes=counts[:,0]*25+counts[:,1]*5+counts[:,2]
                    actual=fragment(self.net,value,'up2_low_dw')['rgb_head']
                    expected=self.palette[codes,:,None,None]*128
                    error=float((actual-expected).abs().max())
                    assert error==0,('packed_decoder_count_invariance',error)
                    print('PACKED_DECODER_32768_PATTERN_PASS',flush=True)
            finally:self.net.quantized=was_quantized
        else:
            for i,(a,z) in enumerate(GROUPS):
                w=self.net.weight['up2_low_pw'][i,a:z]
                assert torch.equal(w,w[:1].expand_as(w))

    def predicted_codes(self,logits):
        count=torch.stack([(logits[a:z]>=0).long().sum(0) for a,z in GROUPS])
        return count[0]*25+count[1]*5+count[2]

    def distance_loss(self,raw,truth):
        return F.smooth_l1_loss(raw,truth,beta=1.)

    def classification_loss(self,logits,wanted):
        return F.binary_cross_entropy_with_logits(logits,wanted)

    def distance_targets(self,bits_full):
        return distances(bits_full[:,:,::4,::4])

    def additional_loss(self,p,target,native):
        key='up1_low_pw';w,b=self.net.weight[key],self.net.bias[key]
        scale=(getattr(self.net,key+'_m')/131072).float().flatten()
        qw=self.net.round(w/scale[:,None,None,None]).clamp(-127,127)*scale[:,None,None,None]
        qb=self.net.round(b/scale)*scale
        raw=(F.conv2d(self.nodes['up1_low_dw'],qw,qb)[:,:15]-63.5)/8
        bits_full=self.bit_table[self.class_labels].permute(0,3,1,2)
        truth=self.distance_targets(bits_full)
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        wanted=F.avg_pool2d(bits_full,2)
        labels=self.labels
        if not native:
            raw=raw[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
            logits=logits[...,20:-20,20:-20];wanted=wanted[...,20:-20,20:-20]
            labels=labels[...,40:-40,40:-40]
        boundary=self.distance_loss(raw,truth)
        cls=self.classification_loss(logits,wanted)
        prob=logits.sigmoid()
        order=torch.stack([F.relu(prob[:,a+1:z]-prob[:,a:z-1]).mean() for a,z in GROUPS]).mean()
        shape,values=formation.formation_loss(p,target,labels)
        loss=boundary+cls+.3*order+.5*shape
        values.update(independent_boundary_distance=boundary,independent_classification=cls,independent_order=order)
        if not native and self.step%4==0 and not getattr(self,'separate_region_projection',False):
            from .minimum_area_blocks import components,area_merge,boundary_mode
            with torch.no_grad():
                actual=self.nodes['up2_low_dw'][0,:15,...,20:-20,20:-20]
                ids=self.predicted_codes(actual-63.5).cpu().numpy()
                pal=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                rgb=pal[ids];merged,_=area_merge(components(ids),rgb,64)
                clean,_=area_merge(components(boundary_mode(merged,1)),rgb,64)
                count=np.bincount(clean.ravel()).clip(1)
                means=np.stack([np.bincount(clean.ravel(),weights=rgb[...,c].ravel(),minlength=len(count))/count for c in range(3)],1)
                chosen=((means[:,None]-pal[None].astype(float))**2).sum(2).argmin(1)[clean]
                correction=torch.from_numpy(chosen.astype(np.int64)).to(logits.device)[None]
                changed=(correction!=torch.from_numpy(ids).to(logits.device)[None]).float()[:,None]
                desired=self.bit_table[correction].permute(0,3,1,2)
            cleanup=(F.binary_cross_entropy_with_logits(logits[:1],desired,reduction='none')*changed).sum()/(15*changed.sum()).clamp_min(1)
            loss+=.5*cleanup;values.update(actual_island_correction=cleanup,actual_island_changed_fraction=changed.mean())
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True);state.pop('binary_regions',None)
        state.update(ordered_regions=True,independent_region_boundaries=True,monotonic_tied_thresholds=False,
                     coverage_layout='15independent360_rgb200',signed_distance_training_only=True)
        torch.save(state,path)


def regression():
    mask=torch.zeros(1,15,32,32);mask[:,:7,:,16:]=1
    dist=distances(mask)
    assert torch.equal(dist[:,:7],dist[:,:1].expand(-1,7,-1,-1))
    assert (dist[:,:7,:,:16]<0).all() and (dist[:,:7,:,16:]>0).all()
    # Seven independent fields can share a zero crossing. Parallel thresholds
    # on one smoothly changing coordinate must also cross intermediate codes.
    smooth=torch.arange(32).float()*127/31
    old=(smooth[None]>(torch.arange(7).float()+.5)[:,None]*127/7).sum(0)
    new=(dist[0,:7,0]>0).sum(0)
    assert old.unique().numel()==8 and new.unique().numel()==2
    print('INDEPENDENT_BOUNDARY_REGRESSION',dict(tied_transition_codes=8,independent_transition_codes=2,
        diagnostic_only=True,no_claim_of_learned_result=True),flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    driver.main(training_factory=IndependentBoundaryTraining,extra_sources=[Path(__file__),here/'train_b1_binary_regions.py',
        here/'train_b1_ordered_regions.py',here/'build_b1_ordered_palette.py',here/'train_b1_flat_colour_fields.py',
        here/'minimum_area_blocks.py',here/'train_b1_clean_regions.py',here/'train_b1_quantum_regions.py',
        here/'fit_b1_sixteen_regions.py',here/'train_b1_sixteen_regions.py',Path(formation.__file__)],
        evaluation=evaluate,preview=preview)
