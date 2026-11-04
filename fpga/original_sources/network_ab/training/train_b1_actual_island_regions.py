"""Supervise actual native720 integer regions, including wrong small islands.

All connected-component analysis is training-only. The exported graph remains
B1. The loss never changes a correct colour to a neighbour's guessed colour.
"""
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_seven_scene_batch import SevenSceneBatchTraining
from .train_b1_count_regions import count_loss
from .train_b1_temporal_count_regions import revised_classes
from .fit_b1_sixteen_regions import region_class_map
from .minimum_area_blocks import components, boundary_mask
from .probe_b1_native_region_objective import region_variance
from .train_b1_ordered_regions import evaluate, preview
from . import train_b1_late_region_supervision as driver


def wrong_island_weights(actual,wanted,valid,minimum=256):
    """Equal weight for each erroneous small component in target interiors."""
    actual_code=(actual[:,0]*25+actual[:,1]*5+actual[:,2]).detach().cpu().numpy().astype(np.int32)
    good=valid.detach().cpu().numpy()
    wanted_code=(wanted[:,0]*25+wanted[:,1]*5+wanted[:,2]).detach().cpu().numpy().astype(np.int32)
    results=[];records=[]
    for code,truth,mask in zip(actual_code,wanted_code,good):
        lab=components(code);sizes=np.bincount(lab.ravel())
        # A component crossing the image/crop boundary has unknown full area.
        touched=np.unique(np.concatenate((lab[0],lab[-1],lab[:,0],lab[:,-1])))
        eligible=sizes<minimum;eligible[touched]=False
        wrong=(code!=truth)&mask&eligible[lab]
        number=np.bincount(lab[wrong],minlength=len(sizes));active=number>0
        weight=wrong/np.maximum(number[lab],1)
        weight=weight/max(1,int(active.sum()))
        results.append(torch.tensor(weight,dtype=actual.dtype,device=actual.device))
        records.append(dict(small_wrong_components=int(active.sum()),corrective_pixels=int(wrong.sum())))
    return torch.stack(results),records


def actual_region_losses(actual,wanted,labels,valid):
    assert actual.shape==wanted.shape and actual.shape[1]==3
    pixel=F.smooth_l1_loss(actual,wanted,beta=.5)
    # Correct coordinates anchor the region mean but receive no grouping
    # gradient. Otherwise variance minimization also drags correct pixels
    # toward an erroneous island, opposing the intended correction.
    anchored=torch.where(actual==wanted,actual.detach(),actual)
    grouping=region_variance(anchored,labels,valid)
    with torch.no_grad():weights,stats=wrong_island_weights(actual,wanted,valid)
    error=F.smooth_l1_loss(actual,wanted,beta=.5,reduction='none').mean(1)
    islands=(error*weights).sum((-1,-2)).mean()
    return pixel+.5*grouping+islands,dict(native_integer_colour=pixel,native_integer_region_variance=grouping,
        native_integer_wrong_islands=islands,small_wrong_components=actual.new_tensor(sum(r['small_wrong_components'] for r in stats)),
        corrected_pixel_fraction=(weights>0).float().mean())


class ActualIslandRegionTraining(SevenSceneBatchTraining):
    def __init__(self,args,data):
        super().__init__(args,data);assert args.qat
        self.full_targets={}
        for key,x,_ in self.fit_cache:
            row=self.by_id[key];entry=self.class_overrides.get(key)
            codes=revised_classes(entry['path'],entry['sha256']) if entry else region_class_map(row['target'],row['region'],self.pk)
            lab=components(codes);valid=cv2.dilate(boundary_mask(codes),np.ones((7,7),np.uint8))==0
            c=torch.tensor(codes.astype('int64'),device=x.device)[None]
            wanted=torch.stack((c//25,(c//5)%5,c%5),1).float()
            self.full_targets[key]=(wanted,torch.tensor(lab.astype('int64'),device=x.device)[None],torch.tensor(valid,device=x.device)[None])

    def update(self,native=None):
        native=True if native is None else native;started=time.perf_counter()
        self.opt.zero_grad(set_to_none=True);parts=[];samples=[]
        if native:
            batches=[(x,truth,self.full_targets[key],[key]) for key,x,truth in self.fit_cache]
        else:
            # Actual batch2 crop backward, not a substitute native-only preflight.
            chosen=self.fit_cache[:2];x=torch.cat([v[1][...,192:512,384:704] for v in chosen])
            truth=[torch.cat([v[2][g][...,116:236,212:332] for v in chosen]) for g in range(3)]
            full=tuple(torch.cat([self.full_targets[v[0]][g][...,192:512,384:704] for v in chosen]) for g in range(3))
            batches=[(x,truth,full,[v[0] for v in chosen])]
        for x,truth,(wanted,labels,valid),ids in batches:
            nodes=self.net(x,all_nodes=True);logits=(nodes['up1_high_dw'][:,:15]-63.5)/4
            coords=nodes['up2_high_dw'][:,:3]
            if not native:
                logits=logits[...,20:-20,20:-20];coords=coords[...,40:-40,40:-40]
                wanted=wanted[...,40:-40,40:-40];labels=labels[...,40:-40,40:-40];valid=valid[...,40:-40,40:-40]
            ce=count_loss(logits,truth);extra,values=actual_region_losses(coords,wanted,labels,valid)
            loss=(ce+extra)/len(batches);loss.backward()
            parts.append(dict(count_ce=float(ce.detach()),**{k:float(v.detach()) for k,v in values.items()}))
            for key in ids:samples.append(dict(id=key,xy=[0,0] if native else [384,192],wh=[1280,720] if native else [320,320],flip=False))
            del nodes,coords,logits,loss
        parameters=[p for p in self.net.parameters() if p.requires_grad]
        grad=torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True)
        assert grad>0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        for group,lr in zip(self.opt.param_groups,self.base_lrs):group['lr']=lr
        self.opt.step()
        with torch.no_grad():
            for key,w in self.net.weight.items():
                if w.requires_grad:
                    limit=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
                    w.copy_(w.maximum(-limit).minimum(limit))
            for key,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[key]) and torch.equal(b,self.net.bias[key])
        self.step+=1;torch.cuda.synchronize()
        values={k:sum(r[k] for r in parts)/len(parts) for k in parts[0]}
        return dict(step=self.step,loss=values['count_ce']+values['native_integer_colour']+.5*values['native_integer_region_variance']+values['native_integer_wrong_islands'],
            values=values,grad=float(grad),samples=samples,native720=native,temporal_pair=False,seconds=time.perf_counter()-started)

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(actual_integer_native_region_objective=True,area_minimum720=256,
            objective='count CE + actual720 coordinate Huber + .5 labelled-region variance + equal-wrong-small-component correction',
            small_island_target='immutable TRAIN truth; exclude correct pixels and target boundary band',
            minimum_area_inference_guarantee=False,visual_acceptance=False)
        torch.save(s,path)

    @torch.no_grad()
    def report_training_fit(self):
        result=super().report_training_fit();records=[]
        for key,x,_ in self.fit_cache:
            nodes=self.net(x,all_nodes=True);actual=nodes['up2_high_dw'][:,:3]
            wanted,_,valid=self.full_targets[key]
            code=(actual[0,0]*25+actual[0,1]*5+actual[0,2]).cpu().numpy().astype(np.int32)
            lab=components(code);size=np.bincount(lab.ravel());exact=(actual==wanted).all(1)
            records.append(dict(id=key,final720_colour_accuracy=float(exact.float().mean()),
                interior_colour_accuracy=float(exact[valid].float().mean()),
                small_area_fraction=float(size[size<256].sum()/size.sum()),regions=len(size)))
        result['actual_final720']=records
        print('ACTUAL_ISLAND_NATIVE_TRAIN_FIT',self.step,records,flush=True)
        return result


def regression():
    wanted=torch.zeros(1,3,64,96);wanted[:,0,:,48:]=4
    labels=(wanted[:,0]>0).long();valid=torch.ones_like(labels,dtype=torch.bool)
    actual=wanted.clone();actual[:,0,20:23,20:23]=1;actual[:,0,40:43,70:73]=3
    actual.requires_grad_(True);weights,stats=wrong_island_weights(actual,wanted,valid)
    assert stats[0]['small_wrong_components']==2 and abs(float(weights.sum())-1)<1e-6
    error=F.smooth_l1_loss(actual,wanted,beta=.5,reduction='none').mean(1)
    (error*weights).sum().backward()
    assert (actual.grad[:,0,20:23,20:23]>0).all() and (actual.grad[:,0,40:43,70:73]<0).all()
    assert not actual.grad[:,:,10,10].any()
    probe=actual.detach().clone().requires_grad_(True)
    combined,_=actual_region_losses(probe,wanted,labels,valid)
    combined.backward()
    assert not probe.grad[probe.detach()==wanted].any()
    # The target need not choose the neighbouring predicted colour.
    wrong=actual.detach().clone();wrong[:,0,40:43,70:73]=2
    clean=wrong_island_weights(wrong,wanted,valid)[0]
    assert clean[0,41,71]>0
    w,_=wrong_island_weights(wanted,wanted,valid);assert w.sum()==0
    border=wanted.clone();border[:,0,0:3,20:23]=1
    assert wrong_island_weights(border,wanted,valid)[0].sum()==0
    result,values=actual_region_losses(wanted.requires_grad_(True),wanted.detach(),labels,valid)
    assert float(result.detach())==0;result.backward()
    print('ACTUAL_NATIVE_ISLAND_REGRESSION_PASS protected_correct_pixels / truth_direction / border / empty',flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    names=['train_b1_seven_scene_batch.py','train_b1_seven_scene_fit.py','train_b1_native_coverage_counts.py',
        'train_b1_truth_area_counts.py','train_b1_temporal_count_regions.py','train_b1_count_regions.py',
        'probe_b1_native_region_objective.py','train_b1_ordered_regions.py','minimum_area_blocks.py']
    driver.main(training_factory=ActualIslandRegionTraining,extra_sources=[Path(__file__)]+[here/n for n in names],evaluation=evaluate,preview=preview)
