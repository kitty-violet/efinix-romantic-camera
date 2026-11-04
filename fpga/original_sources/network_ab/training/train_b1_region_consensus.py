"""Constrain classification fields, not just already-discrete RGB output.

Within teacher-region interiors, centred class logits must agree. Every crop
also receives connected-component cleanup supervision (training only). Exact
integer inference and the existing20KiB B1 graph are unchanged.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_tournament_regions as parent
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components,area_merge,boundary_mode


def region_consensus(scores,labels):
    labels=F.interpolate(labels[:,None].float(),size=scores.shape[-2:],mode='nearest')[:,0].long()
    # Ignore the first coarse pixel next to a target boundary. Do not force
    # different regions to have equal features, or penalize a sharp edge.
    hi=F.max_pool2d(labels[:,None].float(),3,1,1)
    lo=-F.max_pool2d(-labels[:,None].float(),3,1,1)
    valid=(hi==lo).float();z=scores-scores.mean(1,keepdim=True)
    losses=[]
    for field,lab,mask in zip(z,labels,valid):
        index=lab.flatten();n=int(index.max())+1;weight=mask.flatten()
        count=weight.new_zeros(n).scatter_add(0,index,weight).clamp_min(1)
        ids=index[None].expand(field.shape[0],-1)
        mean=field.new_zeros(field.shape[0],n).scatter_add(1,ids,field.flatten(1)*weight)/count[None]
        losses.append(((field.flatten(1)-mean[:,index]).abs()*weight).sum()/(field.shape[0]*weight.sum()).clamp_min(1))
    return torch.stack(losses).mean()


class ConsensusTraining(parent.TournamentTraining):
    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        score=self.nodes['res1_conv2.dw'][:,:16]
        if not native:score=score[...,10:-10,10:-10]
        logits=self.score_logits(score)
        uniform=region_consensus(logits,labels);loss+=2*uniform
        values['whole_region_score_consensus']=uniform
        if not native:
            corrections=[];changes=[]
            pal=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
            with torch.no_grad():
                for field in logits:
                    ids=field.argmax(0).cpu().numpy();rgb=pal[ids]
                    merged,_=area_merge(components(ids),rgb,16)
                    clean,_=area_merge(components(boundary_mode(merged,1)),rgb,16)
                    count=np.bincount(clean.ravel()).clip(1)
                    means=np.stack([np.bincount(clean.ravel(),weights=rgb[...,c].ravel())/count for c in range(3)],1)
                    result=((means[:,None]-pal[None].astype(float))**2).sum(2).argmin(1)[clean]
                    corrections.append(torch.from_numpy(result.astype(np.int64)).cuda())
                    changes.append(torch.from_numpy(result!=ids).cuda())
                correction=torch.stack(corrections);changed=torch.stack(changes).float()
            ce=F.cross_entropy(logits,correction,reduction='none')
            cleanup=(ce*changed).sum()/changed.sum().clamp_min(1)
            loss+=1.2*cleanup
            values.update(every_crop_area_cleanup=cleanup,changed_small_block_fraction=changed.mean())
        return loss,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(whole_region_score_consensus=True,area_cleanup_every_crop=True)
        torch.save(s,path)


def check():
    labels=torch.zeros(1,64,64,dtype=torch.long,device='cuda');labels[:,:,32:]=1
    scores=torch.zeros(1,16,16,16,device='cuda');scores[:,0,:,:8]=8;scores[:,1,:,8:]=8
    exact=region_consensus(scores,labels)
    generator=torch.Generator(device='cuda').manual_seed(1901)
    noisy=scores+torch.randn(scores.shape,device='cuda',generator=generator)
    noise=region_consensus(noisy,labels)
    # Common class offsets do not alter a decision and must not be penalized.
    shifted=scores+torch.randn((1,1,16,16),device='cuda',generator=generator)
    invariant=region_consensus(shifted,labels)
    assert exact<1e-5 and invariant<1e-5 and noise>.3
    print('CONSENSUS_REGRESSION',float(exact),float(noise),float(invariant),flush=True)


if __name__=='__main__':
    check();here=Path(__file__).parent
    driver.main(training_factory=ConsensusTraining,extra_sources=[Path(__file__),Path(parent.__file__),
        here/'train_b1_sixteen_regions.py',here/'train_b1_winner_surrogate.py',here/'train_b1_quantum_regions.py',
        here/'train_b1_region_formation.py',here/'minimum_area_blocks.py',here/'compile_b1_tournament.py',
        here/'repair_b1_region_partition.py',here/'fit_b1_sixteen_regions.py',here/'train_b1_binary_regions.py'],
        evaluation=parent.parent.evaluate,preview=parent.preview)
