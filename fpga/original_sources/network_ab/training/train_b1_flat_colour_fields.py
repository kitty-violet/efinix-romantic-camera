"""Supervise three region coordinates before ordinal colour decisions.

The trained network predicts flat fields; the fixed, existing B1 decoder makes
colour steps. The field loss never rewards a photographic texture inside a
teacher region. Region labels/means are training-only, not inference filters.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_ordered_regions as ordered
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .build_b1_ordered_palette import GROUPS,offsets,levels


def field_terms(field,truth,labels):
    labels=F.interpolate(labels[:,None].float(),size=field.shape[-2:],mode='nearest')[:,0].long()
    hi=F.max_pool2d(labels[:,None].float(),3,1,1)
    lo=-F.max_pool2d(-labels[:,None].float(),3,1,1)
    valid=(hi==lo).float();flat=[];colour=[]
    for p,t,lab,mask in zip(field,truth,labels,valid):
        ids=lab.flatten();n=int(ids.max())+1;v=mask.flatten();ix=ids[None].expand(3,-1)
        count=v.new_zeros(n).scatter_add(0,ids,v).clamp_min(1)
        mean=p.new_zeros(3,n).scatter_add(1,ix,p.flatten(1)*v)/count[None]
        target=t.new_zeros(3,n).scatter_add(1,ix,t.flatten(1)*v)/count[None]
        norm=(3*v.sum()).clamp_min(1)
        flat.append(((p.flatten(1)-mean[:,ids]).abs()*v).sum()/norm)
        colour.append(((mean[:,ids]-target[:,ids]).abs()*v).sum()/norm)
    # Pooled colour retains narrow objects that have no eroded interior.
    coarse=F.smooth_l1_loss(F.avg_pool2d(field,4),F.avg_pool2d(truth,4),beta=.25)
    return dict(field_flat=torch.stack(flat).mean(),field_region_colour=torch.stack(colour).mean(),field_coarse_colour=coarse)


def appearance(p,t,mode):
    loss=F.l1_loss(F.avg_pool2d(p,16),F.avg_pool2d(t,16))
    return .15*loss,dict(coarse_rgb=loss)


class FlatColourTraining(ordered.OrderedTraining):
    appearance_loss=staticmethod(appearance)

    def __init__(self,args,data):
        super().__init__(args,data)
        self.coordinates=levels('cuda').float()*torch.tensor([127/7,127/4,127/4],device='cuda')

    def additional_loss(self,p,target,native):
        feature=self.nodes['up1_low_dw']
        w,b=self.net.weight['up1_low_pw'],self.net.bias['up1_low_pw'];off=offsets(w.device)
        # These three affine fields exist before the15 fixed threshold offsets.
        # This is auxiliary supervision only. Forward tensors remain unchanged.
        cw=torch.stack([w[a:z].mean(0) for a,z in GROUPS])
        cb=torch.stack([(b[a:z]-off[a:z]).mean() for a,z in GROUPS])
        coord=F.conv2d(feature,cw,cb)
        truth=F.avg_pool2d(self.coordinates[self.class_labels].permute(0,3,1,2),4)
        labels=self.labels
        if not native:
            coord=coord[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10];labels=labels[...,40:-40,40:-40]
        values=field_terms(coord/32,truth/32,labels)
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        codes=self.class_labels if native else self.class_labels[...,40:-40,40:-40]
        if not native:logits=logits[...,20:-20,20:-20]
        wanted=F.avg_pool2d(self.bit_table[codes].permute(0,3,1,2),2)
        cls=F.binary_cross_entropy_with_logits(logits,wanted)
        shape,shape_values=formation.formation_loss(p,target,labels)
        values.update(field_classification=cls,**shape_values)
        loss=4*values['field_flat']+2*values['field_region_colour']+values['field_coarse_colour']+.3*cls+.15*shape
        return loss,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(flat_colour_coordinate_supervision=True)
        torch.save(state,path)


def regression():
    lab=torch.zeros(1,64,64,dtype=torch.long,device='cuda');lab[:,:,32:]=1
    target=torch.where(lab[:,None]==0,.3,2.5).expand(-1,3,-1,-1).contiguous()
    scores={}
    for name,p in [('exact',target),('noise',target+.3*torch.randn_like(target)),('constant',target*0+1.4)]:
        v=field_terms(p,target,lab);scores[name]={k:float(z) for k,z in v.items()}
    assert scores['exact']['field_flat']<1e-4 and scores['noise']['field_flat']>.1
    assert scores['constant']['field_region_colour']>.8
    print('FLAT_FIELD_REGRESSION',scores,flush=True)


if __name__=='__main__':
    regression();here=Path(__file__).parent
    driver.main(training_factory=FlatColourTraining,extra_sources=[Path(__file__),Path(ordered.__file__),
        here/'train_b1_binary_regions.py',here/'build_b1_ordered_palette.py',here/'train_b1_winner_surrogate.py',
        here/'train_b1_quantum_regions.py',here/'train_b1_clean_regions.py',here/'fit_b1_sixteen_regions.py',
        here/'train_b1_sixteen_regions.py',Path(formation.__file__)],evaluation=ordered.evaluate,preview=ordered.preview)
