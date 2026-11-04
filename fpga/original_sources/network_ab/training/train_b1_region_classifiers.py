"""Learn independent region decisions, retaining B1's explicit region head."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_quantum_regions import QuantumRegionTraining
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation


class RegionClassifierTraining(QuantumRegionTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('classifier_mode')
        key='res3_conv1__pw';w,b=self.net.weight[key],self.net.bias[key]
        w.requires_grad_(True);b.requires_grad_(True);self.fixed.pop(key,None)
        scale=(getattr(self.net,key+'_m')/131072).float().flatten()
        self.opt.add_param_group(dict(params=[w],lr=args.lr,row_gain=(scale/(1/256))[:,None,None,None]))
        self.opt.add_param_group(dict(params=[b],lr=.02))
        self.base_lrs=[g['lr'] for g in self.opt.param_groups];self.start_integer[key]=self.codes(key).clone()

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        shape,values=formation.formation_loss(p,target,labels)
        feature=self.nodes['res3_conv1.dw']
        if not native:feature=feature[...,10:-10,10:-10]
        key='res3_conv1__pw';s=(self.net.res3_conv1__pw_m/131072).float().flatten()[:16]
        w=self.net.round(self.net.weight[key][:16]/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
        b=self.net.round(self.net.bias[key][:16]/s)*s
        logits=(F.conv2d(feature,w,b)-63.5)/24
        truth=self.coverage_truth(target)
        classification=F.binary_cross_entropy_with_logits(logits,truth)
        boundary=torch.zeros_like(labels,dtype=torch.bool)
        boundary[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
        boundary[:,1:]|=labels[:,1:]!=labels[:,:-1]
        interior=(F.max_pool2d(boundary[:,None].float(),9,1,4)==0).float()
        interior=(F.avg_pool2d(interior,4)==1).float()
        # A positive class withlogit1.5 produces100/127 coverage. The2x2
        # opening/closing decoder can turn this into a partial colour even
        # throughout a constant region. Margin4 clips stable interiors fully;
        # contour pixels remain governed by classification and boundary loss.
        margin=(F.relu(4.-(2*truth-1)*logits)*interior).sum()/(16*interior.sum()).clamp_min(1)
        prob=logits.sigmoid();ordered=[]
        for lo,hi in ((0,8),(8,12),(12,16)):ordered.append(F.relu(prob[:,lo+1:hi]-prob[:,lo:hi-1]).mean())
        nesting=torch.stack(ordered).mean()
        values.update(region_classifier=classification,region_margin=margin,region_order=nesting)
        return self.args.area_weight*shape+self.args.classification*(classification+.4*margin+.3*nesting),values

    @torch.no_grad()
    def coverage_truth(self,target):
        colour=F.avg_pool2d(target,4)
        flat=colour.permute(0,2,3,1).reshape(-1,3)
        distance=flat.square().sum(1,keepdim=True)+self.palette.square().sum(1)[None]-2*flat@self.palette.T
        idx=self.palette_indices[distance.argmin(1)].reshape(colour.shape[0],*colour.shape[2:],3).permute(0,3,1,2)
        return torch.cat([(idx[:,c:c+1]>=torch.arange(1,n+1,device=target.device)[None,:,None,None]).float() for c,n in enumerate((8,4,4))],1)

    def save(self,path):
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True);state['classifier_mode']=True;torch.save(state,path)


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=RegionClassifierTraining,
                extra_sources=[Path(__file__),Path(formation.__file__),here/'train_b1_quantum_regions.py',
                               here/'train_b1_region_capacity.py',here/'train_b1_explicit_regions.py'],
                evaluation=formation.evaluate)
