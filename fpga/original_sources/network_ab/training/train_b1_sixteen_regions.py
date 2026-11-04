"""Train16 complete colour regions, with attainable constant-colour labels.

This removes the separate continuous brightness field that could reintroduce
photographic ramps. Target labels use region IDs, not AA-pixel classification.
Only existing B1 coefficients are used for actual inference.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_winner_surrogate import ExactValueProxyGradient,WinnerSurrogateTraining
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .probe_b1_sixteen_decoder import palette_actual


def colour_target(codes,palette):
    rgb=palette[codes].permute(0,3,1,2)
    kernel=rgb.new_tensor([.190,.620,.190]);kernel=(kernel[:,None]*kernel[None,:])[None,None].expand(3,1,3,3)
    return F.conv2d(F.pad(rgb,(1,1,1,1),mode='replicate'),kernel,groups=3)


class SixteenGradientB1(m.t.r.c.ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up1_low_pw':return value
        scores=nodes['res1_conv2.dw'][:,:16]
        logits=(scores+torch.arange(16,device=scores.device)[None,:,None,None]/16)*2
        return ExactValueProxyGradient.apply(value,127*logits.softmax(1))


class SixteenTraining(m.AreaTraining):
    network_class=SixteenGradientB1
    sampler_bound=7

    def __init__(self,args,data):
        super().__init__(args,data);warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('sixteen_winner');self.net.package['residual_encoding']=warm['residual_encoding']
        self.net.__class__=self.network_class;self.palette=palette_actual(self.net);self.pk=palette_key(self.palette)
        self.fixed={};groups=[]
        for key,w in self.net.weight.items():
            train=key in ('stem','down1__pw','down2__pw','res1_conv1__dw','res1_conv1__pw')
            w.requires_grad_(train);self.net.bias[key].requires_grad_(train)
            if train:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.append(dict(params=[w],lr=args.lr,row_gain=(s/(1/256))[:,None,None,None]))
                groups.append(dict(params=[self.net.bias[key]],lr=.015))
            else:self.fixed[key]=(w.detach().clone(),self.net.bias[key].detach().clone())
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)
        with torch.no_grad():
            k='res1_conv2__dw';s=(getattr(self.net,k+'_m')/131072).float().flatten()[:16]
            wi=self.net.round(self.net.weight[k][:16]/s[:,None,None,None]).clamp(-127,127)
            assert wi.min()>=0 and self.net.round(wi.flatten(1).sum(1)*127*s).max()<=self.sampler_bound

    def batch_style(self,native):
        x,old,specs,flow,valid=super().batch_style(native);codes=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
            a=region_class_map(row['target'],row['region'],self.pk);xx,yy=spec['xy'];w,h=spec['wh'];a=a[yy:yy+h,xx:xx+w]
            if spec['flip']:a=a[:,::-1]
            codes.append(torch.from_numpy(np.ascontiguousarray(a.astype(np.int64))).cuda())
        self.class_labels=torch.stack(codes)
        return x,colour_target(self.class_labels,self.palette),specs,flow,valid

    classification_weights=WinnerSurrogateTraining.classification_weights
    input_stability_loss=CleanRegionTraining.input_stability_loss

    def score_logits(self,scores):return (scores+torch.arange(16,device=scores.device)[None,:,None,None]/16)*2

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        codes=self.class_labels if native else self.class_labels[...,40:-40,40:-40]
        shape,values=formation.formation_loss(p,target,labels)
        scores=self.nodes['res1_conv2.dw'][:,:16]
        if not native:scores=scores[...,10:-10,10:-10]
        logits=self.score_logits(scores)
        truth=F.avg_pool2d(F.one_hot(codes,16).permute(0,3,1,2).float(),4)
        weight=self.classification_weights(labels,scores.shape[-2:])
        ce=-(truth*F.log_softmax(logits,1)).sum(1,keepdim=True)
        cls=(ce*weight).sum()/weight.sum()
        pure=(truth.max(1,keepdim=True).values>.99).float();chosen=truth.argmax(1,keepdim=True)
        correct=logits.gather(1,chosen);other=logits.scatter(1,chosen,-1000).max(1,keepdim=True).values
        margin=(F.relu(1.5-correct+other)*pure*weight).sum()/(pure*weight).sum().clamp_min(1)
        loss=self.args.area_weight*shape+self.args.classification*(cls+.15*margin)
        values.update(sixteen_classification=cls,sixteen_margin=margin)
        if not native and self.step%8==0:
            from .minimum_area_blocks import components,area_merge,boundary_mode
            with torch.no_grad():
                ids=logits[0].argmax(0).cpu().numpy();pal=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                rgb=pal[ids];merged,_=area_merge(components(ids),rgb,16)
                clean,_=area_merge(components(boundary_mode(merged,1)),rgb,16)
                count=np.bincount(clean.ravel());means=np.stack([np.bincount(clean.ravel(),weights=rgb[...,c].ravel())/count for c in range(3)],1)
                colour=((means[:,None]-pal[None].astype(float))**2).sum(2).argmin(1)[clean]
                correction=torch.from_numpy(colour.astype(np.int64)).cuda()[None];changed=(correction!=torch.from_numpy(ids).cuda()[None]).float()
            cleanup=(F.cross_entropy(logits[:1],correction,reduction='none')*changed).sum()/changed.sum().clamp_min(1)
            loss+=.2*cleanup;values.update(colour_area_cleanup=cleanup,cleanup_fraction=changed.mean())
        return loss,values

    def update(self,native=None):
        row=super().update(native)
        if self.step%100==0:
            for k,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[k]) and torch.equal(b,self.net.bias[k]),k
        return row

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(residual_encoding=self.net.package['residual_encoding'],coverage_layout='16exclusive_rgb',sixteen_winner=True,
            winner_backward_proxy=True,attainable_region_colour_targets=True,palette=self.palette.cpu().tolist(),
            partition_preserving_decoder=True)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    palette=palette_actual(net);pk=palette_key(palette);records=[]
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];p=net(x)[0]
        codes=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64)).cuda()[None]
        target=colour_target(codes,palette);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64)).cuda()[None]
        shape,values=formation.formation_loss(p,target,labels);appear,_=m.t.appearance(p,target,'direct')
        metrics=m.t.r.metrics(p,target);metrics.update({k:float(v) for k,v in values.items()})
        original=m.t.r.tensor(m.t.r.read(row['target']))[None]
        metrics.update(score=float(shape+.25*appear),original_target_mae=float((p-original).abs().mean()*128))
        records.append(dict(id=row['id'],**metrics))
    return dict(records=records,mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'})


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=SixteenTraining,extra_sources=[Path(__file__),here/'fit_b1_sixteen_regions.py',here/'probe_b1_sixteen_decoder.py',
        here/'train_b1_winner_surrogate.py',here/'train_b1_winner_regions.py',here/'train_b1_quantum_regions.py',
        here/'train_b1_clean_regions.py',here/'probe_b1_decoder_coverage.py',here/'minimum_area_blocks.py',Path(formation.__file__)],evaluation=evaluate)
