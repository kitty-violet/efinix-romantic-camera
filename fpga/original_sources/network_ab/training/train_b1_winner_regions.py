"""Learn region choice/brightness with an exact fixed8-way B1 winner.

The max tree, score bounds, boundary geometry and colour synthesis are all
ordinary exported B1 coefficients. Softmax/nearest palette are training-only.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_clean_regions import CleanRegionTraining
from .probe_b1_categorical_decoder import native_categorical
from .audit_b1_winner_palette import palette as actual_palette


class WinnerTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm['coverage_layout']=='8exclusive_winner_rays'
        self.net.package['residual_encoding']=warm['residual_encoding']
        assert warm.get('soft_gate')
        self.compact=warm.get('compact_winner',False)
        with torch.no_grad():
            if not warm.get('bounded_scores'):
                self.net.weight['down1__pw'][:8]*=8.5;self.net.bias['down1__pw'][:8]*=8.5
                self.net.weight['down2__dw'][:8]/=8.5
                for key in ('down1__pw','down2__dw'):
                    w=self.net.weight[key]
                    scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
                    getattr(self.net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
            # Previously unused input features become live without changing
            # any RGB output, because their outgoing coefficients start at0.
            if not self.net.weight['down1__pw'][:,3:].count_nonzero():
                for j in range(3,8):
                    self.net.weight['stem'][j].zero_();self.net.weight['stem'][j,(j-3)%3,1,1]=.25
                    self.net.bias['stem'][j]=32;self.net.stem_m[:,j]=512
                    self.net.weight['down1__dw'][j]=self.net.weight['down1__dw'][0]
                    self.net.bias['down1__dw'][j]=0;self.net.down1__dw_m[:,j]=self.net.down1__dw_m[:,0]
            self.palette,self.palette_classes,self.palette_y,self.palette_codes=actual_palette(self.net)
        self.fixed={};groups=[]
        for key,p in self.net.weight.items():
            train=key in ('stem','down1__pw') or (self.compact and (key=='down2__pw' or key.startswith('res1_')))
            p.requires_grad_(train);self.net.bias[key].requires_grad_(train)
            if train:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.append(dict(params=[p],lr=args.lr,row_gain=(s/(1/256))[:,None,None,None]))
                groups.append(dict(params=[self.net.bias[key]],lr=.015))
            else:self.fixed[key]=(p.detach().clone(),self.net.bias[key].detach().clone())
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)
        with torch.no_grad():
            k='res2_conv1__dw' if self.compact else 'down2__dw';s=(getattr(self.net,k+'_m')/131072).float().flatten()[:8]
            wi=self.net.round(self.net.weight[k][:8]/s[:,None,None,None]).clamp(-127,127)
            assert wi.min()>=0
            maximum=self.net.round(wi.flatten(1).sum(1)*127*s)
            assert maximum.max()<=15,maximum
        print('WINNER_SCORE_RANGE_ALL_INPUTS',maximum.tolist(),flush=True)

    @torch.no_grad()
    def targets(self,target):
        n,_,h,w=target.shape;flat=target.permute(0,2,3,1).reshape(-1,3);indices=[]
        for start in range(0,len(flat),8192):
            x=flat[start:start+8192]
            indices.append((x.square().sum(1,keepdim=True)+self.palette.square().sum(1)[None]-2*x@self.palette.T).argmin(1))
        ids=torch.cat(indices).reshape(n,h,w);classes=self.palette_classes[ids]
        valid=torch.ones(n,1,h,w,device=target.device)
        hot=F.one_hot(classes,8).permute(0,3,1,2).float()
        coverage=F.avg_pool2d(hot,4);valid=F.avg_pool2d(valid,4)
        # Exact constant-region brightness input for each reachable colour;
        # labels cannot demand unsupported nominal levels0,1,15 or16.
        luminance=self.palette_y[ids][:,None]
        return coverage,valid,F.avg_pool2d(luminance,4)

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        shape,values=formation.formation_loss(p,target,labels)
        scores=self.nodes['res2_conv1.dw' if self.compact else 'res1_conv1.dw'][:,:8]
        luminance=self.nodes['res1_add' if self.compact else 'down2.pw'][:,16:17]
        if not native:scores=scores[...,10:-10,10:-10];luminance=luminance[...,10:-10,10:-10]
        truth,valid,wanted_y=self.targets(target)
        logits=(scores+torch.arange(8,device='cuda')[None,:,None,None]/8)*2
        class_map=-(truth*F.log_softmax(logits,1)).sum(1,keepdim=True)
        class_weights=self.classification_weights(labels,logits.shape[-2:]) if hasattr(self,'classification_weights') else torch.ones_like(valid)
        classification=(class_map*class_weights).sum()/(valid*class_weights).sum().clamp_min(1)
        confidence=truth.max(1,keepdim=True).values
        chosen=truth.argmax(1,keepdim=True)
        correct=logits.gather(1,chosen);others=logits.scatter(1,chosen,-1000).max(1,keepdim=True).values
        pure=(confidence>.99).float()*valid
        margin=(F.relu(2-correct+others)*pure).sum()/pure.sum().clamp_min(1)
        yloss=F.smooth_l1_loss(luminance/64,wanted_y/64)
        values.update(winner_classification=classification,winner_margin=margin,winner_luminance=yloss)
        return self.args.area_weight*shape+self.args.classification*(classification+.2*margin+yloss),values

    input_stability_loss=CleanRegionTraining.input_stability_loss

    def update(self,native=None):
        row=super().update(native)
        if self.step%100==0:
            with torch.no_grad():
                for key,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[key]) and torch.equal(b,self.net.bias[key]),key
        return row

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(residual_encoding=self.net.package['residual_encoding'],coverage_layout='8exclusive_winner_rays',
            soft_gate=True,winner_shape=True,bounded_scores=True,winner_palette=self.palette.cpu().tolist(),
            palette_classes=self.palette_classes.cpu().tolist(),palette_y=self.palette_y.cpu().tolist(),
            actual_palette_supervision=True,compact_winner=self.compact)
        torch.save(state,path)


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=WinnerTraining,extra_sources=[Path(__file__),here/'probe_b1_exclusive_winner.py',
        here/'train_b1_quantum_regions.py',here/'train_b1_clean_regions.py',here/'audit_b1_winner_palette.py',
        here/'probe_b1_decoder_coverage.py',Path(formation.__file__)],evaluation=formation.evaluate)
