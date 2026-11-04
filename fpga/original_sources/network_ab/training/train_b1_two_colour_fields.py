"""Learn a contour and its two region colours, with an integer B1 renderer."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .probe_b1_two_colour_geometry import latent_full
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_clean_regions import CleanRegionTraining


class TwoColourB1(m.t.r.c.ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up2_low_pw':return value
        f=nodes['up1_high_dw'];prob=((f[:,6:7]-63.5)/2).sigmoid()
        proxy=F.pad(torch.cat((f[:,:3]*(1-prob),f[:,3:6]*prob),1),(0,0,0,0,0,2))
        return ExactValueProxyGradient.apply(value,proxy)


class TwoColourTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data);warm=torch.load(args.warm,map_location='cpu',weights_only=True);assert warm['two_colour_fields']
        self.net.package['residual_encoding']=warm['residual_encoding'];self.net.__class__=TwoColourB1;groups=[];self.fixed={};fixed=False
        for key,w in self.net.weight.items():
            if key=='up1_high_dw':fixed=True
            w.requires_grad_(not fixed);self.net.bias[key].requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),self.net.bias[key].detach().clone())
            else:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.append(dict(params=[w],lr=args.lr,row_gain=(s/(1/256))[:,None,None,None]))
                groups.append(dict(params=[self.net.bias[key]],lr=.004))
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def batch_style(self,native):
        x,target,specs,flow,valid=super().batch_style(native);fields=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
            value=latent_full(row['target'],row['region']);xx,yy=spec['xy'];w,h=spec['wh'];value=value[yy:yy+h,xx:xx+w]
            if spec['flip']:value=value[:,::-1]
            fields.append(torch.from_numpy(np.ascontiguousarray(value)).permute(2,0,1).cuda())
        self.field_target=F.avg_pool2d(torch.stack(fields),4)
        return x,target,specs,flow,valid

    input_stability_loss=CleanRegionTraining.input_stability_loss

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        shape,values=formation.formation_loss(p,target,labels)
        field=self.nodes['up1_low_pw'][:,:7];truth=self.field_target
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
        colour=F.smooth_l1_loss(field[:,:6]/32,truth[:,:6]/32)
        contour=F.smooth_l1_loss((field[:,6:7]-64)/16,(truth[:,6:7]-64)/16)
        near=((truth[:,:3]-truth[:,3:6]).abs().mean(1,keepdim=True)>2).float()
        signed=F.binary_cross_entropy_with_logits((field[:,6:7]-63.5)/4,(truth[:,6:7]>=64).float(),reduction='none')
        side=(signed*near).sum()/near.sum().clamp_min(1)
        loss=self.args.area_weight*shape+self.args.classification*(colour+.25*contour+.2*side)
        values.update(two_side_colours=colour,signed_contour_distance=contour,contour_side=side)
        return loss,values

    def update(self,native=None):
        row=super().update(native)
        if self.step%100==0:
            for k,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[k]) and torch.equal(b,self.net.bias[k])
        return row

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(residual_encoding=self.net.package['residual_encoding'],two_colour_fields=True,
            coverage_layout='two_rgb_sides_and_signed_contour',gradient_proxy_only=True)
        torch.save(s,path)


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=TwoColourTraining,extra_sources=[Path(__file__),here/'fit_b1_two_colour_fields.py',
        here/'probe_b1_two_colour_geometry.py',here/'train_b1_quantum_regions.py',here/'train_b1_winner_surrogate.py',
        here/'train_b1_clean_regions.py',Path(formation.__file__)],evaluation=formation.evaluate)
