"""Learn region decisions while retaining explicit integer region formation.

The diffusion and monotone colour-region renderer are fixed B1 coefficients.
Only the existing input encoder learns; it cannot turn the colour-region head
back into a continuous photographic decoder. Palette search is supervision only.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_region_formation as formation
from . import train_b1_late_region_supervision as driver


class ExplicitRegionTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        assert args.qat, 'The fixed region decoder uses native INT8 coverage rounding'
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('coverage_layout','4Y_2Co_2Cg')=='4Y_2Co_2Cg', 'This trainer needs8 colour features; the7+outline head has a different palette encoding'
        if 'residual_encoding' in warm:self.net.package['residual_encoding']=warm['residual_encoding']
        self.field_node=warm.get('field_node',warm.get('args',{}).get('field_node','res3_add'))
        args.field_node=self.field_node
        self.fixed={}
        for key in self.net.weight:
            if not (key=='stem' or key.startswith('down')):
                self.net.weight[key].requires_grad_(False);self.net.bias[key].requires_grad_(False)
                self.fixed[key]=(self.net.weight[key].detach().clone(),self.net.bias[key].detach().clone())
        self.fixed_sampling=warm.get('fixed_sampling',False)
        if self.fixed_sampling:
            for key in ('down1__dw','down2__dw'):
                self.net.weight[key].requires_grad_(False);self.net.bias[key].requires_grad_(False)
                self.fixed[key]=(self.net.weight[key].detach().clone(),self.net.bias[key].detach().clone())
        # Unused stem channels gain live signals. The following zero weights
        # initially discard them, so this does not change the initial output.
        with torch.no_grad():
            if not self.net.weight['down1__pw'][:,3:].count_nonzero():
                for j in range(3,8):
                    self.net.weight['stem'][j].zero_()
                    self.net.weight['stem'][j,(j-3)%3,1,1]=.25
                    self.net.bias['stem'][j]=40
                    if not self.fixed_sampling:
                        self.net.weight['down1__dw'][j].zero_();self.net.weight['down1__dw'][j,0,1,1]=1
                # Continuing a trained encoder preserves its existing scales.
                for key in (('stem',) if self.fixed_sampling else ('stem','down1__dw')):
                    w=self.net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
                    getattr(self.net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
        self.project()
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)
        with torch.no_grad():
            self.coverage_states=round(float(self.net.weight['up2_high_dw'][0,0].sum()*127))
            assert self.coverage_states in (4,8,16)
            indices=torch.cartesian_prod(torch.arange(9,device='cuda'),torch.arange(5,device='cuda'),torch.arange(5,device='cuda'))
            features=[]
            for c,groups in enumerate((4,2,2)):
                for g in range(groups):features.append((indices[:,c]-2*g).clamp(0,2)*(self.coverage_states/2))
            features=torch.stack(features,1).float()
            scale=(self.net.rgb_head_m/131072).float().flatten()
            wi=self.net.round(self.net.weight['rgb_head'][:,:,0,0]/scale[:,None]).clamp(-127,127)
            bi=self.net.round(self.net.bias['rgb_head']/scale)
            self.palette=self.net.round((features@wi.T+bi)*scale).clamp(-128,127)/128
            self.palette_indices=indices

    @torch.no_grad()
    def project(self):
        w,b=self.net.weight['down2__pw'],self.net.bias['down2__pw']
        for c in range(3):
            rows=[axis*12+c*4+r for axis in range(2) for r in range(4)]
            cw=w[rows].mean(0);cb=b[rows].mean();w[rows]=cw;b[rows]=cb
        w[24:]=0;b[24:]=0

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        shape,values=formation.formation_loss(p,target,labels)
        coords=self.nodes[self.field_node][:,[0,4,8]]
        if not native:coords=coords[...,10:-10,10:-10]
        colour=F.avg_pool2d(target,4)
        with torch.no_grad():
            flat=colour.permute(0,2,3,1).reshape(-1,3)
            distance=flat.square().sum(1,keepdim=True)+self.palette.square().sum(1)[None]-2*flat@self.palette.T
            ids=distance.argmin(1)
            index=self.palette_indices[ids].reshape(colour.shape[0],*colour.shape[2:],3).permute(0,3,1,2)
        logits=[];truth=[]
        for c,thresholds in enumerate(((14,28,42,56,70,84,98,112),(49,59,69,79),(49,59,69,79))):
            q=coords.new_tensor(thresholds)[None,:,None,None]
            logits.append((coords[:,c:c+1]-q)/3)
            levels=torch.arange(1,len(thresholds)+1,device=coords.device)[None,:,None,None]
            truth.append((index[:,c:c+1]>=levels).float())
        ordinal=F.binary_cross_entropy_with_logits(torch.cat(logits,1),torch.cat(truth,1))
        centers=torch.stack((7+14*index[:,0],44+10*index[:,1],44+10*index[:,2]),1).float()
        field=F.smooth_l1_loss(coords/32,centers/32)
        values.update(region_ordinal=ordinal,region_centres=field)
        return self.args.area_weight*shape+self.args.classification*(ordinal+field),values

    def update(self,native=None):
        result=super().update(native);self.project()
        if self.step%100==0:
            for key,(w,b) in self.fixed.items():
                assert torch.equal(w,self.net.weight[key]) and torch.equal(b,self.net.bias[key]),key
        return result

    def save(self,path):
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(residual_encoding=self.net.package['residual_encoding'],field_node=self.field_node,fixed_sampling=self.fixed_sampling)
        torch.save(state,path)


if __name__=='__main__':
    driver.main(training_factory=ExplicitRegionTraining,extra_sources=[Path(__file__),Path(formation.__file__)],evaluation=formation.evaluate)
