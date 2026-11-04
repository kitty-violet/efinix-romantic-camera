"""Deep region coverage learning, fixed late cleanup, full B1 byte metadata."""
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_late_region_supervision as driver
from .seed_b1_deep_regions import THRESHOLDS
from . import train_b1_minimum_area as m


class DeepB1(m.t.r.c.ColourB1):
    def forward(self,x,gate=None,mix=0.,all_nodes=False):
        nodes=super().forward(x,gate=gate,mix=mix,all_nodes=True)
        if nodes['res2_conv2.pw'].requires_grad:
            nodes['res2_conv2.pw'].register_hook(lambda g:g*.001)
        return nodes if all_nodes else (nodes['rgb_head']/128,nodes['res3_add'][:,24:25])


class DeepTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        state=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
        self.net.__class__=DeepB1
        reached=False;self.frozen={}
        for key in self.net.weight:
            if key=='res3_conv1__dw':reached=True
            if reached:
                self.net.weight[key].requires_grad_(False);self.net.bias[key].requires_grad_(False)
                self.frozen[key]=(self.net.weight[key].detach().clone(),self.net.bias[key].detach().clone())
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def additional_loss(self,p,target,native):
        area,values=super().additional_loss(p,target,native)
        feature=self.nodes['res2_conv2.dw']
        if not native:feature=feature[...,10:-10,10:-10]
        raw=F.conv2d(feature,self.net.weight['res2_conv2__pw'][:16],self.net.bias['res2_conv2__pw'][:16])
        z=(target+1)*64;r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
        colour=torch.cat(((r+2*g+b)/4,(r-b)/2+64,(-r+2*g-b)/4+64),1)
        masks=torch.cat([(colour[:,c:c+1]>threshold).float() for c,threshold in THRESHOLDS],1)
        truth=F.avg_pool2d(masks,4)
        logits=(raw-63.5)/16;prediction=torch.sigmoid(logits)
        classification=F.binary_cross_entropy_with_logits(logits,truth)
        # Separate interior confidence and fractional boundary coverage. A
        # boundary pixel can legitimately cover two neighbouring regions.
        interior=((truth==0)|(truth==1)).float()
        clean=((prediction-truth).abs()*interior).sum()/interior.sum().clamp_min(1)
        nesting=F.relu(prediction[:,1:8]-prediction[:,:7]).mean()
        loss=classification+.5*clean+.5*nesting
        values.update(deep_coverage_bce=classification,interior_region_error=clean,ordinal_nesting=nesting)
        return area+self.args.classification*loss,values

    def save(self,path):
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        state['residual_encoding']=copy.deepcopy(self.net.package['residual_encoding'])
        torch.save(state,path)

    def restore(self,path):
        super().restore(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])


if __name__=='__main__':
    driver.main(training_factory=DeepTraining,extra_sources=[Path(__file__),Path(__file__).with_name('seed_b1_deep_regions.py')])
