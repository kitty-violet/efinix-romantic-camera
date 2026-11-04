"""Map source images to target-domain reconstruction codes with a fixed decoder.

The frozen teacher receives training labels only. Deployment is one ordinary
B1, with no teacher, material classifier, quantizer or external postfilter.
"""
from pathlib import Path
import copy
import torch
from torch.nn import functional as F
from . import train_b1_late_region_supervision as driver
from . import train_b1_minimum_area as m


class TargetLatentTraining(m.AreaTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.teacher=m.t.r.initialize()
        teacher_path=getattr(args,'teacher_checkpoint','') or args.warm
        state=torch.load(teacher_path,map_location='cpu',weights_only=True)
        self.teacher.load_state_dict(state['model']);self.teacher.quantized=args.qat
        if 'residual_encoding' in state:
            self.teacher.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
        teacher_is_quantized=bool(state.get('qat',False) or state.get('args',{}).get('qat',False))
        if args.qat and not teacher_is_quantized:self.teacher.calibrate()
        self.teacher.eval().requires_grad_(False)
        reached=False;self.fixed={}
        for key in self.net.weight:
            if key=='up1_low_dw':reached=True
            if reached:
                self.net.weight[key].requires_grad_(False);self.net.bias[key].requires_grad_(False)
                self.fixed[key]=(self.net.weight[key].detach().clone(),self.net.bias[key].detach().clone())
        for key,(w,b) in self.fixed.items():
            assert torch.equal(self.teacher.weight[key],w) and torch.equal(self.teacher.bias[key],b),('Decoder/teacher mismatch',key)
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def batch_style(self,native):
        result=super().batch_style(native);self.full_target=result[1];return result

    def additional_loss(self,p,target,native):
        area,values=super().additional_loss(p,target,native)
        with torch.no_grad():wanted=self.teacher(self.full_target,all_nodes=True)['res3_add']
        actual=self.nodes['res3_add']
        if not native:actual=actual[...,10:-10,10:-10];wanted=wanted[...,10:-10,10:-10]
        scale=wanted.std((0,2,3),keepdim=True).clamp_min(2).detach()
        pp,tt=actual/scale,wanted/scale
        code=F.smooth_l1_loss(pp,tt)
        edges=sum(F.l1_loss(torch.diff(pp,dim=axis),torch.diff(tt,dim=axis)) for axis in (-1,-2))/2
        values.update(target_latent=code,target_latent_edges=edges)
        return area+self.args.classification*(code+.25*edges),values

    def update(self,native=None):
        result=super().update(native)
        if self.step%100==0:
            for key,(w,b) in self.fixed.items():
                assert torch.equal(self.net.weight[key],w) and torch.equal(self.net.bias[key],b),key
        return result


if __name__=='__main__':
    driver.main(training_factory=TargetLatentTraining,extra_sources=[Path(__file__)])
