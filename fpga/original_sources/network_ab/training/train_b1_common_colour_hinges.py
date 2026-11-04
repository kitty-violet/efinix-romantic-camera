"""Learn three shared colour coordinates, retaining the fixed B1 decoder.

Shared30 hinge rows are ordinary B1 PW coefficients; their differences are
compiled into the existing15 colour fields. Training-only coordinate and region
losses do not appear in the exported graph or weight package.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_predecision_regions import PredecisionTraining,evaluate,driver,preview
from .train_b1_quantum_regions import RowScaledAdam
from .build_b1_thirty_binary_regions import GROUPS30


@torch.no_grad()
def project_head(net):
    key='res3_conv1__pw';w,b=net.weight[key],net.bias[key]
    for a,z in GROUPS30:
        w[a:z]=w[a:z].mean(0,keepdim=True)
        offset=torch.arange(z-a,device=b.device)*127
        b[a:z]=(b[a:z]+offset).mean()-offset
        assert torch.equal(getattr(net,key+'_m')[:,a:z],getattr(net,key+'_m')[:,a:a+1].expand_as(getattr(net,key+'_m')[:,a:z]))


def tied_gradient(grad):
    grad=grad.clone()
    for a,b in GROUPS30:grad[a:b]=grad[a:b].mean(0,keepdim=True)
    return grad


class CommonHingeTraining(PredecisionTraining):
    def __init__(self,args,data):
        super().__init__(args,data);assert self.setup.get('common_coordinate_hinges')
        self.fixed={};groups=[]
        for key,w in self.net.weight.items():
            fixed=key.startswith(('res3_conv2','up1_','up2_','rgb_head')) or key in ('down1__dw','down2__dw')
            b=self.net.bias[key];w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),b.detach().clone())
            else:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(s*256)[:,None,None,None]),dict(params=[b],lr=.008)])
        project_head(self.net)
        self.net.weight['res3_conv1__pw'].register_hook(tied_gradient);self.net.bias['res3_conv1__pw'].register_hook(tied_gradient)
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        hinge=self.nodes['res3_conv1.pw'];coordinate=torch.stack([hinge[:,a:b].sum(1)/127 for a,b in GROUPS30],1)
        code=self.class_labels;target=F.avg_pool2d(torch.stack((code//81,code//9%9,code%9),1).float(),4)
        if not native:coordinate=coordinate[...,10:-10,10:-10];target=target[...,10:-10,10:-10]
        colour=F.smooth_l1_loss(coordinate,target,beta=.5);v['shared_coordinate_colour']=colour
        return loss+3*colour,v

    def update(self,native=None):
        result=super().update(native);project_head(self.net);return result

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(common_coordinate_hinges=True,tied_hinge_rows=[list(v) for v in GROUPS30],
            coordinate_target_weight=3,extra_inference_ops=0,visual_acceptance=False);torch.save(s,path)


def regression(checkpoint):
    from .probe_b1_encoder_gauge import load
    from .seed_b1_common_colour_hinges import regression as byte_regression
    state,net=load(checkpoint);project_head(net);byte_regression(net,state['binary_coverage720'])
    key='res3_conv1__pw';before=net.export();project_head(net);assert net.export()==before
    grad=tied_gradient(torch.randn_like(net.weight[key]));assert all(torch.equal(grad[a:b],grad[a:a+1].expand_as(grad[a:b])) for a,b in GROUPS30)
    x=torch.rand(1,3,64,80)*1.9-1;nodes=net(x,all_nodes=True)
    colour=sum(nodes['res3_conv1.pw'][:,a:b].sum(1).square().mean()/127**2 for a,b in GROUPS30)
    colour.backward();assert net.weight[key].grad.norm()>0 and torch.isfinite(net.weight[key].grad).all()
    print('COMMON_HINGE_TRAIN_REGRESSION_PASS shared_rows/gradient/forward/bytes',flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    if '--check' in sys.argv:regression(sys.argv[sys.argv.index('--check')+1])
    else:driver.main(training_factory=CommonHingeTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
