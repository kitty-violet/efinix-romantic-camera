"""S4R1 model: 11 convolutions, one residual, stride4 / nearest4.

Forward/quantization methods are copied from the original training sources.
Input is NCHW float (RGB uint8 / 128 - 1); output uses the same scale.
"""
from pathlib import Path
import copy,json
import torch
from torch import nn
from torch.nn import functional as F

class HalfAway(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.sign()*(x.abs()+.5).floor()
    @staticmethod
    def backward(ctx,g):return g

class TrainingLeakyClamp(torch.autograd.Function):
    """Ordinary exact clamp forward; explicit training-only recovery derivative."""
    @staticmethod
    def forward(ctx,x,lo,hi,leak):
        ctx.save_for_backward((x>=lo)&(x<=hi));ctx.leak=leak
        return x.clamp(lo,hi)
    @staticmethod
    def backward(ctx,g):
        (inside,)=ctx.saved_tensors
        return g*torch.where(inside,1.,ctx.leak),None,None,None

class S4R1(nn.Module):
    def __init__(self, graph_path=None):
        super().__init__()
        graph_path=Path(graph_path) if graph_path else Path(__file__).with_name('graph.json')
        self.description=json.loads(graph_path.read_text('utf-8'))
        self.rows=copy.deepcopy(self.description['nodes'])
        self.residual=copy.deepcopy(self.description['residual']);self.qat=True;self.leak=0.
        self.w=nn.ParameterDict();self.b=nn.ParameterDict()
        for r in self.rows:
            if 'kernel' not in r:continue
            n=r['id'];shape=(r['channels'],r['cin']//r['groups'],r['kernel'],r['kernel'])
            self.w[n]=nn.Parameter(torch.zeros(shape));self.b[n]=nn.Parameter(torch.zeros(r['channels']))
            self.register_buffer(n+'_m',torch.ones(r['channels'],dtype=torch.float64)*1024)
    @torch.no_grad()
    def calibrate(self):
        for name,w in self.w.items():
            m=(w.abs().flatten(1).amax(1)*131072/120).ceil().clamp(64,131071)
            getattr(self,name+'_m').copy_(m.double())
    def forward(self,x,all_nodes=False):
        clip=lambda z,lo,hi:TrainingLeakyClamp.apply(z,lo,hi,float(getattr(self,'leak',0.)))
        values={'input':x*128}
        if self.qat:values['input']=HalfAway.apply(values['input']).clamp(-128,127)
        for r in self.rows:
            name=r['id'];a=values[r['inputs'][0]]
            if name in self.w:
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                if self.qat:
                    m=getattr(self,name+'_m');s=(m/131072).float()
                    wi=HalfAway.apply(self.w[name]/s[:,None,None,None]).clamp(-127,127)
                    bi=HalfAway.apply(self.b[name]/s).double()[None,:,None,None]
                    dot=F.conv2d(a,wi,stride=r['stride'],groups=r['groups'])
                    y=clip(HalfAway.apply((dot.double()+bi).clamp(-(2**31),2**31-1)*m[None,:,None,None]/131072).float(),-128,127)
                else:y=clip(F.conv2d(a,self.w[name],self.b[name],stride=r['stride'],groups=r['groups']),-128,127)
                if r['activation']=='relu':y=clip(y,0,float('inf'))
            elif r['kind']=='nearest':y=a.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:
                z=values[r['inputs'][1]];p=self.residual[name];small,other=(a,z) if p['main_scaled'] else (z,a)
                if self.qat:
                    aligned=HalfAway.apply(small.double()*p['multiplier']/131072).clamp(-128,127)
                    y=HalfAway.apply((aligned+other.double())*p['post_multiplier']/131072).clamp(-128,127).float()
                else:y=((small*p['multiplier']/131072).clamp(-128,127)+other).mul(p['post_multiplier']/131072).clamp(-128,127)
            values[name]=y
        return values if all_nodes else values['rgb']/128

def load_checkpoint(style_folder,device='cpu'):
    folder=Path(style_folder)
    model=S4R1(folder/'graph.json')
    checkpoint=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.residual=copy.deepcopy(checkpoint['residual']);model.qat=True;model.leak=0.
    assert model.residual==model.description['residual']
    return model.to(device).eval()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=range(5),default=0);a=p.parse_args()
    m=load_checkpoint(Path(__file__).parent/'models'/f'slot{a.slot}')
    print(m)
    for n in m.rows:print(n)
    print('Parameters:',sum(p.numel() for p in m.parameters()))
