"""Fixed hardware scales, exact forward Q17 arithmetic, STE training of B1."""
import copy
import gc
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .train_b1_deploy_preview import OUT, setup, Training, evaluate
from .probe_b1_cartoon_int8 import IntegerB1
from .runtime import file_hash,write_json,contract_guard
from .train_b1_distillation import read_image,as_tensor


class Round(nn.Module):
    class Op(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return torch.sign(x)*torch.floor(torch.abs(x)+.5)
        @staticmethod
        def backward(ctx,g):return g
    def forward(self,x):return self.Op.apply(x)


class QuantB1(nn.Module):
    def __init__(self,package):
        super().__init__();self.package=copy.deepcopy(package);self.plan=contract_guard()[1]
        self.weight=nn.ParameterDict();self.bias=nn.ParameterDict();self.round=Round()
        for row in self.plan['nodes']:
            name=row['id'];key=name.replace('.','__')
            if name not in package['parameters']:continue
            p=package['parameters'][name];wi=torch.tensor(p['weights'],dtype=torch.float32)
            if row['kind']=='dw3x3':wi=wi.reshape(row['channels'],1,3,3)
            else:wi=wi.reshape(row['channels'],row['kernel'],row['kernel'],row['input_channels']).permute(0,3,1,2).contiguous()
            # Train directly in integer parameter units. Separate optimizer LR
            # compensates this unit, and forward remains exact integer operands.
            self.weight[key]=nn.Parameter(wi);self.bias[key]=nn.Parameter(torch.tensor(p['bias'],dtype=torch.float64))
            self.register_buffer(key+'_m',torch.tensor(p['multipliers'],dtype=torch.float64)[None,:,None,None])

    def forward(self,x):
        values={'input':self.round(x*128).clamp(-128,127)}
        for row in self.plan['nodes']:
            name=row['id'];key=name.replace('.','__');a=values[row['inputs'][0]]
            if key in self.weight:
                w=self.round(self.weight[key]).clamp(-127,127);b=self.round(self.bias[key])[None,:,None,None]
                if row['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                dot=F.conv2d(a,w,None,stride=row['stride'],groups=row['groups'])
                accum=(dot.double()+b).clamp(-(2**31),2**31-1)
                y=self.round(accum*getattr(self,key+'_m')/131072).clamp(-128,127).float()
                if row['activation']=='relu':y=y.clamp_min(0)
            elif row['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                r=self.package['residual_encoding'][name];z=values[row['inputs'][1]]
                small,other=(a,z) if r['main_scaled'] else (z,a)
                aligned=self.round(small.double()*r['multiplier']/131072).clamp(-128,127)
                y=self.round((aligned+other.double())*r['post_multiplier']/131072).clamp(-128,127).float()
            values[name]=y
        return values['rgb_head']/128

    def integer_package(self):
        p=copy.deepcopy(self.package)
        for row in self.plan['nodes']:
            name=row['id'];key=name.replace('.','__')
            if key not in self.weight:continue
            wi=self.round(self.weight[key]).clamp(-127,127).detach().cpu().to(torch.int32)
            p['parameters'][name]['weights']=wi.permute(0,2,3,1).reshape(row['channels'],-1).tolist()
            p['parameters'][name]['bias']=self.round(self.bias[key]).detach().cpu().to(torch.int64).tolist()
        return p


def main():
    cfg,data,binding=setup();base=OUT/'package-2000/runtime_parameters.json';p=json.loads(base.read_text())
    cfg=copy.deepcopy(cfg);cfg.update(steps=600,lr=.04,lr_floor=.006)
    # Adam step now uses integer weight units. 0.04 codes/update is small.
    binding=dict(parent=binding,base_sha256=file_hash(base),code_sha256=file_hash(__file__),qat=True)
    def make():
        tr=Training(cfg,data,binding);tr.net=QuantB1(p).cuda().eval();tr.opt=torch.optim.Adam(tr.net.parameters(),lr=cfg['lr']);return tr
    out=OUT/'qat';out.mkdir(exist_ok=False);tr=make();integer=IntegerB1(p)
    with torch.no_grad():
        x=as_tensor(read_image(data['validation'][0]['source']))[None].cuda()
        difference=float((tr.net(x)*128-integer(x*128)).abs().max());assert difference==0
    pre=[tr.update(True),tr.update(False)];tr.save(out/'resume.pt');ref=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=make();tr.restore(out/'resume.pt');got=tr.update(False)
    error=max(float((v.cpu()-expected[k]).abs().max()) for k,v in tr.net.state_dict().items())
    assert error<2e-5 and ref['sample']==got['sample'] and ref['crop']==got['crop']
    write_json(out/'preflight.json',dict(status='PASS',initial_integer_max_error=difference,resume_max_error=error,tests=pre,binding=binding,config=cfg))
    del tr;gc.collect();torch.cuda.empty_cache();tr=make()
    with (out/'metrics.jsonl').open('x',buffering=1) as log:
        while tr.step<cfg['steps']:
            row=tr.update((tr.step+1)%100==0);log.write(json.dumps(row)+'\n')
            if tr.step%25==0:print('QAT_B1',tr.step,round(row['loss'],4),flush=True)
            if tr.step%200==0:
                tr.save(out/f'checkpoint_{tr.step:06d}.pt')
                payload=tr.net.integer_package();payload.update(qat_steps=tr.step,qat_binding=binding,qat_checkpoint_sha256=file_hash(out/f'checkpoint_{tr.step:06d}.pt'))
                write_json(out/f'runtime_{tr.step:06d}.json',payload)
                evaluate(tr.net,data,f'qat_{tr.step:06d}')
    print('QAT_COMPLETE',tr.step,flush=True)


if __name__=='__main__':main()
