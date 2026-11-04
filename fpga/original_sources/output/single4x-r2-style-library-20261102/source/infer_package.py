"""Standalone S4R2 BIN inference. Requires existing torch/numpy/Pillow only."""
from pathlib import Path
import argparse
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from verify_package import load

def rnd(x):return x.sign()*(x.abs()+.5).floor()

class Engine:
    def __init__(self,folder,device='cuda'):
        self.graph,params=load(Path(folder));self.device=device;self.ops={}
        for r in self.graph['nodes']:
            n=r['id']
            if n not in params:continue
            p=params[n];shape=(r['channels'],r['cin']//r['groups'],r['kernel'],r['kernel'])
            self.ops[n]=(torch.tensor(p['weight'],device=device,dtype=torch.float32).reshape(shape),torch.tensor(p['bias'],device=device,dtype=torch.float64)[None,:,None,None],torch.tensor(p['multiplier'],device=device,dtype=torch.float64)[None,:,None,None])
    @torch.inference_mode()
    def __call__(self,a):
        assert a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3 and a.shape[0]%4==a.shape[1]%4==0
        x=torch.from_numpy(np.array(a,copy=True)).to(self.device).permute(2,0,1)[None].float()-128;v={'input':x};uses={}
        for r in self.graph['nodes']:
            for k in r['inputs']:uses[k]=uses.get(k,0)+1
        for r in self.graph['nodes']:
            n=r['id'];x=v[r['inputs'][0]]
            if n in self.ops:
                w,b,m=self.ops[n]
                if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
                dot=F.conv2d(x,w,stride=r['stride'],groups=r['groups'])
                y=rnd((dot.double()+b).clamp(-(2**31),2**31-1)*m/131072).clamp(-128,127).float()
                if r['activation']=='relu':y=y.clamp_min(0)
            elif r['kind']=='nearest':y=x.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:
                p=self.graph['residual'][n];z=v[r['inputs'][1]];small,other=(x,z) if p['main_scaled'] else (z,x)
                aligned=rnd(small.double()*p['multiplier']/131072).clamp(-128,127)
                y=rnd((aligned+other.double())*p['post_multiplier']/131072).clamp(-128,127).float()
            v[n]=y
            for k in r['inputs']:
                uses[k]-=1
                if not uses[k]:del v[k]
        return (v['rgb'][0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)

def main():
    p=argparse.ArgumentParser();p.add_argument('--style-root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    img=np.asarray(Image.open(a.input).convert('RGB'));assert img.shape==(720,1280,3),'The delivered contract is native 720p RGB.'
    Image.fromarray(Engine(a.style_root,a.device)(img)).save(a.output)
    print('S4R2_INTEGER_OUTPUT_SAVED',a.output)
if __name__=='__main__':main()
