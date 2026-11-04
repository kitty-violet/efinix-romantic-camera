"""Fixed B1 only. Training losses and optical flow are never inference operators."""
from pathlib import Path
import sys, os, json, hashlib, copy, shutil, time
sys.dont_write_bytecode = True

ROOT = Path(r'E:\fpga')
OUT = Path(__file__).resolve().parents[1]
WORK = ROOT/'model-training-workspace'
OLD = ROOT/'tmp/style-review/v14-starry-strokes-20261020/review-files/v14_starry_strokes'
DEPLOY = ROOT/'output/deployment/board-full-150-video-standalone222-20261024/styles/V14/coefficients-padded.bin'
sys.path.insert(0, str(ROOT))
os.environ['TORCH_HOME'] = str(WORK/'cache/torch')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from network_ab.model_contract import compile_contract, load_contract
from network_ab.training.b1_trained_image import Parameters, encode, decode, runtime_from_image


def setup(seed=241014):
    torch.set_num_threads(4); cv2.setNumThreads(2)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Windows readers/AV can briefly deny replacement of an open report.
    # Preserve atomic replacement; bound the retry and surface real failures.
    for attempt in range(7):
        try:
            os.replace(tmp,path)
            break
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==6:
                raise
            time.sleep(.03*(2**attempt))


def disk_guard():
    free=shutil.disk_usage(ROOT).free
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if free < 20*1024**3 or size > 8*1024**3:
        raise RuntimeError(f'Disk budget stop: free={free}, own_bytes={size}')
    return dict(free_E_GiB=round(free/1024**3,3),own_GiB=round(size/1024**3,3))


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


class B1(nn.Module):
    def __init__(self, source=DEPLOY):
        super().__init__()
        self.plan=compile_contract(load_contract())
        self.package=runtime_from_image(Path(source).read_bytes())
        self.weight=nn.ParameterDict();self.bias=nn.ParameterDict();self.quantized=True
        for r in self.plan['nodes']:
            name=r['id'];key=name.replace('.','__')
            if name not in self.package['parameters']:continue
            p=self.package['parameters'][name]
            w=torch.tensor(p['weights'],dtype=torch.float32)
            if r['kind']=='dw3x3':w=w.reshape(r['channels'],1,3,3)
            else:w=w.reshape(r['channels'],r['kernel'],r['kernel'],r['input_channels']).permute(0,3,1,2).contiguous()
            m=torch.tensor(p['multipliers'],dtype=torch.float64)[None,:,None,None]
            s=(m.flatten()/131072).float()
            self.weight[key]=nn.Parameter(w*s[:,None,None,None])
            self.bias[key]=nn.Parameter(torch.tensor(p['bias'],dtype=torch.float32)*s)
            self.register_buffer(key+'_m',m)

    def forward(self,x,all_nodes=False):
        rnd=HalfAway.apply if self.quantized else lambda a:a
        leak=float(getattr(self,'training_saturation_leak',0.))
        assert 0<=leak<=1
        clip=(lambda a,lo,hi:TrainingLeakyClamp.apply(a,lo,hi,leak)) if leak else (lambda a,lo,hi:a.clamp(lo,hi))
        nodes={'input':rnd(x*128).clamp(-128,127)}
        for r in self.plan['nodes']:
            name=r['id'];key=name.replace('.','__');a=nodes[r['inputs'][0]]
            if key in self.weight:
                w,b=self.weight[key],self.bias[key]
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                if self.quantized:
                    m=getattr(self,key+'_m');s=(m/131072).float().flatten()
                    wi=rnd(w/s[:,None,None,None]).clamp(-127,127)
                    bi=rnd(b/s).double()[None,:,None,None]
                    dot=F.conv2d(a,wi,stride=r['stride'],groups=r['groups'])
                    y=clip(rnd((dot.double()+bi).clamp(-(2**31),2**31-1)*m/131072).float(),-128,127)
                else:y=clip(F.conv2d(a,w,b,stride=r['stride'],groups=r['groups']),-128,127)
                if r['activation']=='relu':y=clip(y,0,float('inf'))
                if self.quantized and name=='up2_low_pw' and getattr(self,'binary_pigment_surrogate',False):
                    # Training derivative only. The forward value remains
                    # exactly the ordinary exported INT8 result, including
                    # its existing ReLU and saturation. No soft gate exports.
                    soft=(127*torch.sigmoid((dot[:,6:8].double()+bi[:,6:8])/(127*8))).float()
                    gate=y[:,6:8].detach()+(soft-soft.detach())
                    y=torch.cat((y[:,:6],gate,y[:,8:]),1)
            elif r['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                z=nodes[r['inputs'][1]];p=self.package['residual_encoding'][name]
                small,other=(a,z) if p['main_scaled'] else (z,a)
                if self.quantized:
                    v=clip(rnd(small.double()*p['multiplier']/131072),-128,127)
                    y=clip(rnd((v+other.double())*p['post_multiplier']/131072),-128,127).float()
                else:y=((small*p['multiplier']/131072).clamp(-128,127)+other).mul(p['post_multiplier']/131072).clamp(-128,127)
            nodes[name]=y
        return nodes if all_nodes else nodes['rgb_head']/128

    @torch.no_grad()
    def calibrate(self):
        for k,w in self.weight.items():
            # Frozen deployed fast-RQ numerical policy excludes MAC m=1..63.
            # This bound applies to convolution multipliers, not residual alignment.
            m=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/120).ceil().clamp(64,131071)
            getattr(self,k+'_m').copy_(m.double()[None,:,None,None])

    @torch.no_grad()
    def export(self):
        parameters={}
        for r in self.plan['nodes']:
            k=r['id'].replace('.','__')
            if k not in self.weight:continue
            m=getattr(self,k+'_m').flatten();s=(m/131072).float()
            w=HalfAway.apply(self.weight[k]/s[:,None,None,None]).clamp(-127,127).long()
            b=HalfAway.apply(self.bias[k]/s).long();assert b.abs().max()<2**31
            parameters[r['id']]=Parameters(tuple(map(tuple,w.permute(0,2,3,1).reshape(len(w),-1).cpu().tolist())),tuple(b.cpu().tolist()),tuple(m.long().cpu().tolist()),r['activation']=='relu')
        raw=encode(parameters,self.package['residual_encoding'])
        assert len(raw)==17792
        return raw+bytes(2688)


def load_model(path=None,device='cuda'):
    model=B1()
    if path:
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True)
        model.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
        model.quantized=state.get('qat',True)
    return model.to(device).eval()


class IntegerEngine:
    """Decode the delivered bytes; no floating learned coefficients are used."""
    def __init__(self,raw,device='cuda'):
        self.plan=compile_contract(load_contract());self.device=device
        self.package=runtime_from_image(raw);self.ops={}
        for r in self.plan['nodes']:
            if r['id'] not in self.package['parameters']:continue
            p=self.package['parameters'][r['id']];w=np.array(p['weights'],np.float32)
            if r['kind']=='dw3x3':w=w.reshape(r['channels'],1,3,3)
            else:w=w.reshape(r['channels'],r['kernel'],r['kernel'],r['input_channels']).transpose(0,3,1,2)
            self.ops[r['id']]=(torch.tensor(w,device=device),torch.tensor(p['bias'],device=device,dtype=torch.float64)[None,:,None,None],torch.tensor(p['multipliers'],device=device,dtype=torch.float64)[None,:,None,None])

    @torch.inference_mode()
    def __call__(self,x,all_nodes=False):
        assert x.dtype==torch.float32 and torch.equal(x,x.round()) and x.min()>=-128 and x.max()<=127
        values={'input':x};uses={};stats={}
        for r in self.plan['nodes']:
            for key in r['inputs']:uses[key]=uses.get(key,0)+1
        for r in self.plan['nodes']:
            name=r['id'];a=values[r['inputs'][0]]
            if name in self.ops:
                w,b,m=self.ops[name]
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                dot=F.conv2d(a,w,stride=r['stride'],groups=r['groups'])
                if all_nodes:assert torch.equal(dot,dot.round())
                accum=(dot.double()+b).clamp(-(2**31),2**31-1)
                y=HalfAway.apply(accum*m/131072)
                if all_nodes:stats[name]=float(((y< -128)|(y>127)).float().mean())
                y=y.clamp(-128,127).float()
                if r['activation']=='relu':y=y.clamp_min(0)
            elif r['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                p=self.package['residual_encoding'][name];z=values[r['inputs'][1]]
                small,other=(a,z) if p['main_scaled'] else (z,a)
                v=HalfAway.apply(small.double()*p['multiplier']/131072).clamp(-128,127)
                y=HalfAway.apply((v+other.double())*p['post_multiplier']/131072).clamp(-128,127).float()
            values[name]=y
            if not all_nodes:
                for key in r['inputs']:
                    uses[key]-=1
                    if uses[key]==0:del values[key]
        return (values,stats) if all_nodes else values['rgb_head']


def read(path):return np.asarray(Image.open(path).convert('RGB'))
def tensor(a,device='cuda'):return torch.from_numpy(np.array(a,copy=True)).permute(2,0,1).float().to(device)/128-1
def rgb(t):return (t.detach()[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)


def board(rows,path,labels,width=480):
    # Diagnostic native crops can be square; never squeeze them to 16:9.
    heights=[max(round(width*a.shape[0]/a.shape[1]) for a in row) for row in rows]
    margin=28
    im=Image.new('RGB',(width*len(labels),sum(h+margin for h in heights)),(20,24,32));draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    top=0
    for h,row in zip(heights,rows):
        for j,(label,a) in enumerate(zip(labels,row)):
            ah=round(width*a.shape[0]/a.shape[1])
            im.paste(Image.fromarray(a).resize((width,ah)),(j*width,top+margin+(h-ah)//2))
            draw.text((j*width+5,top+4),label,font=font,fill='white')
        top+=h+margin
    im.save(path)
