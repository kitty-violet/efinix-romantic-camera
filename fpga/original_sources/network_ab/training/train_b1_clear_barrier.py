"""Learn a narrow, explicit barrier instead of a diffuse distance response.

Labels rasterize the same frozen thin curves onto the existing quarter grid.
Hard negative mining includes weak positive responses actually used by B1.
There is no thresholding or cleaning of a published network prediction.
"""
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_deep_contour import DeepTraining,OUT as LABELS
from . import resume_b1_contour_precision as general
from .probe_b1_crop_quantization import calibrate_parameter_precision
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1

OUT=LABELS.parent/'b1-clear-barrier-20261005-v1'
PARENT=LABELS.parent/'b1-feature-flow-repair-20261005-v1/reconnected_001000.pt'


def rasterize(thin):
    # Cell k is centred at source 4*k. Tie at +2 goes to the next cell;
    # [-2,-1,0,1] are covered by cell0. Labels alone use this reduction.
    h,w=thin.shape[-2:]
    assert h%4==0 and w%4==0
    cells=F.max_pool2d(F.pad(thin,(2,5,2,5)),4,stride=4)
    out=cells[...,:h//4,:w//4].clone()
    # Clamp the last two source rows/columns to the final existing grid cell.
    out[...,-1,:]=torch.maximum(out[...,-1,:],cells[...,h//4,:w//4])
    out[...,:,-1]=torch.maximum(out[...,:,-1],cells[...,:h//4,w//4])
    out[...,-1,-1]=torch.maximum(out[...,-1,-1],cells[...,h//4,w//4])
    return out


def target_gate(row):
    thin=torch.tensor((np.load(row['label'])>=.999).astype(np.float32))[None,None]
    return rasterize(thin)[0,0].numpy()*24


def loss_terms(p,t):
    p=p[:,:,8:-8,8:-8];t=t[:,:,8:-8,8:-8];pos=t>0;neg=~pos
    error=F.smooth_l1_loss(p/16,(t*32-8)/16,reduction='none')
    positive=error[pos].mean() if pos.any() else p.sum()*0
    negative=error[neg]
    count=max(1,int(negative.numel()*.20))
    hard=torch.topk(negative.flatten(),count).values.mean()
    # Penalize every false positive, not only responses exceeding display4.
    ghost=F.relu(p[neg]+2)/16
    ghosts=torch.topk(ghost.flatten(),count).values.mean()
    loss=positive+.4*negative.mean()+.7*hard+.8*ghosts
    return loss,dict(positive=positive,negative=negative.mean(),hard=hard,ghost=ghosts)


class Trainer(DeepTraining):
    def __init__(self,data):
        super().__init__(data)
        state=torch.load(PARENT,map_location='cpu',weights_only=True)
        self.net.load_state_dict(state['model']);self.net.package=state['runtime'];self.net.quantized=False
        self.opt=torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],lr=.04)
        self.cache=[(r,rgb,(label>=.999).astype(np.float32)) for r,rgb,label in self.cache]

    def update(self,native=False):
        start=time.perf_counter();r,rgb,thin=self.rng.choice(self.cache);h,w=rgb.shape[:2]
        y=0 if native else self.rng.randrange(h-320+1);x=0 if native else self.rng.randrange(w-320+1)
        hh,ww=(h,w) if native else (320,320)
        source=torch.tensor(rgb[y:y+hh,x:x+ww].transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)/128-1
        truth=rasterize(torch.tensor(thin[y:y+hh,x:x+ww].copy(),device='cuda')[None,None])
        loss,parts=loss_terms(self.net(source),truth)
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for kind,key in self.net.masks:
            p=(self.net.weight if kind=='w' else self.net.bias)[key]
            if p.grad is not None:p.grad.mul_(getattr(self.net,kind+'_'+key+'_mask'))
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.net.parameters() if p.requires_grad],10,error_if_nonfinite=True)
        assert torch.isfinite(loss) and norm>0
        self.opt.step();self.net.project();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,sample=r['id'],crop=[x,y,ww,hh],loss=float(loss.detach()),
            terms={k:float(v.detach()) for k,v in parts.items()},grad=float(norm),quantized=self.net.quantized,
            seconds=time.perf_counter()-start)


@torch.no_grad()
def error_review(data,tag):
    actual=IntegerB1(runtime_from_image((OUT/tag/'coefficients-padded.bin').read_bytes()))
    images={};scores=[]
    for r in data:
        if r['split']=='train':continue
        a=np.array(Image.open(r['source']));x=torch.tensor(a.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
        nodes,_=actual(x,True);p=nodes['res3_conv1.dw'][0,24].cpu().numpy();t=target_gate(r)
        empty=t==0;overlay=np.zeros((*p.shape,3),np.uint8)
        overlay[(p>0)&~empty]=[180,180,180];overlay[(p>0)&empty]=[255,75,65]
        overlay[(p<=4)&~empty]=[0,200,255]
        images[r['id']]=dict(source=Image.fromarray(a),target=general.gray(t),actual=general.gray(p),
            errors=Image.fromarray(overlay).resize((1280,720),Image.Resampling.NEAREST))
        valid=np.zeros_like(empty);valid[8:-8,8:-8]=True;empty&=valid
        scores.append(dict(id=r['id'],false_nonzero=float(((p>0)&empty).sum()/max(empty.sum(),1)),
            miss=float(((p<=4)&(t>0)&valid).sum()/max(((t>0)&valid).sum(),1))))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,_) in enumerate(general.gate.CROPS):
        for j,(name,title) in enumerate([('source','原片'),('target','窄边界训练信号'),('actual','实际权重重载；未清理'),('errors','红：杂响应；蓝：漏边')]):
            board.paste(images[key][name].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),title,font=font,fill='white')
    board.save(OUT/tag/'native_errors.png');write_json(OUT/tag/'all_responses.json',scores)


def main():
    configure_workspace();set_reproducible(2026100512);contract_guard();OUT.mkdir(exist_ok=False);general.OUT=OUT
    data=json.loads((LABELS/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):assert file_hash(r[key])==r[key+'_sha256']
    # Geometric test: every thin pixel maps to the nearest stride4 cell.
    for y,x in [(0,0),(1,1),(2,2),(5,6),(30,30),(31,31)]:
        a=torch.zeros(1,1,32,32);a[0,0,y,x]=1;b=rasterize(a)
        assert b[0,0,min((y+2)//4,7),min((x+2)//4,7)]==1 and b.sum()==1
    t=torch.zeros(1,1,80,80);t[:,:,20:60,40]=1;exact=t*32-8
    fuzzy=exact.clone();fuzzy[:,:,10:70,10]=1
    missing=exact.clone();missing[:,:,30:40,40]=-8
    assert loss_terms(exact,t)[0] < loss_terms(fuzzy,t)[0] and loss_terms(exact,t)[0] < loss_terms(missing,t)[0]
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),
        label='Same thin curves, nearest stride4 occupancy, positive24/background-8',
        steps=1600,fp32_steps=1000,qat_steps=600,hard_negative_fraction=.2,no_prediction_threshold=True))
    for name in ('train_b1_clear_barrier.py','train_b1_structural_features.py','train_b1_deep_contour.py','resume_b1_contour_precision.py'):
        shutil.copy2(Path(__file__).with_name(name),OUT/name)
    tr=Trainer(data);native=tr.update(True);crop=tr.update();general.save(tr,OUT/'resume_probe.pt')
    ref=tr.update();expected={k:v.clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(data);general.restore(fresh,OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error==0 and got['crop']==ref['crop'];write_json(OUT/'preflight.json',dict(native=native,crop=crop,disk_resume_error=error))
    del tr,fresh;torch.cuda.empty_cache();tr=Trainer(data)
    general.review(tr,data,'before',True,target_gate);error_review(data,'before')
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<1600:
            if tr.step==1000:
                write_json(OUT/'calibration.json',calibrate_parameter_precision(tr.net));tr.net.quantized=True
                tr.opt=torch.optim.Adam([p for p in tr.net.parameters() if p.requires_grad],lr=.012)
            lr=.04-.02*(tr.step/1000) if tr.step<1000 else .002+.01*.5*(1+math.cos(math.pi*(tr.step-1000)/600))
            for group in tr.opt.param_groups:group['lr']=lr
            row=tr.update((tr.step+1)%100==0);f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('CLEAR_BARRIER',tr.step,round(row['loss'],5),flush=True)
            if tr.step in (1000,1300,1600):
                general.save(tr,OUT/f'checkpoint_{tr.step:06d}.pt');tag=f'step_{tr.step:06d}'
                general.review(tr,data,tag,tr.net.quantized,target_gate)
                if tr.net.quantized:error_review(data,tag)
    print('CLEAR_BARRIER_COMPLETE',flush=True)


if __name__=='__main__':main()
