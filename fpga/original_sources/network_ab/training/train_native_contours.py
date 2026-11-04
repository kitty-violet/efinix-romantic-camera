"""Isolated learnability test, then source-separated contour-only training.

No teacher is evaluated during inference. Canny is used ONLY to label saved
teacher images, not to sharpen a student prediction. Metrics use raw network
probabilities with a fixed threshold, without NMS or dilation of predictions.
The optional tracing term is independently implemented from the CATS boundary
competition definition (WHUHLX/CATS/loss/tracingloss.py); not the full CATS net.
"""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from .native_contour_net import NativeContourNet
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible

ROOT = Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
DATA = ROOT/'sharp-contour-region-v2/data.json'
LABEL = dict(kind='native_RGB_Canny', low=80, high=160, aperture=3, l2=True,
             label_only=True, prediction_postprocessing=False)


def load_rgb(p): return np.array(Image.open(p).convert('RGB'))
def tensor(a): return torch.from_numpy(a.copy()).permute(2,0,1).float()/128-1
def edges(a): return cv2.Canny(a, LABEL['low'], LABEL['high'], apertureSize=3, L2gradient=True).astype(np.float32)/255


def criterion(logit, label, tracing=.35, margin=16):
    z = logit[..., margin:-margin, margin:-margin]
    t = label[..., margin:-margin, margin:-margin]
    p = z.sigmoid()
    ce = F.binary_cross_entropy_with_logits(z, t, reduction='none')
    pos = (ce*t).sum()/t.sum().clamp_min(1)
    neg = (ce*(1-t)).sum()/(1-t).sum().clamp_min(1)
    # Nearby off-boundary responses compete with correct boundary responses.
    pool = lambda x: F.avg_pool2d(x,5,1,2)*25
    nearby = (pool(t)>0).to(p.dtype)*(1-t)
    correct = pool(p*t)
    wrong = pool(p*nearby)
    trace = (-torch.log((correct/(correct+wrong+1e-6)).clamp_min(1e-6))*t).sum()/t.sum().clamp_min(1)
    far = 1-(pool(t)>0).to(p.dtype)
    extra = (p*far).sum()/far.sum().clamp_min(1)
    loss = pos + 1.5*neg + tracing*trace + .2*extra
    return loss, dict(positive=float(pos.detach()), negative=float(neg.detach()),
                      tracing=float(trace.detach()), outside=float(extra.detach()))


def scores(prob, target, threshold=.5, margin=24):
    p=prob[margin:-margin, margin:-margin]>=threshold
    t=target[margin:-margin, margin:-margin]>.5
    tp=float((p&t).sum());npred=float(p.sum());nt=float(t.sum())
    precision=tp/max(1,npred);recall=tp/max(1,nt)
    dp=cv2.dilate(p.astype(np.uint8),np.ones((3,3),np.uint8))>0
    dt=cv2.dilate(t.astype(np.uint8),np.ones((3,3),np.uint8))>0
    p1=float((p&dt).sum()/max(1,npred));r1=float((t&dp).sum()/max(1,nt))
    dist=cv2.distanceTransform((~t).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return dict(precision_exact=precision,recall_exact=recall,f1_exact=2*precision*recall/max(1e-9,precision+recall),
        precision_1px=p1,recall_1px=r1,f1_1px=2*p1*r1/max(1e-9,p1+r1),
        area_ratio=npred/max(1,nt),off_boundary_1px=float((p&~dt).sum()/max(1,npred)),
        mean_distance_native720=float(dist[p].mean()) if p.any() else None,
        target_fraction=float(t.mean()),predicted_fraction=float(p.mean()),fixed_threshold=threshold,
        tolerant_scores_not_one_to_one=True)


class Samples:
    def __init__(self, data, seed, scope):
        self.data=data;self.rng=random.Random(seed);self.cache={};self.scope=scope
        self.groups={}
        for i,r in enumerate(data['pairs']): self.groups.setdefault(r['source'],[]).append(i)
        self.fixed=[]
        # Fixed patches chosen by geometry before looking at prediction results.
        # Indoor long boundaries plus foliage; all are training-source images.
        for prefix,x,y in [('39716881525a',320,128),('f30b208e3685',704,384)]:
            i=next(i for i,r in enumerate(data['pairs']) if Path(r['current']).name.startswith(prefix))
            self.fixed.append((i,False,x,y,192))

    def image(self,i,nxt=False):
        key=(i,nxt)
        if key not in self.cache:
            r=self.data['pairs'][i];a=load_rgb(r['next' if nxt else 'current']);b=load_rgb(r['next_target' if nxt else 'current_target'])
            self.cache[key]=(a,b,edges(b))
        return self.cache[key]

    def batch(self,native=False):
        specs=[]
        if native:
            specs=[(self.fixed[0][0],False,0,0,0)]
        elif self.scope=='tiny': specs=self.fixed
        else:
            for _ in range(2):
                i=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))]);nxt=bool(self.rng.randrange(2))
                size=256;x=self.rng.randrange(0,1280-size+1)//4*4;y=self.rng.randrange(0,720-size+1)//4*4
                specs.append((i,nxt,x,y,size))
        xs=[];ys=[]
        for i,nxt,x,y,size in specs:
            src,_,lab=self.image(i,nxt)
            if size:src=src[y:y+size,x:x+size];lab=lab[y:y+size,x:x+size]
            xs.append(tensor(src));ys.append(torch.from_numpy(lab.copy())[None])
        return torch.stack(xs).cuda(),torch.stack(ys).cuda(),specs


class Trainer:
    def __init__(self,cfg,data):
        self.cfg=cfg;self.net=NativeContourNet(cfg['difference']).cuda();self.step=0
        self.opt=torch.optim.Adam(self.net.parameters(),lr=cfg['lr'])
        self.samples=Samples(data,cfg['seed'],cfg['scope'])

    def update(self,native=False):
        start=time.perf_counter();x,t,spec=self.samples.batch(native)
        self.opt.zero_grad(set_to_none=True);p=self.net(x)
        loss,terms=criterion(p,t,self.cfg['tracing']);assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),20,error_if_nonfinite=True)
        assert grad>0;self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**terms,grad=float(grad),samples=spec,
            native=native,seconds=time.perf_counter()-start,
            allocated_MiB=torch.cuda.max_memory_allocated()/2**20)

    def state(self,binding):
        return dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,config=self.cfg,binding=binding,
            rng=self.samples.rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())

    def restore(self,state,binding):
        assert state['binding']==binding and state['config']==self.cfg
        self.net.load_state_dict(state['model']);self.opt.load_state_dict(state['optimizer']);self.step=state['step']
        self.samples.rng.setstate(state['rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])


@torch.no_grad()
def evaluate(t,out,baseline=True):
    out.mkdir(exist_ok=True);records=[];panels=[]
    entries=[]
    for j,(i,n,x,y,size) in enumerate(t.samples.fixed):
        a,b,e=t.samples.image(i,n)
        entries.append((f'training_patch_{j}',a[y:y+size,x:x+size],b[y:y+size,x:x+size],e[y:y+size,x:x+size],True))
    for r in t.samples.data['validation']:
        a=load_rgb(r['source']);b=load_rgb(r['target'])
        entries.append((r['id'],a,b,edges(b),False))
    old=None
    if baseline:
        from .contour_region_net import ContourRegionNet
        old=ContourRegionNet().cuda().eval()
        ck=torch.load(ROOT/'sharp-contour-region-v2/fp32_checkpoint_001200.pt',map_location='cpu',weights_only=True)
        old.load_state_dict(ck['model'])
    for name,a,b,e,trained in entries:
        x=tensor(a)[None].cuda();prob=t.net(x).sigmoid()[0,0].cpu().numpy();source=edges(a)
        pred=Image.fromarray(np.uint8(np.clip(1-prob,0,1)*255));pred.save(out/(name+'_probability.png'))
        target=Image.fromarray(np.uint8((1-e)*255));target.save(out/(name+'_target.png'))
        binary=Image.fromarray(np.uint8((prob<.5)*255));binary.save(out/(name+'_binary.png'))
        row=dict(id=name,training_patch=trained,predicted=scores(prob,e),source_canny=scores(source,e))
        oldim=None
        if old is not None:
            op=F.interpolate((6*(old.encode(x)[:,7:]-.5)).sigmoid(),scale_factor=2,mode='nearest')[0,0].cpu().numpy()
            row['previous_model']=scores(op,e);oldim=Image.fromarray(np.uint8((1-op)*255));oldim.save(out/(name+'_previous.png'))
        records.append(row)
        if trained: crops=[(0,0,192,192)]
        elif name=='val_stairs_090':crops=[(140,360,460,680),(420,180,740,500)]
        elif name=='val_plaza_090':crops=[(80,380,400,700)]
        else:crops=[]
        for box in crops:
            panels.append((name,box,[Image.fromarray(a),Image.fromarray(b),target,oldim,pred,binary]))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1920,len(panels)*358),'#202630');draw=ImageDraw.Draw(board)
    titles=['原片','教师画面','教师边界标签','上一版预测','本次预测（无细化）','本次固定0.5阈值']
    for j,(name,box,ims) in enumerate(panels):
        for k,im in enumerate(ims):
            draw.text((320*k+3,358*j+2),titles[k],font=font,fill='white')
            if im is not None:board.paste(im.crop(box).convert('RGB'),(320*k,358*j+36))
    board.save(out/'comparison.png')
    report=dict(step=t.step,scope=t.cfg['scope'],records=records,label=LABEL,
        no_prediction_nms=True,no_output_filter=True,not_style_output=True,reviewed_validation_not_blind=True)
    write_json(out/'report.json',report)
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train','evaluate'])
    ap.add_argument('--name',default='native-contour-tiny-v1');ap.add_argument('--scope',choices=['tiny','all'],default='tiny')
    ap.add_argument('--steps',type=int,default=800);ap.add_argument('--resume',type=Path);ap.add_argument('--initial',type=Path)
    ap.add_argument('--plain',action='store_true');ap.add_argument('--tracing',type=float,default=.35)
    a=ap.parse_args();configure_workspace();contract_guard();set_reproducible(2026100481);cv2.setNumThreads(2);torch.set_num_threads(4)
    out=ROOT/a.name
    if a.mode=='prepare':
        out.mkdir(exist_ok=False);shutil.copy2(DATA,out/'data.json');data=json.loads(DATA.read_text(encoding='utf-8'))
        hashes={}
        for r in data['pairs']:
            for key in ['current','current_target','next','next_target']:
                assert file_hash(r[key])==r[key+'_sha256'];hashes[r[key]]=r[key+'_sha256']
        for r in data['validation']:
            for key in ['source','target']:assert file_hash(r[key])==r[key+'_sha256'];hashes[r[key]]=r[key+'_sha256']
        assert set(data['training_sources']).isdisjoint(r['raw_source'] for r in data['validation'])
        cfg=dict(seed=2026100481,scope=a.scope,steps=a.steps,lr=.002 if a.scope=='tiny' else .0008,
            difference=not a.plain,tracing=a.tracing,label=LABEL,fp32=True,amp=False,tf32=False,
            initial=str(a.initial) if a.initial else None,initial_sha256=file_hash(a.initial) if a.initial else None)
        write_json(out/'config.json',cfg)
        paths=[Path(__file__),Path(__file__).with_name('native_contour_net.py'),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('engine.py')]
        binding=dict(code={str(p):file_hash(p) for p in paths},config=file_hash(out/'config.json'),data=file_hash(out/'data.json'),images=hashes)
        write_json(out/'binding.json',binding);(out/'source').mkdir()
        for p in paths:shutil.copy2(p,out/'source'/p.name)
        print('PREPARED',a.name,flush=True);return
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));binding=json.loads((out/'binding.json').read_text(encoding='utf-8'))
    assert all(file_hash(k)==v for k,v in binding['code'].items())
    assert file_hash(out/'config.json')==binding['config'] and file_hash(out/'data.json')==binding['data']
    data=json.loads((out/'data.json').read_text(encoding='utf-8'));t=Trainer(cfg,data)
    if cfg['initial']:
        assert file_hash(cfg['initial'])==cfg['initial_sha256']
        t.net.load_state_dict(torch.load(cfg['initial'],map_location='cpu',weights_only=True)['model'])
    if a.mode=='preflight':
        rows=[t.update(native=True),t.update()]
        torch.save(t.state(binding),out/'preflight_resume.pt')
        ref=t.update();expected=copy.deepcopy(t.net.state_dict())
        t.restore(torch.load(out/'preflight_resume.pt',map_location='cpu',weights_only=True),binding);replay=t.update()
        err=max(float((v-expected[k]).abs().max()) for k,v in t.net.state_dict().items());assert err<=1e-7 and ref['samples']==replay['samples']
        x,_,_=t.samples.batch(native=True)
        with torch.no_grad():fold_error=float((t.net(x)-t.net.folded()(x)).abs().max())
        assert fold_error<=1e-6
        target=torch.zeros(1,1,64,64,device='cuda');target[:,:,:,32]=1
        exact=torch.where(target>0,10.,-10.);wide=torch.where(F.max_pool2d(target,5,1,2)>0,10.,-10.)
        le=float(criterion(exact,target)[0]);lw=float(criterion(wide,target)[0]);ls=float(criterion(exact.roll(3,-1),target)[0]);assert le<lw and le<ls
        budget=t.net.budget();assert budget['convolution_weights']<=10256 and budget['ideal_macs']<=966758400
        write_json(out/'preflight.json',dict(status='PASS',binding=binding,updates=rows,reference=ref,replay=replay,
            resume_max_error=err,fold_max_error=fold_error,localization_loss=dict(exact=le,wide=lw,shifted=ls),budget=budget))
        print('PREFLIGHT_PASS',err,fold_error,budget['parameters'],budget['ideal_macs'],flush=True);return
    pre=json.loads((out/'preflight.json').read_text(encoding='utf-8'));assert pre['status']=='PASS' and pre['binding']==binding
    if a.resume:t.restore(torch.load(a.resume,map_location='cpu',weights_only=True),binding)
    if a.mode=='evaluate':evaluate(t,out/f'eval-{t.step:06d}');return
    assert a.resume or not (out/'metrics.jsonl').exists()
    write_json(out/'status.json',dict(status='RUNNING',step=t.step,binding=binding))
    with (out/'metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while t.step<min(a.steps,cfg['steps']):
            row=t.update(native=cfg['scope']=='all' and (t.step+1)%100==0);f.write(json.dumps(row)+'\n')
            if t.step%50==0:write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
            if t.step%400==0 or t.step==min(a.steps,cfg['steps']):
                torch.save(t.state(binding),out/f'checkpoint_{t.step:06d}.pt');evaluate(t,out/f'eval-{t.step:06d}')
    folded=t.net.folded();torch.save(dict(model=folded.state_dict(),config=cfg,step=t.step,binding=binding,
        folded=True,complete_style_network=False),out/'contour_fp32.pt')
    write_json(out/'status.json',dict(status='COMPLETED_BOUNDED_RUN',step=t.step,
        weights_sha256=file_hash(out/'contour_fp32.pt'),board_validated=False,style_complete=False))
    print('CONTOUR_TRAIN_DONE',t.step,flush=True)


if __name__=='__main__':main()
