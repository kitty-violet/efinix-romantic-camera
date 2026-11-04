"""Learn B1's internal contour barrier; export and inspect the actual bytes.

Training labels may use a teacher and Canny. Deployed inference uses only the
unchanged B1 graph. The predicted barrier is consumed by both surface updates.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F

from .runtime import configure_workspace, contract_guard, write_json, file_hash
from .engine import set_reproducible
from .inspect_b1_structural_contours import build as build_initial
from .b1_trained_image import Parameters, encode, decode, runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .qat_b1_deploy_preview import QuantB1
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

ROOT = Path(r'E:\fpga\model-training-workspace\runs')
BASE = ROOT / 'b1-deploy-preview-20261005-v1'
OUT = ROOT / 'b1-learned-structural-gate-20261005-v1'
CROPS = [('val_stairs_090', (120,360,540,640), '草丛'),
         ('val_stairs_090', (780,70,1200,350), '建筑'),
         ('val_plaza_090', (0,180,420,460), '树冠'),
         ('val_plaza_210', (300,370,720,650), '人物')]


def contour_label(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 2.6)
    edge = cv2.Canny(gray, 18, 36, L2gradient=True)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edge, 8)
    keep = np.zeros(n, np.uint8)
    for k in range(1,n):
        if stats[k,4] >= 28 and max(stats[k,2],stats[k,3]) >= 14:
            keep[k] = 255
    thin = keep[labels]
    # Wide enough for the quarter-resolution barrier; no threshold is applied
    # to the network prediction during inference or publication.
    distance = cv2.distanceTransform(255-thin, cv2.DIST_L2, 5)
    target = np.maximum(0, 1-distance/5.)
    return thin, target.astype(np.float32)


def prepare(labeler=contour_label):
    OUT.mkdir(exist_ok=False)
    data = json.loads((BASE/'data.json').read_text(encoding='utf-8'))
    manifest = []
    targets = OUT/'labels'; targets.mkdir()
    rows = []
    for pair in data['pairs']:
        for side in ('current','next'):
            rows.append(dict(id=file_hash(pair[side])[:16], source=pair[side],
                             target=pair[side+'_target'], split='train'))
    rows += [dict(r, split='reviewed_validation') for r in data['validation']]
    seen = set()
    for r in rows:
        if r['source'] in seen:continue
        seen.add(r['source'])
        rgb = np.array(Image.open(r['target']).convert('RGB'))
        thin, soft = labeler(rgb)
        Image.fromarray(thin).save(targets/(r['id']+'_thin.png'))
        np.save(targets/(r['id']+'_soft.npy'),soft.astype(np.float16))
        manifest.append(dict(r, label=str(targets/(r['id']+'_soft.npy')),
            source_sha256=file_hash(r['source']), target_sha256=file_hash(r['target']),
            label_sha256=file_hash(targets/(r['id']+'_soft.npy'))))
    assert not {r['source'] for r in manifest if r['split']=='train'} & {r['source'] for r in manifest if r['split']!='train'}
    write_json(OUT/'data.json',manifest)
    write_json(OUT/'config.json',dict(steps=1200,crop=320,seed=2026100506,
        label='Frozen cartoon target: sigma2.6 Canny18/36, component28 extent14, distance width5',
        scope='B1 res1 internal barrier, then res2/res3 consume it; no outside inference operations',
        validation='Six previously reviewed frames, not blind generalization'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    board=Image.new('RGB',(1260,314*len(CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for row,(key,box,label) in enumerate(CROPS):
        r=next(r for r in manifest if r['id']==key)
        for col,(path,title) in enumerate([(r['source'],'原片 '+label),(r['target'],'冻结风格目标'),
                                          (targets/(key+'_thin.png'),'训练标签，非网络输出')]):
            board.paste(Image.open(path).convert('RGB').crop(box),(col*420,row*314+34))
            draw.text((col*420+5,row*314+4),title,font=font,fill='white')
    board.save(OUT/'label_review.png')
    print('LABELS_READY',len(manifest),flush=True)


def initial():
    p,res=build_initial(dict(penalty=5,threshold=1))
    # Learn the existing first residual. Keep colour lanes 0..5 intact and tie
    # the eight gate copies. Expand the feature workspace, not the B1 graph.
    for name in ('res1_conv1.dw','res1_conv2.dw'):
        old=p[name];w=np.array(old.weights);m=np.array(old.multiplier);b=np.array(old.bias)
        for c in range(6,24):
            if not w[c].any():w[c,4]=2;m[c]=65536
        p[name]=Parameters(tuple(tuple(int(v) for v in a) for a in w),tuple(int(v) for v in b),tuple(int(v) for v in m),True)
    old=p['res1_conv1.pw'];w=np.array(old.weights)
    for c in range(6,24):
        if not w[c].any():w[c,c]=2
    p['res1_conv1.pw']=Parameters(tuple(tuple(int(v) for v in a) for a in w),old.bias,old.multiplier,True)
    # More coefficient precision for learning, while retaining identical Q17
    # initial effective filters. This changes only coefficients/scales in RAM.
    for name in ('res1_conv1.dw','res1_conv1.pw','res1_conv2.dw','res1_conv2.pw'):
        old=p[name];w=np.array(old.weights)*8;b=np.array(old.bias)*8;m=np.array(old.multiplier)//8
        assert abs(w).max()<=127 and m.min()>0
        p[name]=Parameters(tuple(tuple(int(v) for v in a) for a in w),tuple(int(v) for v in b),tuple(int(v) for v in m),old.relu)
    raw=encode(p,res);return raw+bytes(20480-len(raw))


class Student(QuantB1):
    def __init__(self):
        super().__init__(runtime_from_image(initial()))
        self.quantized=False
        self.masks={}
        for kind,params in [('w',self.weight),('b',self.bias)]:
            for key,value in params.items():
                value.requires_grad_(key.startswith('res1_'))
                if not value.requires_grad:continue
                mask=torch.zeros_like(value)
                if key=='res1_conv2__pw':
                    if kind=='w':mask[24:32,6:24]=1
                    else:mask[24:32]=1
                elif kind=='w' and key=='res1_conv1__pw':mask[6:24,6:24]=1
                else:mask[6:24]=1
                self.register_buffer(kind+'_'+key+'_mask',mask)
                self.register_buffer(kind+'_'+key+'_initial',value.detach().clone())
                self.masks[(kind,key)]=True

    def project(self):
        with torch.no_grad():
            for kind,key in self.masks:
                value=(self.weight if kind=='w' else self.bias)[key]
                mask=getattr(self,kind+'_'+key+'_mask');base=getattr(self,kind+'_'+key+'_initial')
                value.copy_(value*mask+base*(1-mask))
                if kind=='w':value.clamp_(-127,127)
                if key=='res1_conv2__pw':value[25:32].copy_(value[24:25].expand_as(value[25:32]))

    def forward(self,x,all_nodes=False,full=False,stop_node=None):
        rnd=self.round if self.quantized else lambda v:v
        values={'input':rnd(x*128).clamp(-128,127)}
        for row in self.plan['nodes']:
            name=row['id'];key=name.replace('.','__');a=values[row['inputs'][0]]
            if key in self.weight:
                w=rnd(self.weight[key]).clamp(-127,127);b=rnd(self.bias[key])[None,:,None,None]
                if row['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                dot=F.conv2d(a,w,None,stride=row['stride'],groups=row['groups'])
                accum=(dot.double()+b).clamp(-(2**31),2**31-1)
                y=rnd(accum*getattr(self,key+'_m')/131072).clamp(-128,127).float()
                if row['activation']=='relu':y=y.clamp_min(0)
            elif row['kind']=='nearest_x2':y=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
            else:
                r=self.package['residual_encoding'][name];z=values[row['inputs'][1]]
                small,other=(a,z) if r['main_scaled'] else (z,a)
                aligned=rnd(small.double()*r['multiplier']/131072).clamp(-128,127)
                y=rnd((aligned+other.double())*r['post_multiplier']/131072).clamp(-128,127).float()
            values[name]=y
            if name==(stop_node or 'res1_add') and not full:break
        return values if all_nodes else values['res1_add'][:,24:25]

    def export(self):
        package=self.integer_package();p={}
        relus={r['id']:r['activation']=='relu' for r in self.plan['nodes'] if r['kernel']}
        for name,v in package['parameters'].items():
            p[name]=Parameters(tuple(tuple(row) for row in v['weights']),tuple(v['bias']),tuple(v['multipliers']),relus[name])
        raw=encode(p,package['residual_encoding']);return raw+bytes(20480-len(raw))


class Training:
    def __init__(self,data):
        self.net=Student().cuda().eval();self.opt=torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],lr=.025)
        self.data=data;self.rng=random.Random(2026100506);self.step=0
        self.cache=[(r,np.array(Image.open(r['source']).convert('RGB')),np.load(r['label']).astype(np.float32)) for r in data if r['split']=='train']

    def update(self,native=False):
        start=time.perf_counter();r,rgb,label=self.rng.choice(self.cache)
        h,w=rgb.shape[:2];size=320
        y=0 if native else self.rng.randrange(0,h-size+1);x=0 if native else self.rng.randrange(0,w-size+1)
        hh,ww=(h,w) if native else (size,size)
        source=torch.from_numpy(rgb[y:y+hh,x:x+ww].copy().transpose(2,0,1)).float()[None].cuda()/128-1
        target=torch.from_numpy(label[y:y+hh,x:x+ww].copy())[None,None].cuda()
        # Stride2 twice corresponds to input positions 0,4,8,... . Match that
        # geometry instead of an area-resize with a different pixel centre.
        target=target[:,:,::4,::4]
        pred=self.net(source);p=pred[:,:,6:-6,6:-6];t=target[:,:,6:-6,6:-6]
        pos=t>.12;neg=~pos
        desired=t*28-4
        error=F.smooth_l1_loss(p/16,desired/16,reduction='none')
        positive=(error*pos).sum()/pos.sum().clamp_min(1)
        negative=(error*neg).sum()/neg.sum().clamp_min(1)
        loss=positive+1.5*negative
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for kind,key in self.net.masks:
            value=(self.net.weight if kind=='w' else self.net.bias)[key]
            if value.grad is not None:value.grad.mul_(getattr(self.net,kind+'_'+key+'_mask'))
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.net.parameters() if p.requires_grad],10,error_if_nonfinite=True)
        assert torch.isfinite(loss) and norm>0
        self.opt.step();self.net.project();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss),positive=float(positive),negative=float(negative),grad=float(norm),
                    sample=r['id'],crop=[x,y,ww,hh],seconds=time.perf_counter()-start,quantized=self.net.quantized)

    def save(self,path):
        torch.save(dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,random=self.rng.getstate(),
            quantized=self.net.quantized,code_sha256=file_hash(__file__),data_sha256=file_hash(OUT/'data.json')),path)

    def restore(self,path):
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['code_sha256']==file_hash(__file__) and s['data_sha256']==file_hash(OUT/'data.json')
        self.net.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.step=s['step'];self.rng.setstate(s['random']);self.net.quantized=s['quantized']


@torch.no_grad()
def evaluate(tr,data,tag):
    dest=OUT/tag;dest.mkdir(exist_ok=False);raw=tr.net.export();(dest/'coefficients-padded.bin').write_bytes(raw)
    p,r=decode(raw);assert encode(p,r)==raw[:17792]
    net=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()));rows=[]
    was=tr.net.quantized;tr.net.quantized=True
    for row in data:
        if row['split']=='train':continue
        rgb=np.array(Image.open(row['source']).convert('RGB'));x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()-128
        nodes,_=net(x,True);first=getattr(tr.net,'first_gate_node','res2_conv1.dw');rectified=getattr(tr.net,'rectified_gate_node','res2_add');actual=nodes[first][:,24:25]
        learned=tr.net(x/128,True,True)
        error=max(float((learned[k]-v).abs().max()) for k,v in nodes.items());assert error==0
        assert torch.equal(nodes[rectified][:,24:25],actual)
        gate=actual[0,0].cpu().numpy();np.save(dest/(row['id']+'_gate.npy'),gate.astype(np.int8))
        Image.fromarray(np.clip(gate*(255/24),0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).save(dest/(row['id']+'_gate.png'))
        # Keep an unaltered RGB diagnostic. This is not a colour acceptance.
        out=(nodes['rgb_head'][0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
        Image.fromarray(out).save(dest/(row['id']+'_raw_rgb.png'))
        t=np.load(row['label']).astype(float)[::4,::4]>.12;positive=gate>4
        rows.append(dict(id=row['id'],export_node_error=error,precision=float((positive&t).sum()/max(positive.sum(),1)),
            recall=float((positive&t).sum()/max(t.sum(),1)),gate_nonzero=float((gate>0).mean())))
    tr.net.quantized=was
    op,rp=oracle_records(p);contract,_=contract_guard();x=np.random.default_rng(67).integers(-128,128,(12,12,3),dtype=np.int16)
    ref=execute_integer(contract,x.tolist(),op,rp);nodes,_=net(torch.tensor(x.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
    errors={k:int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in nodes.items()};assert max(errors.values())==0
    write_json(dest/'verification.json',dict(step=tr.step,trained_internal_gate=True,transport_bytes=len(raw),
        weight_sha256=file_hash(dest/'coefficients-padded.bin'),every_node_oracle=errors,validation=rows,
        graph_unchanged=True,extra_postprocessing=False,visual_accepted=False,actual_board=False))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(1260,314*len(CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(CROPS):
        row=next(r for r in data if r['id']==key)
        for col,(path,title) in enumerate([(row['source'],'原片 '+label),(OUT/'labels'/(key+'_thin.png'),'监督标签'),
                                          (dest/(key+'_gate.png'),'权重重载：实际轮廓约束')]):
            board.paste(Image.open(path).convert('RGB').crop(box),(col*420,i*314+34));draw.text((col*420+5,i*314+4),title,font=font,fill='white')
    board.save(dest/'native_contour_review.png');print('EVALUATE',tag,rows,flush=True)


def train(preflight=False,steps=1200):
    data=json.loads((OUT/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):assert file_hash(r[key])==r[key+'_sha256']
    tr=Training(data)
    if preflight:
        dest=OUT/'preflight';dest.mkdir(exist_ok=False)
        native=tr.update(True);crop=tr.update();tr.save(dest/'resume.pt');expected=tr.update()
        state={k:v.clone() for k,v in tr.net.state_dict().items()};tr.restore(dest/'resume.pt');repeated=tr.update()
        error=max(float((v-state[k]).abs().max()) for k,v in tr.net.state_dict().items());assert error<1e-6 and expected['crop']==repeated['crop']
        tr.net.quantized=True;qat=tr.update();evaluate(tr,data,'preflight-export')
        write_json(dest/'result.json',dict(status='PASS',native=native,crop=crop,qat=qat,resume_error=error,code_sha256=file_hash(__file__)))
        print('PREFLIGHT_PASS',error,flush=True);return
    pre=json.loads((OUT/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['code_sha256']==file_hash(__file__)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<steps:
            tr.net.quantized=tr.step>=steps*2//3
            lr=.025 if not tr.net.quantized else .012
            for group in tr.opt.param_groups:group['lr']=lr
            row=tr.update((tr.step+1)%100==0);log.write(json.dumps(row)+'\n')
            if tr.step%25==0:print('LEARN_GATE',tr.step,round(row['loss'],4),round(row['seconds'],3),'QAT',tr.net.quantized,flush=True)
            if tr.step%400==0 or tr.step==steps:tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');evaluate(tr,data,f'step_{tr.step:06d}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train']);ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100506);cv2.setNumThreads(2);contract_guard()
    if a.mode=='prepare':prepare()
    else:train(a.mode=='preflight',a.steps)


if __name__=='__main__':main()
