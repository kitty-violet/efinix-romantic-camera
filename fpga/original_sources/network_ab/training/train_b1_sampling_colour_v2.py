"""13 learned value regions, sampling-phase training and a trainable B1 decoder.

All output operations are the existing B1 graph. Labels/flow only supervise
training; they are never run alongside the exported style coefficient image.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_learned_colour as colour
from . import train_b1_luma_regions as old
from .audit_b1_sampling_v2 import OUT,PARENT
from .train_b1_distillation import read_flow
from .coherent_shared_model import warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

LOW=(12,28,44,60,77,98)
HIGH=(20,36,52,68,87,110)
PARENT_CKPT=PARENT/'checkpoint_001800.pt'


def labels(t):
    v=(t+1)*64;r,g,b=v[:,0:1],v[:,1:2],v[:,2:3]
    y=(r+2*g+b)/4;co=(r-b)/2+64;cg=(-r+2*g-b)/4+64
    levels=[]
    for a,b in zip(LOW,HIGH):
        pair=torch.cat([(y>a).float(),(y>b).float()],1)
        # Three-pixel majority removes isolated one-pixel class noise while
        # avoiding the prior seven-pixel erosion of corners and small people.
        pair=(F.avg_pool2d(F.pad(pair,(1,1,1,1),mode='replicate'),3,1)>=.5).float()
        levels.append(pair.sum(1,keepdim=True)*63.5)
    return torch.cat(levels+[co,cg],1)


@torch.no_grad()
def initialize():
    net=colour.ColourB1(revive=False,source=PARENT/'step_001800/coefficients-padded.bin').cuda().eval()
    net.load_state_dict(torch.load(PARENT_CKPT,map_location='cpu',weights_only=True)['model'])
    # Restore feature-space FP32 units from the parent's binary deployment head.
    net.weight['up2_high_dw'][:6]*=127
    net.weight['up2_high_dw'][6:]*=2
    net.weight['rgb_head'][:,:6]/=127
    net.weight['rgb_head'][:,6:]/=2
    net.quantized=False
    for k in net.weight:
        net.weight[k].requires_grad_(k!='rgb_head')
        net.bias[k].requires_grad_(k not in ('up2_high_dw','rgb_head'))
    return net


@torch.no_grad()
def quantize(net):
    if net.quantized:return copy.deepcopy(net)
    q=copy.deepcopy(net)
    q.weight['up2_high_dw'][:6]*=2/127;q.weight['up2_high_dw'][6:]/=2
    q.weight['rgb_head'][:,:6]*=127/2;q.weight['rgb_head'][:,6:]*=2
    q.calibrate()
    # RGB coefficients are all integer multiples of 1/2. Preserve their exact
    # colour conversion; generic max/112 calibration introduced a proven bias.
    q.rgb_head_m.fill_(65536)
    q.up2_high_dw_m[:,:6].fill_(16)
    q.up2_high_dw_m[:,6:].fill_(1024)
    q.quantized=True
    return q


def features(net,nodes):
    if not net.quantized:return nodes['up2_high_dw']
    # Actual quantized coefficients before final RQ. The supervision derivative
    # must follow the trained spatial kernel, not the old hard-coded cross blur.
    w,b=net.weight['up2_high_dw'],net.bias['up2_high_dw']
    s=(net.up2_high_dw_m.flatten()/131072).float()
    w=net.round(w/s[:,None,None,None]).clamp(-127,127)*s[:,None,None,None]
    b=net.round(b/s)*s
    z=F.conv2d(F.pad(nodes['up2_nn'],(1,1,1,1),mode='replicate'),w,b,groups=8)
    scale=z.new_tensor([127/2]*6+[2,2])[None,:,None,None]
    return z*scale


def ordinal(f):
    # Two ordered decisions per feature produce three learned region states.
    return torch.stack([(f[:,:6]-31.75)/8,(f[:,:6]-95.25)/8],2).flatten(1,2)


class Trainer(colour.Trainer):
    def __init__(self,cache,selected,steps,qat=False):
        self.cached,self.selected,self.mode,self.steps=cache,selected,'sampling_colour_v2',steps
        self.net=initialize()
        if qat:self.net=quantize(self.net)
        self.rng,self.step=random.Random(2026100528),0
        self.targets={r['row']['id']:self.make_targets(r['target'][None]) for r in cache}
        index={r['row']['source']:r for r in cache}
        manifest=json.loads((colour.ROOT/'b1-deploy-preview-20261005-v1/data.json').read_text(encoding='utf-8'))
        self.pairs=[]
        for p in manifest['pairs']:
            assert p['split']=='train' and index[p['current']]['row']['split']=='train' and index[p['next']]['row']['split']=='train'
            assert file_hash(p['flow'])==p['flow_sha256']
            flow,mask=read_flow(p['flow'])
            self.pairs.append(dict(current=index[p['current']],next=index[p['next']],
                flow=torch.tensor(flow.transpose(2,0,1).copy(),device='cuda'),mask=torch.tensor(mask[None].copy(),device='cuda')))
        self.new_optimizer()

    def new_optimizer(self):
        groups=[]
        for kind,params in [('w',self.net.weight),('b',self.net.bias)]:
            for key,p in params.items():
                if not p.requires_grad:continue
                lr=(.00012 if kind=='w' else .008)*(.3 if self.net.quantized else 1)
                if key=='up2_high_dw' and self.net.quantized:
                    # Gradients use deployment units; parameter steps must too.
                    lr*=2/127
                groups.append(dict(params=[p],lr=lr,initial_lr=lr))
        self.opt=torch.optim.Adam(groups)

    @torch.no_grad()
    def constrain_decoder(self):
        w=self.net.weight['up2_high_dw']
        scale=w.new_tensor([getattr(self,'max_region_state',2)/127]*6+[.5,.5]) if self.net.quantized else w.new_ones(8)
        a=w/scale[:,None,None,None]
        negative=a.clamp_max(0)
        # Bounded negative side lobes allow edge refinement without allowing a
        # ternary luma channel to overflow to a fourth, unintended state.
        mass=(-negative).flatten(1).sum(1)
        negative*=torch.minimum(torch.ones_like(mass),.1/mass.clamp_min(1e-6))[:,None,None,None]
        positive=a.clamp_min(0)
        positive*=((1-negative.flatten(1).sum(1))/positive.flatten(1).sum(1).clamp_min(1e-6))[:,None,None,None]
        w.copy_((positive+negative)*scale[:,None,None,None])

    def colour_loss(self,f,t):
        return F.smooth_l1_loss(f[:,6:]/32,t[:,6:]/32)

    def make_targets(self,t):
        return labels(t)

    def get_features(self,nodes):
        return features(self.net,nodes)

    def region_logits(self,f):
        return ordinal(f)

    def region_truth(self,t):
        return torch.stack([(t[:,:6]>0).float(),(t[:,:6]>64).float()],2).flatten(1,2)

    def lattice_loss(self,nodes,target):
        return target.new_zeros(())

    def stability_margin(self,logits,truth):
        return logits.new_zeros(())

    def rgb_temporal(self,nodes,flow,valid):
        return valid.new_zeros(())

    def appearance_loss(self,nodes,target):
        return target.new_zeros(())

    def update(self,native=False):
        start=time.perf_counter();pair=self.rng.choice(self.pairs)
        if native:x0=y0=0;x1=1280;y1=720;rows=[pair['current']]
        else:
            # Exercise every subpixel phase of both stride-2 stages.
            x0=self.rng.randrange(961);y0=self.rng.randrange(401);x1=x0+320;y1=y0+320;rows=[pair['current'],pair['next']]
        x=torch.stack([r['source'][:,y0:y1,x0:x1] for r in rows])
        target=torch.cat([self.targets[r['row']['id']][:,:,y0:y1,x0:x1] for r in rows])
        nodes=self.net(x,all_nodes=True);f=self.get_features(nodes);cut=lambda a:a[:,:,40:-40,40:-40]
        logits=self.region_logits(cut(f));t=cut(target)
        truth=self.region_truth(t)
        bce=F.binary_cross_entropy_with_logits(logits,truth)
        chroma=self.colour_loss(cut(f),t)
        # Match teacher boundary changes explicitly, normalized separately from
        # flat interiors so simply blurring the image cannot win this term.
        prob=logits.sigmoid();edges=[]
        for dim in (-1,-2):
            dp,dt=torch.diff(prob,dim=dim),torch.diff(truth,dim=dim)
            boundary=dt.abs()>0
            edges.append(((dp-dt).abs()*boundary).sum()/boundary.sum().clamp_min(1))
        boundary=torch.stack(edges).mean()
        temporal=f.new_zeros(());rgb_temporal=f.new_zeros(())
        if not native:
            flow=pair['flow'][None,:,y0:y1,x0:x1];mask=pair['mask'][None,:,y0:y1,x0:x1]
            h,w=x.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
            valid=mask*((xx+flow[:,0]>=40)&(xx+flow[:,0]<w-40)&(yy+flow[:,1]>=40)&(yy+flow[:,1]<h-40))[:,None]
            z=torch.cat((self.region_logits(f).sigmoid(),f[:,6:]/127),1)
            temporal=(cut((z[1:]-warp(z[:1],flow)).abs())*cut(valid)).sum()/(cut(valid).sum()*z.shape[1]).clamp_min(1)
            rgb_temporal=self.rgb_temporal(nodes,flow,valid)
        phase=f.new_zeros(())
        if not native and self.step%8==0:
            shifted=F.pad(x[:1],(1,0,0,0),mode='replicate')[:,:,:,:x.shape[-1]]
            other=self.get_features(self.net(shifted,all_nodes=True))
            phase=F.l1_loss(other[:,:,40:-40,41:-40]/127,f[:1,:,40:-40,40:-41]/127)
        lattice=self.lattice_loss(nodes,target)
        margin=self.stability_margin(logits,truth)
        appearance=self.appearance_loss(nodes,target)
        loss=bce+2*chroma+.25*boundary+getattr(self,'temporal_weight',2)*temporal+.4*phase+lattice+margin+rgb_temporal+appearance
        for g in self.opt.param_groups:g['lr']=g['initial_lr']*(.5+.25*(1+math.cos(math.pi*min(self.step,self.steps)/max(self.steps,1))))
        self.opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True)
        assert grad>0 and torch.isfinite(loss);self.opt.step();self.constrain_decoder();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),bce=float(bce.detach()),chroma=float(chroma.detach()),
            boundary=float(boundary.detach()),temporal=float(temporal.detach()),phase=float(phase.detach()),lattice=float(lattice.detach()),
            margin=float(margin.detach()),rgb_temporal=float(rgb_temporal.detach()),
            appearance=float(appearance.detach()),
            grad=float(grad),crop=[x0,y0,x1,y1],samples=[r['row']['id'] for r in rows],seconds=time.perf_counter()-start)


@torch.inference_mode()
def review(tr,data,selected,dest):
    saved=tr.net;tr.net=quantize(saved)
    colour.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
    board=Image.new('RGB',(1680,1248),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    fallback=colour.IntegerB1(colour.runtime_from_image((PARENT/'step_001800/coefficients-padded.bin').read_bytes()))
    records=[]
    for i,(key,box,label) in enumerate(colour.CROPS):
        row=next(v for v in tr.cached if v['row']['id']==key);x=row['source'][None]
        nodes=tr.net(x,all_nodes=True);g=nodes['up2_high_dw'][:,:6]
        assert g.min()>=0 and g.max()<=2,(g.min(),g.max())
        old_rgb=fallback(x*128)/128;p=nodes['rgb_head']/128
        for j,(name,img) in enumerate([('冻结教师',colour.rgb_image(row['target'])),('V1兜底',colour.rgb_image(old_rgb[0])),
                                       ('V2实际整数包',colour.rgb_image(p[0])),('原片',colour.rgb_image(x[0]))]):
            board.paste(img.crop(box),(420*j,312*i+32));draw.text((420*j+4,312*i+4),name,font=font,fill='white')
        t=tr.targets[key][:,:6]/63.5
        records.append(dict(id=key,ternary_accuracy=float((g==t).float().mean()),
                            middle_state_fraction=float((g==1).float().mean()),feature_min=float(g.min()),feature_max=float(g.max())))
    board.save(dest/'v1_v2_teacher.png')
    write_json(dest/'regions.json',dict(records=records,levels=[8,16,24,32,40,48,56,64,72,82,92,104,116],
        pixel_scale='half RGB range; double for 8-bit luminance',visual_accepted=False))
    tr.net=saved


def preflight(cache,selected,steps,path,qat=False,state=None):
    tr=Trainer(cache,selected,steps,qat)
    if state is not None:tr.net.load_state_dict(state)
    native=tr.update(True);crop=tr.update();tr.save(path/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,steps,qat);fresh.restore(path/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error<1e-5 and ref['crop']==got['crop'] and ref['samples']==got['samples']
    write_json(path/'preflight.json',dict(status='PASS',native=native,real_pair_crop=crop,resume_error=error,quantized=qat))
    print('V2_PREFLIGHT_PASS',qat,error,flush=True)
    del tr,fresh;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--qat',type=int,default=800);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100528);contract_guard();run=OUT/'learned';run.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data()
    write_json(run/'binding.json',dict(parent=str(PARENT_CKPT),parent_sha256=file_hash(PARENT_CKPT),
        new_fp32_steps=a.steps,new_qat_steps=a.qat,parent_cumulative_steps=5300,real_b1_graph=True,
        no_external_inference=True,teacher_only_labels=True,code={str(p):file_hash(p) for p in [Path(__file__),Path(colour.__file__)]},
        sampling_changes='all crop phases; learned bounded final DW; actual-kernel QAT surrogate; explicit teacher boundary loss',
        colour_changes='six ternary features, thirteen luminance levels, same two chroma features; 3x3 label majority'))
    shutil.copy2(__file__,run/Path(__file__).name)
    preflight(cache,selected,a.steps,run)
    tr=Trainer(cache,selected,a.steps)
    with (run/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<a.steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('V2_FP32',tr.step,round(row['loss'],4),round(row['boundary'],4),flush=True)
            if tr.step%1000==0 or tr.step==a.steps:
                tr.save(run/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,run/f'step_{tr.step:06d}')
        q=quantize(tr.net);state=copy.deepcopy(q.state_dict());del q,tr;torch.cuda.empty_cache()
        qdir=run/'qat';qdir.mkdir();preflight(cache,selected,a.qat,qdir,True,state)
        tr=Trainer(cache,selected,a.qat,True);tr.net.load_state_dict(state)
        while tr.step<a.qat:
            row=tr.update();log.write(json.dumps(row,ensure_ascii=False)+'\n')
            if tr.step%100==0:print('V2_QAT',tr.step,round(row['loss'],4),flush=True)
        tr.save(qdir/f'checkpoint_{tr.step:06d}.pt');review(tr,data,selected,qdir/'package')
    print('V2_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
