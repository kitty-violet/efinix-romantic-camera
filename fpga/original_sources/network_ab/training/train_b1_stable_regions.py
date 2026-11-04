"""Clean teacher class islands and train on real adjacent-frame pairs.

Majority masks and optical flow are supervision only. Inference remains the
unaltered single-frame B1 byte program. No runtime region filter or frame blend.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import train_b1_luma_regions as region
from . import train_b1_learned_colour as colour
from .train_b1_distillation import read_flow
from .coherent_shared_model import warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=region.OUT/'stable-regions'
PARENT=region.OUT/'exact-head/checkpoint_003500_exact_head.pt'


class Trainer(region.Trainer):
    def __init__(self,cache,selected,steps):
        super().__init__(cache,selected,steps)
        self.net=region.integer_version(self.net)
        self.net.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['model']);self.new_optimizer()
        # Median of ordered binary planes preserves their ordering and removes
        # sub-7px speckles without making the target RGB boundary a blur.
        for key,t in self.targets.items():
            t[:,:6]=(F.avg_pool2d(F.pad(t[:,:6]/127,(3,3,3,3),mode='replicate'),7,1)>=.5).float()*127
        manifest=json.loads((colour.ROOT/'b1-deploy-preview-20261005-v1/data.json').read_text(encoding='utf-8'))
        index={v['row']['source']:v for v in cache};self.pairs=[]
        for p in manifest['pairs']:
            assert p['split']=='train' and index[p['current']]['row']['split']=='train' and index[p['next']]['row']['split']=='train'
            assert file_hash(p['flow'])==p['flow_sha256']
            flow,mask=read_flow(p['flow'])
            self.pairs.append(dict(current=index[p['current']],next=index[p['next']],
                flow=torch.tensor(flow.transpose(2,0,1).copy(),device='cuda'),mask=torch.tensor(mask[None].copy(),device='cuda')))

    def update(self,native=False):
        start=time.perf_counter();pair=self.rng.choice(self.pairs)
        if native:x0=y0=0;x1=1280;y1=720;rows=[pair['current']]
        else:
            x0=self.rng.randrange(241)*4;y0=self.rng.randrange(101)*4;x1=x0+320;y1=y0+320;rows=[pair['current'],pair['next']]
        x=torch.stack([r['source'][:,y0:y1,x0:x1] for r in rows]);t=torch.cat([self.targets[r['row']['id']][:,:,y0:y1,x0:x1] for r in rows])
        nodes=self.net(x,all_nodes=True);feature=region.soft_features(self.net,nodes);cut=lambda v:v[:,:,40:-40,40:-40]
        f0=cut(feature);label=cut(t);z=(f0[:,:6]-63.5)/10
        bce=F.binary_cross_entropy_with_logits(z,label[:,:6]/127)
        chroma=F.smooth_l1_loss(f0[:,6:]/32,label[:,6:]/32)
        binary=label[:,:6]/127
        maximum=F.max_pool2d(binary,7,1,3);minimum=-F.max_pool2d(-binary,7,1,3)
        interior=(maximum==minimum).float()
        margin=(F.relu(20-(2*binary-1)*(f0[:,:6]-63.5))/20*interior).mean()
        temporal=feature.new_zeros(())
        if not native:
            flow=pair['flow'][None,:,y0:y1,x0:x1];mask=pair['mask'][None,:,y0:y1,x0:x1]
            h,w=x.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
            valid=mask*((xx+flow[:,0]>=40)&(xx+flow[:,0]<w-40)&(yy+flow[:,1]>=40)&(yy+flow[:,1]<h-40))[:,None]
            probs=torch.cat((((feature[:,:6]-63.5)/10).sigmoid(),feature[:,6:]/127),1)
            diff=cut(probs[1:]-warp(probs[:1],flow));valid=cut(valid)
            temporal=(diff.abs()*valid).sum()/(valid.sum()*8).clamp_min(1)
        loss=bce+2*chroma+.2*margin+2*temporal
        for g in self.opt.param_groups:g['lr']=g['initial_lr']*(.5+.5*.5*(1+math.cos(math.pi*self.step/max(1,self.steps))))
        self.opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5,error_if_nonfinite=True)
        assert grad>0 and torch.isfinite(loss);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),bce=float(bce.detach()),chroma=float(chroma.detach()),
            margin=float(margin.detach()),temporal=float(temporal.detach()),grad=float(grad),crop=[x0,y0,x1,y1],
            samples=[r['row']['id'] for r in rows],seconds=time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1800);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100526);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=Trainer(cache,selected,a.steps)
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),steps=a.steps,
        code={str(p):file_hash(p) for p in [Path(__file__),Path(region.__file__),Path(colour.__file__)]},
        temporal_pairs=len(tr.pairs),mask_cleanup='7x7 majority of ordered target luma planes, supervision only',
        actual_B1=True,external_inference=False))
    for src in (__file__,region.__file__,colour.__file__):shutil.copy2(src,OUT/Path(src).name)
    native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update();expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items());assert error<1e-5 and ref['crop']==got['crop'] and ref['samples']==got['samples']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,real_pair_crop=crop,resume_error=error));print('STABLE_REGION_PREFLIGHT_PASS',error,flush=True)
    del tr,fresh;torch.cuda.empty_cache();tr=Trainer(cache,selected,a.steps)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('STABLE_REGION',tr.step,round(row['loss'],4),round(row['temporal'],4),flush=True)
            if tr.step%600==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');region.review(tr,data,selected,OUT/f'step_{tr.step:06d}')
    print('STABLE_REGION_COMPLETE',flush=True)


if __name__=='__main__':main()
