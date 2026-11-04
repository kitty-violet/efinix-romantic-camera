"""C3 ordered regions with a source-bounded temporal objective.

Never use framewise teacher flicker as permission for output flicker. Train
actual palette colors with a component straight-through derivative, retaining
continuous color supervision and real adjacent motion/occlusion masks.
"""
import argparse,copy,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .component_region_net import ComponentNet,plan,decode
from .ordinal_coarse_net import OrdinalCoarseNet
from .train_detail_rgb import Training as DataTraining
from .train_luma_regions import ROOT,RUN,centers,components,reconstruct,labels,warp
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def straight_colors(p):
    parts=[]
    for i,c in enumerate(centers(p)):
        z=p[:,i];hard=c[(z[:,None]-c[None,:,None,None]).abs().argmin(1)]
        parts.append(z+(hard-z).detach())
    return reconstruct(*parts)


def prepare(out,initial):
    s=torch.load(initial,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in s['binding']['code'].items())
    assert s['scales'] is None;out.mkdir(exist_ok=False)
    source=OrdinalCoarseNet().eval();source.load_state_dict(s['model']);net=ComponentNet().eval();net.from_ordinal(source)
    torch.save(dict(model=net.state_dict(),source=str(initial),source_sha256=file_hash(initial)),out/'initial.pt')
    (out/'data.json').write_bytes((initial.parent/'data.json').read_bytes());write_json(out/'graph.json',plan())
    cfg=copy.deepcopy(s['config']);cfg.update(seed=2026100449,initial=str(out/'initial.pt'),initial_sha256=file_hash(out/'initial.pt'),fp32_steps=4000,fp32_lr=.0005,
        objective_weights=dict(ce=.5,color=4,flat=24,region=8,temporal=8,phase=4),
        region_objective='C3 nearest levels with direct component straight-through derivative',
        temporal_reference='Original unaugmented source residual only, capped at 3 RGB levels +1 tolerance; never teacher delta',
        curriculum_probability=.5,source_texture_gain=[.5,1.5],synthetic_noise_rgb255=[0,8])
    write_json(out/'config.json',cfg);binding=copy.deepcopy(s['binding'])
    for f in [Path(__file__),Path(__file__).with_name('component_region_net.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(DataTraining):
    def __init__(self,out):
        super().__init__(out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        self.net=ComponentNet().cuda().eval();self.net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);original=x;m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        curriculum=self.rng.random()<self.cfg['curriculum_probability'];aug=self.rng.random()<.5
        if curriculum:
            gain=self.rng.uniform(*self.cfg['source_texture_gain']);noise=self.rng.uniform(*self.cfg['synthetic_noise_rgb255'])
            coarse=F.avg_pool2d(F.pad(x,(3,3,3,3),mode='replicate'),7,stride=1)
            x=(t+gain*(x-coarse)+torch.randn_like(x)*(noise/128)).clamp(-1,127/128);spec=[*spec,'curriculum',gain,noise]
        if aug:x=(x+torch.randn_like(x)*2/128).clamp(-1,127/128)
        p=self.net(x);y=straight_colors(p);lab=labels(t);cs=centers(p);ce=[]
        for i,(c,l,temp) in enumerate(zip(cs,lab,[128,256,512])):
            logits=temp*(2*c[None,:,None,None]*p[:,i,None]-c.square()[None,:,None,None]);ce.append(F.cross_entropy(cut(logits),cut(l)))
        ce=sum(ce)/3;code=torch.stack(lab,1);target=reconstruct(*[c[l] for c,l in zip(cs,lab)])
        color=F.l1_loss(cut(reconstruct(*p.unbind(1))),cut(t))+.5*F.l1_loss(cut(y),cut(target));flat=[];region=[];tc=components(t)
        for d in [1,3,7]:
            for axis in [-1,-2]:
                aa=[slice(None)]*4;bb=aa.copy();aa[axis]=slice(None,-d);bb[axis]=slice(d,None);aa=tuple(aa);bb=tuple(bb)
                same=(code[aa]==code[bb]).all(1,keepdim=True);w=cut(same);den=w.sum().clamp_min(1)
                flat.append((cut(F.relu((y[aa]-y[bb]).abs().mean(1,keepdim=True)-1/128))*w).sum()/den)
                err=F.relu((p[aa]-p[bb]).abs().mean(1,keepdim=True)-(tc[aa]-tc[bb]).abs().mean(1,keepdim=True)-.5/128)
                region.append((cut(err)*w).sum()/den)
        flat=sum(flat)/len(flat);region=sum(region)/len(region)
        with torch.no_grad():
            xd=(original[1:]-warp(original[:1],flow)).abs().mean(1,keepdim=True);valid=mask*(xd<8/128);allowed=xd.clamp(max=3/128)+1/128
        temporal=(cut(F.relu((y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True)-allowed))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];q=straight_colors(self.net(shift))
        phase=(q[...,m+sy:-m,m+sx:-m]-y[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        losses=dict(ce=ce,color=color,flat=flat,region=region,temporal=temporal,phase=phase);ramp=min(1,(self.step+1)/150)
        loss=sum(v*self.cfg['objective_weights'][k]*(ramp if k not in ('ce','color') else 1) for k,v in losses.items())
        lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},grad=float(norm),lr=lr,native=native,curriculum=curriculum,noise=aug,sample=spec,seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'{self.mode}_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();p=decode(self.net(x));Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'))
            q=decode(self.net(F.pad(x,(1,0,0,0),mode='replicate')[:,:,:,:1280]));rows.append(dict(id=r['id'],phase_rgb255=float((q[:,:,32:-32,33:-32]-p[:,:,32:-32,32:-33]).abs().mean()*128)))
        write_json(folder/'scores.json',dict(records=rows,style_accepted=False,not_blind=True))
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(960,894),'#18202a');d=ImageDraw.Draw(board);root=ROOT/'runs'/RUN
        for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
            r=next(r for r in self.data['validation'] if r['id']==key)
            for i,(path,title) in enumerate([(r['source'],'原片'),(root/'intermediate-target-screen'/f'{key}_0.008.png','区域目标'),(folder/(key+'.png'),f'C3 FP32 {self.step}步')]):
                d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
        board.save(folder/'comparison.png')


def preflight(out):
    tr=Training(out);calls=[];hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in tr.net.modules() if isinstance(m,torch.nn.Conv2d)]
    with torch.no_grad():v=tr.net(torch.zeros(1,3,720,1280,device='cuda'));assert v.shape==(1,3,720,1280)
    for h in hooks:h.remove()
    assert sum(calls)==plan()['macs'];tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,macs=sum(calls),convolutions=len(calls)));print('COMPONENT_PREFLIGHT_PASS',err,sum(calls),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--initial',type=Path);p.add_argument('--stop',type=int,default=4000);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/'runs'/RUN/'component-regions-fp32'
    if a.action=='prepare':assert a.initial;prepare(out,a.initial);return
    configure_workspace();contract_guard();set_reproducible(2026100449)
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('COMPONENT_TRAIN',tr.step,round(row['loss'],4),row['curriculum'],round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('COMPONENT_DONE',tr.step,flush=True)


if __name__=='__main__':main()
