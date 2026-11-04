"""Clean-region distillation in the existing 20-KiB candidate graph.

Train-only texture critic + teacher-feature supervision of the shallow skip.
The generator graph, parameter count, fixed palette and deployment ABI do not
change. No discriminator, teacher, blur or temporal postfilter is deployed.
Inspired by the separate texture/structure objectives in Wang & Yu CVPR2020;
this is a custom paired distillation experiment, not a reproduction of it.
"""
import argparse,copy,gc,json,math,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .train_detail_rgb import Training as DataTraining
from .train_b1_capacity import Training as CapacityTraining
from .b1_capacity_component_net import CapacityComponentNet,plan,budget
from .train_component_centers import spatial,target_components
from .train_component_regions import straight_colors
from .component_region_net import decode
from .train_luma_regions import ROOT,RUN,reconstruct,warp
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .audit_capacity_texture import edge_map,texture_scores
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class TextureCritic(nn.Module):
    def __init__(self):
        super().__init__();layers=[];ci=1
        for co,stride in [(16,1),(32,2),(64,2),(64,1)]:
            layers.extend([nn.utils.spectral_norm(nn.Conv2d(ci,co,3,stride,1)),nn.LeakyReLU(.2)]);ci=co
        layers.append(nn.utils.spectral_norm(nn.Conv2d(ci,1,3,1,1)));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x)


def clean_structure(y,target,margin):
    cut=lambda x:x[...,margin:-margin,margin:-margin]
    gt=edge_map(target).detach();interior=(F.max_pool2d((gt>4/128).float(),5,1,2)==0).float()
    gp=edge_map(y);clean=(cut(gp)*cut(interior)).sum()/cut(interior).sum().clamp_min(1)
    # Soft count of gradients, compared LOCALLY with the teacher. A flat
    # output cannot satisfy the separate signed-boundary and color losses.
    rho=lambda z:1-torch.exp(-(z/(4/128)).square())
    density=F.relu(F.avg_pool2d(cut(rho(gp)),16)-F.avg_pool2d(cut(rho(gt)),16)).mean()
    return dict(clean=clean,density=density)


def prepare(out,steps):
    initial=ROOT/'runs'/RUN/'b1-capacity-control-v2/capacity/fp32_checkpoint_002000.pt';s=torch.load(initial,map_location='cpu',weights_only=True)
    assert s['scales'] is None and all(file_hash(k)==v for k,v in s['binding']['code'].items())
    out.mkdir(exist_ok=False);(out/'data.json').write_bytes((initial.parent/'data.json').read_bytes());write_json(out/'graph.json',plan())
    cfg=copy.deepcopy(s['config']);cfg.update(seed=2026100461,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=steps,fp32_lr=.0005,
        objective_weights=dict(center=8,boundary=6,flat=4,clean=24,density=.6,feature=.6,adversarial=.08,temporal=6,phase=4),
        critic_lr=.00015,critic_patch=128,critic_warmup=100,feature_nodes=['native.pw','down.pw'],
        method='Paired structure + frozen clean-input skip features + train-only grayscale texture LSGAN',
        critic_deployed=False,teacher_deployed=False,inference_changed=False)
    write_json(out/'config.json',cfg);binding=copy.deepcopy(s['binding'])
    for f in [Path(__file__),Path(__file__).with_name('audit_capacity_texture.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    # Behavioral checks: extra texture is penalized, and blurred boundaries
    # cannot pass the complete structure objective as a clean cartoon.
    t=torch.zeros(1,3,96,96);t[:,:,:,48:]=.5
    noisy=t.clone();noisy[:,:,::2,::2]+=.125
    blurred=F.avg_pool2d(F.pad(t,(4,)*4,mode='replicate'),9,1)
    clean=clean_structure(t,t,8);noise=clean_structure(noisy,t,8)
    assert float(clean['clean'])==0 and float(clean['density'])==0
    assert float(noise['clean'])>0 and float(noise['density'])>0
    assert float((edge_map(blurred)-edge_map(t)).abs().mean())>0
    write_json(out/'objective_checks.json',dict(perfect={k:float(v) for k,v in clean.items()},speckles={k:float(v) for k,v in noise.items()},blur_boundary_mismatch=float((edge_map(blurred)-edge_map(t)).abs().mean()),budget=budget()))
    print('CLEAN_PREPARED',flush=True)


class Training(CapacityTraining):
    def __init__(self,out):
        super().__init__(out)
        self.reference=CapacityComponentNet().cuda().eval();self.reference.load_state_dict(self.net.state_dict());self.reference.requires_grad_(False)
        self.critic=TextureCritic().cuda().train();self.dopt=torch.optim.Adam(self.critic.parameters(),lr=self.cfg['critic_lr'],betas=(.5,.99))
        self.fixed_critic_initial=None

    def state(self):
        s=super().state();s.update(critic=self.critic.state_dict(),critic_optimizer=self.dopt.state_dict());return s

    def restore(self,s):
        super().restore(s);self.critic.load_state_dict(s['critic']);self.dopt.load_state_dict(s['critic_optimizer'])

    def features(self,model,x):
        values={};hooks=[]
        for name in self.cfg['feature_nodes']:
            hooks.append(model.layers[name.replace('.','__')].register_forward_hook(lambda m,a,y,key=name:values.__setitem__(key,y)))
        y=model(x)
        for h in hooks:h.remove()
        return y,values

    def critic_images(self,a,b,m):
        # Same random color projection and crop for real/fake. Cropping alone
        # keeps original pixel spacing; no resizing can hide fine texture.
        w=a.new_tensor([self.rng.uniform(.2,.4),self.rng.uniform(.45,.65),self.rng.uniform(.05,.2)]);w/=w.sum()
        size=self.cfg['critic_patch'];yy=self.rng.randrange(m,a.shape[2]-m-size+1);xx=self.rng.randrange(m,a.shape[3]-m-size+1)
        gray=lambda x:(x[:,:,yy:yy+size,xx:xx+size]*w[None,:,None,None]).sum(1,keepdim=True)
        return gray(a),gray(b)

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);original=x;m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m]
        aug=self.rng.random()<.5
        if aug:x=(x+torch.randn_like(x)*2/128).clamp(-1,127/128)
        p,features=self.features(self.net,x);y=straight_colors(p);target=decode(target_components(t));losses=spatial(p,t,m);losses.update(clean_structure(y,target,m))
        with torch.no_grad():_,ref=self.features(self.reference,t)
        feature=[]
        for key in features:
            scale=ref[key].square().mean((0,2,3),keepdim=True).sqrt().clamp_min(.1)
            z=(features[key]-ref[key])/scale;feature.append(z[...,m//2:-m//2,m//2:-m//2].abs().mean())
        losses['feature']=sum(feature)/len(feature)
        fake,real=self.critic_images(y,target,m)
        self.critic.requires_grad_(True);self.critic.train();self.dopt.zero_grad(set_to_none=True)
        dr=self.critic(real);df=self.critic(fake.detach());dl=.5*((dr-1).square().mean()+df.square().mean());assert torch.isfinite(dl);dl.backward();self.dopt.step()
        self.critic.requires_grad_(False);self.critic.eval();losses['adversarial']=(self.critic(fake)-1).square().mean()
        with torch.no_grad():
            xd=(original[1:]-warp(original[:1],flow)).abs().mean(1,keepdim=True);valid=mask*(xd<8/128);allowed=xd.clamp(max=3/128)+1/128
        delta=(y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True)
        losses['temporal']=(cut(F.relu(delta-allowed))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];q=straight_colors(self.net(shift))
        losses['phase']=(q[...,m+sy:-m,m+sx:-m]-y[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        ramp=min(1,(self.step+1)/self.cfg['critic_warmup']);loss=sum(v*self.cfg['objective_weights'][k]*(ramp if k=='adversarial' else 1) for k,v in losses.items())
        lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},critic_loss=float(dl.detach()),grad=float(norm),lr=lr,native=native,noise=aug,sample=spec,seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'fp32_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();p=decode(self.net(x))
            target=as_tensor(read_image(ROOT/'runs'/RUN/'intermediate-target-screen'/f'{r["id"]}_0.008.png'))[None].cuda();t=decode(target_components(target))
            Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'));rows.append(dict(id=r['id'],**texture_scores(p,t)))
        write_json(folder/'scores.json',dict(records=rows,not_blind=True,style_accepted=False))
        board=Image.new('RGB',(1280,1192),'#18202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
            r=next(r for r in self.data['validation'] if r['id']==key);base=ROOT/'runs'/RUN
            paths=[(r['target'],'原教师参考'),(base/'intermediate-target-screen'/f'{key}_0.008.png','当前区域监督'),(base/'b1-capacity-control-v2/capacity/fp32_002000'/(key+'.png'),'本轮修改前'),(folder/(key+'.png'),f'清洁约束 {self.step}步')]
            for i,(path,title) in enumerate(paths):d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(i*320,j*298+30))
        board.save(folder/'comparison.png')


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};expected_d={k:v.cpu().clone() for k,v in tr.critic.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());de=max(float((expected_d[k]-v.cpu()).abs().max()) for k,v in tr.critic.state_dict().items())
    assert err<2e-6 and de<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_generator_error=err,restore_critic_error=de,budget=budget()));print('CLEAN_PREFLIGHT_PASS',err,de,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='clean-capacity-v1');p.add_argument('--steps',type=int,default=2000);p.add_argument('--stop',type=int,default=500);p.add_argument('--resume',type=Path);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100461);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':prepare(out,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists();tr.evaluate()
    write_json(out/f'launch_{tr.step:06d}.json',dict(start=tr.step,stop=a.stop,graph_changed=False,bindings=tr.binding,preflight='PASS'))
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('CLEAN_TRAIN',tr.step,round(row['loss'],4),round(row['clean']*128,4),round(row['density'],4),flush=True)
            if tr.step%500==0 or tr.step==a.stop:torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('CLEAN_DONE',tr.step,flush=True)


if __name__=='__main__':main()
