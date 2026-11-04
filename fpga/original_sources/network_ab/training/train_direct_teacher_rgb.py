"""Direct teacher RGB + structure/texture training, without fixed color steps."""
import argparse,copy,gc,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .train_detail_rgb import Training as DataTraining
from .train_clean_capacity import TextureCritic
from .clean_skip_capacity_net import CleanSkipNet
from .clean_rgb_skip_net import CleanRGBNet,plan,budget
from .audit_capacity_texture import edge_map,texture_scores
from .train_teacher_clean_skip import Training as Previous
from .train_luma_regions import ROOT,RUN,reconstruct,warp
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def spatial(p,t,m):
    cut=lambda x:x[...,m:-m,m:-m];edge=[];flat=[]
    for d in (1,3,7):
        for axis in (-1,-2):
            a=[slice(None)]*4;b=a.copy();a[axis]=slice(None,-d);b[axis]=slice(d,None);a=tuple(a);b=tuple(b)
            dp,dt=p[a]-p[b],t[a]-t[b];strong=cut(dt.abs().mean(1,keepdim=True)>4/128);weak=~strong;error=cut((dp-dt).abs().mean(1,keepdim=True))
            edge.append((error*strong).sum()/strong.sum().clamp_min(1));flat.append((error*weak).sum()/weak.sum().clamp_min(1))
    gp,gt=edge_map(p),edge_map(t);interior=F.max_pool2d((gt>4/128).float(),5,1,2)==0
    clean=(cut(F.relu(gp-gt-.5/128))*cut(interior)).sum()/cut(interior).sum().clamp_min(1)
    rho=lambda x:1-torch.exp(-(x/(4/128)).square())
    density=(F.avg_pool2d(cut(rho(gp)),16)-F.avg_pool2d(cut(rho(gt)),16)).abs().mean()
    return dict(center=F.l1_loss(cut(p),cut(t)),boundary=sum(edge)/len(edge),flat=sum(flat)/len(flat),clean=clean,density=density)


def prepare(out,steps):
    source=ROOT/'runs'/RUN/'teacher-clean-skip-v1/fp32_checkpoint_000500.pt';s=torch.load(source,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in s['binding']['code'].items())
    out.mkdir(exist_ok=False);component=CleanSkipNet().eval();component.load_state_dict(s['model']);net=CleanRGBNet().eval();net.from_components(component)
    initial=out/'initial.pt';torch.save(dict(model=net.state_dict(),source=str(source),source_sha256=file_hash(source),linear_head_conversion=True),initial)
    (out/'data.json').write_bytes((source.parent/'data.json').read_bytes());write_json(out/'graph.json',plan());cfg=copy.deepcopy(s['config']);cfg.update(seed=2026100464,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=steps,fp32_lr=.0008,graph_schema='direct-teacher-rgb-skip-v1',
        method='Direct original-teacher RGB with balanced signed boundaries, patch edge density, train-only texture critic and source-bounded temporal loss',
        objective_weights=dict(center=6,boundary=2,flat=3,clean=12,density=1,feature=.3,adversarial=.5,temporal=4,phase=2),
        palette=False,inference_changed=True,output='RGB3, signed INT8+128 after quantization',critic_patch_mining_candidates=8)
    write_json(out/'config.json',cfg);binding=copy.deepcopy(s['binding'])
    for f in [Path(__file__),Path(__file__).with_name('clean_rgb_skip_net.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    # Check arbitrary real targets, not only perfectly constant synthetic ones.
    data=json.loads((out/'data.json').read_text(encoding='utf-8'));t=as_tensor(read_image(data['pairs'][0]['current_target'])[160:480,400:720])[None];v=spatial(t,t,24);assert all(float(a)==0 for a in v.values())
    x=as_tensor(read_image(data['pairs'][0]['current'])[160:480,400:720])[None]
    with torch.no_grad():error=float((net(x)-reconstruct(*component(x).unbind(1))).abs().max())
    write_json(out/'objective_checks.json',dict(perfect_real_teacher_spatial_loss={k:float(a) for k,a in v.items()},head_conversion_max_error_on_training_crop=error,conversion_can_differ_if_old_components_saturate=True,budget=budget()))
    print('DIRECT_RGB_PREPARED',error,flush=True)


class Training(Previous):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        self.net=CleanRGBNet().cuda().eval();self.net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model']);self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])
        self.reference=CleanRGBNet().cuda().eval();self.reference.load_state_dict(self.net.state_dict());self.reference.requires_grad_(False)
        self.critic=TextureCritic().cuda().train();self.dopt=torch.optim.Adam(self.critic.parameters(),lr=self.cfg['critic_lr'],betas=(.5,.99))

    def critic_images(self,a,b,m):
        w=a.new_tensor([self.rng.uniform(.2,.4),self.rng.uniform(.45,.65),self.rng.uniform(.05,.2)]);w/=w.sum();size=self.cfg['critic_patch'];positions=[(self.rng.randrange(m,a.shape[2]-m-size+1),self.rng.randrange(m,a.shape[3]-m-size+1)) for _ in range(8)]
        with torch.no_grad():
            error=(edge_map(a)-edge_map(b)).abs();scores=[float(error[...,y:y+size,x:x+size].mean()) for y,x in positions]
        yy,xx=positions[max(range(len(scores)),key=scores.__getitem__)];self.critic_xy=[xx,yy]
        gray=lambda x:(x[:,:,yy:yy+size,xx:xx+size]*w[None,:,None,None]).sum(1,keepdim=True)
        return gray(a),gray(b)

    def update(self,native=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        x,t,flow,mask,spec=self.batch(native);original=x;m=self.cfg['margin'];cut=lambda z:z[...,m:-m,m:-m];aug=self.rng.random()<.5
        if aug:x=(x+torch.randn_like(x)*2/128).clamp(-1,127/128)
        y,features=self.features(self.net,x);losses=spatial(y,t,m)
        with torch.no_grad():_,ref=self.features(self.reference,t)
        losses['feature']=sum(((features[k]-ref[k])/ref[k].square().mean((0,2,3),keepdim=True).sqrt().clamp_min(.1))[...,m//2:-m//2,m//2:-m//2].abs().mean() for k in features)/len(features)
        fake,real=self.critic_images(y,t,m);self.critic.requires_grad_(True);self.critic.train();self.dopt.zero_grad(set_to_none=True);dr=self.critic(real);df=self.critic(fake.detach());dl=.5*((dr-1).square().mean()+df.square().mean());assert torch.isfinite(dl);dl.backward();self.dopt.step()
        self.critic.requires_grad_(False);self.critic.eval();losses['adversarial']=(self.critic(fake)-1).square().mean()
        with torch.no_grad():
            xd=(original[1:]-warp(original[:1],flow)).abs().mean(1,keepdim=True);valid=mask*(xd<8/128);allowed=xd.clamp(max=3/128)+1/128
        delta=(y[1:]-warp(y[:1],flow)).abs().mean(1,keepdim=True);losses['temporal']=(cut(F.relu(delta-allowed))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
        sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);shift=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[:,:,:x.shape[2],:x.shape[3]];q=self.net(shift);losses['phase']=(q[...,m+sy:-m,m+sx:-m]-y[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
        ramp=min(1,(self.step+1)/self.cfg['critic_warmup']);loss=sum(v*self.cfg['objective_weights'][k]*(ramp if k=='adversarial' else 1) for k,v in losses.items());lr=self.cfg['fp32_lr']*(.1+.9*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for g in self.opt.param_groups:g['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()},critic_loss=float(dl.detach()),grad=float(norm),lr=lr,native=native,noise=aug,sample=spec,critic_xy=self.critic_xy,seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'fp32_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();p=self.net(x);t=as_tensor(read_image(r['target']))[None].cuda();Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'));rows.append(dict(id=r['id'],**texture_scores(p,t)))
        write_json(folder/'scores.json',dict(records=rows,not_blind=True,continuous_original_teacher=True,palette=False,style_accepted=False));board=Image.new('RGB',(1280,1192),'#18202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
            r=next(r for r in self.data['validation'] if r['id']==key);base=ROOT/'runs'/RUN
            for i,(path,title) in enumerate([(r['source'],'原片'),(r['target'],'原教师目标'),(base/'b1-capacity-control-v2/capacity/fp32_002000'/(key+'.png'),'修改前：固定色阶'),(folder/(key+'.png'),f'直接学习RGB {self.step}步')]):
                d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(i*320,j*298+30))
        board.save(folder/'comparison.png')


def main():
    from . import train_clean_skip as runner
    import sys
    runner.Training=Training;runner.prepare=prepare;runner.plan=plan;runner.budget=budget
    # Its extra MAC forward check uses the same node shapes and convolution
    # count; generator/backward/restore tests use Training's actual RGB model.
    if '--name' not in sys.argv:sys.argv.extend(['--name','direct-teacher-rgb-v1'])
    runner.main()


if __name__=='__main__':main()
