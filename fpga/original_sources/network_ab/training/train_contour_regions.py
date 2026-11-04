"""Stage-wise supervised contour/region bottleneck, then RGB rendering.

Oracle rendering is explicitly a diagnostic/training input, never an inference
result. Production forward accepts ONLY a source frame. Familiar regression
images remain disjoint from training sources, but are not a blind test set.
"""
import argparse,copy,gc,json,math,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .contour_region_net import ContourRegionNet,plan,budget,scaffold_target
from .clean_rgb_skip_net import CleanRGBNet
from .train_detail_rgb import Training as DataTraining
from .train_direct_teacher_rgb import spatial
from .audit_capacity_texture import texture_scores
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .train_luma_regions import ROOT,RUN,warp
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def guide_losses(p,t,margin):
    m=margin//2;cut=lambda a:a[...,m:-m,m:-m]
    color=(cut(p[:,:3])-cut(t[:,:3])).abs().mean()
    e=cut((p[:,3:]-t[:,3:]).abs());strong=cut(t[:,3:])>.04
    # Normalize foreground separately: sparse boundaries must not lose to
    # the many zero-valued interior pixels.
    boundary=(e*strong).sum()/strong.sum().clamp_min(1)
    interior=(e*(~strong)).sum()/(~strong).sum().clamp_min(1)
    return dict(guide_color=color,guide_boundary=boundary,guide_interior=interior)


@torch.no_grad()
def prepare(out,steps):
    source=ROOT/'runs'/RUN/'direct-teacher-rgb-v1/fp32_checkpoint_002000.pt'
    s=torch.load(source,map_location='cpu',weights_only=True)
    assert s['scales'] is None and all(file_hash(k)==v for k,v in s['binding']['code'].items())
    out.mkdir(exist_ok=False);(out/'data.json').write_bytes((source.parent/'data.json').read_bytes())
    data=json.loads((out/'data.json').read_text(encoding='utf-8'))
    old=CleanRGBNet().eval();old.load_state_dict(s['model']);net=ContourRegionNet().eval();net.initialize(old)
    # A small, training-only linear fit initializes scaffold colors without
    # importing the old decoder's arbitrary feature-channel interpretation.
    net=net.cuda();features=[];labels=[]
    for r in data['pairs'][::4]:
        x=as_tensor(read_image(r['current']))[None].cuda();t=as_tensor(read_image(r['current_target']))[None].cuda()
        v=net(x,True)['structure.dw'];target=scaffold_target(t)
        features.append(v[...,::8,::8].permute(0,2,3,1).reshape(-1,8).cpu())
        labels.append(target[...,::8,::8].permute(0,2,3,1).reshape(-1,8).cpu())
    a=torch.cat(features).double();b=torch.cat(labels).double();a=torch.cat([a,torch.ones(len(a),1)],1)
    solution=torch.linalg.solve(a.T@a+torch.eye(9,dtype=torch.float64)*.01,a.T@b)
    head=net.conv('structure.pw');head.weight.copy_(solution[:8].T.float().view(8,8,1,1));head.bias.copy_(solution[8].float())
    initial=out/'initial.pt';torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},source=str(source),source_sha256=file_hash(source)),initial)
    cfg=dict(seed=2026100470,fp32_steps=steps,fp32_lr=.001,crop=320,margin=48,native_interval=100,
        initial=str(initial),initial_sha256=file_hash(initial),graph_schema='contour-region-bottleneck-v1',
        precision='FP32 no AMP/TF32/EMA',scaffold_pretrain_steps=400,
        objective_weights=dict(center=6,boundary=3,flat=2,clean=12,density=1),
        guide_weights=dict(guide_color=8,guide_boundary=3,guide_interior=3),temporal_weight=3,phase_weight=2,
        oracle_weight_joint=.3,teacher_deployed=False,source_bypass=False,palette=False,
        method='Predict explicit teacher contour/region scaffold, then render RGB solely from scaffold; pretrain both stages then joint learning')
    write_json(out/'config.json',cfg);write_json(out/'graph.json',plan())
    binding=copy.deepcopy(s['binding'])
    for f in [Path(__file__),Path(__file__).with_name('contour_region_net.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    # Label sanity: no edges in a constant region; a true boundary survives.
    constant=torch.zeros(1,3,96,96);step=constant.clone();step[...,48:]=.5
    assert scaffold_target(constant)[:,3:].abs().max()==0
    assert scaffold_target(step)[:,3:].max()>0
    write_json(out/'preparation.json',dict(training_only_fit_samples=len(a),pairs=len(data['pairs']),budget=budget(),
        target='actual original teacher, not source edge detector',semantic_class_labels=False,
        initial_sha256=file_hash(initial),no_new_data_or_dependencies=True))
    print('CONTOUR_PREPARED',budget()['trainable_parameters'],budget()['ideal_macs'],flush=True)


class Training(DataTraining):
    def __init__(self,out):
        super().__init__(out,'fp32');assert file_hash(self.cfg['initial'])==self.cfg['initial_sha256']
        self.net=ContourRegionNet().cuda().eval();self.net.load_state_dict(torch.load(self.cfg['initial'],map_location='cpu',weights_only=True)['model'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    def update(self,native=None,joint=None):
        start=time.perf_counter();native=(self.step+1)%self.cfg['native_interval']==0 if native is None else native
        joint=self.step>=self.cfg['scaffold_pretrain_steps'] if joint is None else joint
        x,t,flow,mask,spec=self.batch(native);m=self.cfg['margin'];cut=lambda a:a[...,m:-m,m:-m]
        target=scaffold_target(t);g=self.net.encode(x);losses=guide_losses(g,target,m)
        guide_loss=sum(v*self.cfg['guide_weights'][k] for k,v in losses.items())
        oracle=self.net.render(target);o=spatial(oracle,t,m)
        oracle_loss=sum(v*self.cfg['objective_weights'][k] for k,v in o.items())
        # During warmup the encoder learns its explicit labels independently
        # while the decoder learns what a correct scaffold should produce.
        losses['oracle']=oracle_loss
        if joint:
            p=self.net.render(g);rgb_losses=spatial(p,t,m);losses.update(rgb_losses)
            with torch.no_grad():
                xd=(x[1:]-warp(x[:1],flow)).abs().mean(1,keepdim=True);valid=mask*(xd<8/128);allowed=xd.clamp(max=3/128)+1/128
            delta=(p[1:]-warp(p[:1],flow)).abs().mean(1,keepdim=True)
            temporal=(cut(F.relu(delta-allowed))*cut(valid)).sum()/cut(valid).sum().clamp_min(1)
            sx,sy=self.rng.choice([(1,0),(0,1),(1,1)]);z=F.pad(x[:1],(sx,0,sy,0),mode='replicate')[...,:x.shape[2],:x.shape[3]];q=self.net(z)
            phase=(q[...,m+sy:-m,m+sx:-m]-p[:1,...,m:-m-sy if sy else -m,m:-m-sx if sx else -m]).abs().mean()
            losses.update(temporal=temporal,phase=phase)
            loss=guide_loss+self.cfg['oracle_weight_joint']*oracle_loss+sum(v*self.cfg['objective_weights'][k] for k,v in rgb_losses.items())+temporal*self.cfg['temporal_weight']+phase*self.cfg['phase_weight']
        else:loss=guide_loss+oracle_loss
        lr=self.cfg['fp32_lr']*(.15+.85*.5*(1+math.cos(math.pi*self.step/self.cfg['fp32_steps'])))
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.zero_grad(set_to_none=True);assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.,error_if_nonfinite=True);self.opt.step();self.step+=1;torch.cuda.synchronize()
        return dict(step=self.step,stage='joint' if joint else 'separate_scaffold_renderer',loss=float(loss.detach()),
            **{k:float(v.detach()) for k,v in losses.items()},native=native,sample=spec,lr=lr,grad=float(norm),seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'fp32_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();t=as_tensor(read_image(r['target']))[None].cuda()
            guide=self.net.encode(x);p=self.net.render(guide);truth=scaffold_target(t);oracle=self.net.render(truth)
            Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'))
            Image.fromarray(to_rgb(oracle[0])).save(folder/(r['id']+'_ORACLE_NOT_INFERENCE.png'))
            for key,v in [('predicted',guide),('teacher',truth)]:
                edge=(v[0,7].clamp(0,1)*255).byte().cpu().numpy();Image.fromarray(edge).resize((1280,720),Image.Resampling.NEAREST).save(folder/(r['id']+'_'+key+'_contour.png'))
            rows.append(dict(id=r['id'],**texture_scores(p,t),guide={k:float(v) for k,v in guide_losses(guide,truth,48).items()},oracle_rgb_mae=float((oracle-t).abs().mean()*128)))
        write_json(folder/'scores.json',dict(records=rows,style_accepted=False,not_blind=True,oracle_is_diagnostic_only=True))
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1600,1192),'#18202a');d=ImageDraw.Draw(board)
        for j,(key,(xx,yy)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
            r=next(r for r in self.data['validation'] if r['id']==key)
            for i,(path,title) in enumerate([(r['source'],'原片'),(r['target'],'原教师'),(ROOT/'runs'/RUN/'direct-teacher-rgb-v1/fp32_002000'/(key+'.png'),'上版：直接RGB'),(folder/(key+'.png'),f'轮廓→区域 {self.step}步'),(folder/(key+'_ORACLE_NOT_INFERENCE.png'),'诊断：输入正确结构（非推理）')]):
                d.text((i*320+4,j*298+5),title,font=font,fill='white');board.paste(Image.open(path).crop((xx,yy,xx+320,yy+264)),(i*320,j*298+30))
        board.save(folder/'comparison.png')
        board=Image.new('RGB',(1280,1524),'#18202a');d=ImageDraw.Draw(board)
        for j,key in enumerate(['val_stairs_090','val_plaza_090','val_new_hallway_120','val_stairs_120']):
            # Manifest identifiers vary; only include existing familiar views.
            if not (folder/(key+'_teacher_contour.png')).exists():continue
            for i,name in enumerate(['teacher','predicted']):
                d.text((i*640+5,j*381+4),key+(' 教师轮廓' if i==0 else ' 网络预测轮廓'),font=font,fill='white')
                board.paste(Image.open(folder/(key+'_'+name+'_contour.png')).resize((640,360)),(i*640,j*381+21))
        board.save(folder/'contours.png')


def preflight(out):
    tr=Training(out);calls=[]
    hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in tr.net.modules() if isinstance(m,torch.nn.Conv2d)]
    with torch.no_grad():
        x=torch.rand(1,3,720,1280,device='cuda')*1.8-.9;y=tr.net(x)
    for h in hooks:h.remove()
    assert sum(calls)==plan()['macs']
    with torch.no_grad():
        sequential=tr.net.render(tr.net.encode(x));split_error=float((y-sequential).abs().max());assert split_error==0
        reference=tr.net(x,True)['structure.pw'];assert torch.equal(reference,tr.net.encode(x))
    del x,y,sequential,reference;gc.collect();torch.cuda.empty_cache()
    full=tr.update(True,True);gc.collect();torch.cuda.empty_cache();small=tr.update(False)
    torch.save(tr.state(),out/'preflight.pt');a=tr.update(False);expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()}
    del tr;gc.collect();torch.cuda.empty_cache();tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    error=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert error<2e-6 and a['sample']==b['sample']
    write_json(out/'preflight.json',dict(status='PASS',actual720_joint_backward=True,batch2=True,restore_error=error,split_forward_error=split_error,actual_macs=sum(calls),budget=budget(),checks=[full,small]))
    print('CONTOUR_PREFLIGHT_PASS',error,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','train']);ap.add_argument('--name',default='contour-region-v1');ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--stop',type=int,default=500);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100470);cv2.setNumThreads(2);out=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':prepare(out,a.steps);return
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists();tr.evaluate()
    write_json(out/f'launch_{tr.step:06d}.json',dict(start=tr.step,stop=a.stop,binding=tr.binding,budget=budget(),preflight='PASS'))
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            r=tr.update();f.write(json.dumps(r)+'\n')
            if r['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('CONTOUR_TRAIN',tr.step,r['stage'],round(r['loss'],4),round(r['guide_boundary'],4),flush=True)
            if tr.step%500==0 or tr.step==a.stop:
                torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
    print('CONTOUR_DONE',tr.step,flush=True)


if __name__=='__main__':main()
