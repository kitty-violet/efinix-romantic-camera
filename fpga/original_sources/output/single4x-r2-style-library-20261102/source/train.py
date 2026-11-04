"""True FP32/QAT distillation into the isolated 15-convolution model."""
from pathlib import Path
import argparse,time,copy,json,math,ast
import numpy as np
from PIL import Image
import library as l
from library import D,ROOT,OLD,RECON,torch,F,setup,save_json,sha,tx,StyleModel,Teacher,guard

def prepare_inputs():
    split=json.loads((ROOT/'output/b1-mc-pixel-20261027/data_split.json').read_text('utf-8'))
    preview=json.loads((OLD/'acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))
    assert not {r['video'] for r in split['training']}&{r['video'] for r in split['validation']}
    assert not {r['video'] for k in ['training','validation'] for r in split[k]}&{Path(r['source']).name for r in preview['scenes']}
    rng=np.random.default_rng(2026110202);parts=[];natural_counts=[];records=[]
    for key in ['training','validation']:
        arr=[]
        for row in split[key]:
            assert sha(row['path'])==row['sha256'];a=np.array(Image.open(row['path']).convert('RGB'));h,w=a.shape[:2]
            for _ in range(4 if key=='training' else 8):
                y=int(rng.integers(0,(h-256)//4+1))*4;x=int(rng.integers(0,(w-256)//4+1))*4
                arr.append(a[y:y+256,x:x+256].copy());records.append(dict(split=key,path=row['path'],sha256=row['sha256'],y=y,x=x,size=256))
        natural_counts.append(len(arr))
        # Teacher responses to flat colours and ramps constrain palette/curves;
        # these are supervised training data, never inference generators.
        for i in range(48 if key=='training' else 24):
            colour=rng.integers(0,256,3,dtype=np.uint8)
            if i%3==0:colour[:]=int(rng.integers(0,256))
            a=np.broadcast_to(colour,(256,256,3)).copy()
            if i%4==3:
                other=rng.integers(0,256,3);t=np.linspace(0,1,256)[None,:,None]
                a=np.broadcast_to(np.floor(colour[None,None,:]*(1-t)+other[None,None,:]*t+.5),(256,256,3)).astype(np.uint8).copy()
            arr.append(a)
        parts.append(tx(np.stack(arr)))
    save_json(D/'training-data.json',dict(crops=records,natural_counts=natural_counts,total_counts=[len(x) for x in parts],patch=256,loss_region=[48,48,160,160],teacher_context_halo=48,seed=2026110202,source_disjoint=True,acceptance_videos_not_training_or_validation=True,synthetic_palette_training_only=True))
    return parts,natural_counts

def cut(x):return x[:,:,48:-48,48:-48]
def objective(o,y):
    d=o-y
    z=d.abs().mean()+.2*d.square().mean()
    z+=.15*((o[:,:,:,1:]-o[:,:,:,:-1])-(y[:,:,:,1:]-y[:,:,:,:-1])).abs().mean()
    z+=.15*((o[:,:,1:]-o[:,:,:-1])-(y[:,:,1:]-y[:,:,:-1])).abs().mean()
    return z

@torch.no_grad()
def targets(teacher,x,latent):
    ys=[];ls=[]
    for batch in x.split(4):
        if latent:
            nodes=teacher(batch,True);ys.append(nodes['rgb_head']/128);ls.append(nodes['down2.dw'][:,:8]/128)
        else:ys.append(teacher(batch))
    return torch.cat(ys),torch.cat(ls) if latent else None

def mono_project(m):
    with torch.no_grad():
        m.w['rgb'].copy_(m.w['rgb'].mean(0,keepdim=True).expand_as(m.w['rgb']))
        m.b['rgb'].fill_(m.b['rgb'].mean())
        getattr(m,'rgb_m').fill_(getattr(m,'rgb_m').max())

def pixel_project(m):
    with torch.no_grad():
        w=m.w['up1_high'];centre=w[:,:,1,1].clone();w.zero_();w[:,:,1,1].copy_(centre)

def fit(entry,inputs,natural_counts,revision=1):
    key=entry['id'];folder=D/'models'/key
    if (folder/'training.json').exists():
        report=json.loads((folder/'training.json').read_text('utf-8'))
        assert sha(entry['source'])==entry['source_sha256'] and report['teacher_sha256']==entry['source_sha256']
        print('SKIP_VERIFIED_TRAINED',key,flush=True);return report
    guard();folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();teacher=Teacher(entry)
    use_latent=entry['family']!='legacy' and key!='BW_LINE'
    X,V=inputs;Y,LT=targets(teacher,X,use_latent);W,LV=targets(teacher,V,use_latent)
    m=l.initialize(entry);m.leak=.002
    mono=key in ['INK','BW_LINE']
    if mono:mono_project(m)
    if key=='PIXEL_R39':pixel_project(m)
    rng=np.random.default_rng(20261102);logs=[]
    def score():
        with torch.no_grad():
            out=torch.cat([cut(m(x)) for x in V.split(4)])
            err=(out-cut(W)).abs().mean((1,2,3))*128
            a=float(err[:natural_counts[1]].mean());b=float(err[natural_counts[1]:].mean())
            return a+.25*b,a,b
    m.qat=True;initial=score();initial_raw=l.export(m,folder/'initial');m.qat=False
    heavy=entry['family'] in ['legacy','legacy_fp32']
    stages=[('fp32',1400 if key=='BW_LINE' else 1000 if heavy else 500,.0004 if key=='BW_LINE' else .00015),('qat',900 if key=='BW_LINE' else 700 if heavy else 500,.00008)]
    max_grad=0.;stage_reports=[]
    for stage,steps,lr in stages:
        m.qat=stage=='qat'
        if m.qat:m.calibrate()
        opt=torch.optim.Adam(m.parameters(),lr=lr)
        best=score()[0];state=copy.deepcopy(m.state_dict());selected=0
        for step in range(steps+1):
            if time.perf_counter()-start>1200:raise TimeoutError('20 minute cap per style')
            if step%50==0 or step==steps:
                sc,nat,col=score();logs.append(dict(stage=stage,step=step,score=sc,natural_mae=nat,palette_mae=col,seconds=time.perf_counter()-start))
                if sc<best-1e-6:best=sc;state=copy.deepcopy(m.state_dict());selected=step
                if step%100==0 or step==steps:print('TRAIN',key,stage,step,'score',round(sc,3),'natural',round(nat,3),'palette',round(col,3),'best',selected,'sec',round(time.perf_counter()-start,1),flush=True)
            if step==steps:break
            if step%100==0:guard()
            ids=np.r_[rng.choice(natural_counts[0],4,replace=False),rng.choice(np.arange(natural_counts[0],len(X)),2,replace=False)]
            idx=torch.as_tensor(ids,device='cuda');x=X[idx];y=Y[idx]
            opt.zero_grad(set_to_none=True)
            if use_latent:
                nodes=m(x,True);out=nodes['rgb']/128
                latent=(nodes['down_dw'][:,:,12:-12,12:-12]/128-LT[idx][:,:,12:-12,12:-12]).abs().mean()
            else:out=m(x);latent=0
            loss=objective(cut(out),cut(y))+.15*latent
            if step%16==0:
                # Match the frozen teacher's change under one-pixel motion.
                # This penalizes excess flicker without suppressing real change.
                shifted=torch.roll(x,1,-1);shift_y=teacher(shifted);shift_o=m(shifted)
                loss+=.15*((cut(shift_o)-cut(out))-(cut(shift_y)-cut(y))).abs().mean()
            loss.backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),3);max_grad=max(max_grad,float(g));opt.step()
            if mono:mono_project(m)
            if key=='PIXEL_R39':pixel_project(m)
        m.load_state_dict(state);m.eval()
        torch.save(dict(model=m.state_dict(),residual=m.residual,qat=m.qat,style_id=key,teacher_sha256=entry['source_sha256'],actual_gradient_steps=steps,selected_step=selected),folder/(stage+'.pt'))
        stage_reports.append(dict(stage=stage,gradient_steps=steps,selected_step=selected,best_score=best))
    m.qat=True;m.leak=0;m.eval();raw=l.export(m,folder);engine=l.PackedEngine(folder)
    with torch.inference_mode():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    oracle=l.independent_oracle(folder);final=score()
    assert sha(entry['source'])==entry['source_sha256']
    result=dict(style_id=key,name=entry['name'],teacher_sha256=entry['source_sha256'],source=str(entry['source']),initial_score=initial[0],initial_natural_mae=initial[1],initial_palette_mae=initial[2],selected_score=final[0],selected_natural_mae=final[1],selected_palette_mae=final[2],validation_improved=final[0]<initial[0],actual_gradient_training=True,stages=stage_reports,records=logs,max_gradient_norm=max_grad,weights_changed=raw!=initial_raw,coefficient_sha256=sha(folder/'coefficients-padded.bin'),cost=l.costs(m.rows),residual=m.residual,teacher_residual_semantics_preserved=True,training_only_clamp_gradient_leak=.002,latent_supervision=use_latent,pure_integer_forward=True,qat_vs_export_all_validation_exact=True,independent_oracle=oracle,no_inference_filter=True,no_inference_flow=True,seconds=time.perf_counter()-start,visual_acceptance=False,hardware_test=False)
    save_json(folder/'training.json',result)
    print('STYLE_TRAINING_FINISHED',key,'initial',round(initial[0],3),'final',round(final[0],3),'seconds',round(time.perf_counter()-start,1),flush=True)
    del m,teacher,Y,LT,W,LV,opt,engine;torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids',nargs='*');a=parser.parse_args();setup(20261102);entries=l.inventory();inputs,nat=prepare_inputs();reports=[]
    for entry in entries:
        if a.ids and entry['id'] not in a.ids:continue
        reports.append(fit(entry,inputs,nat))
        completed=[e['id'] for e in entries if (D/'models'/e['id']/'training.json').exists()]
        save_json(D/'training-progress.json',dict(complete=len(completed)==len(entries),completed=completed,expected=len(entries),last=entry['id']))
    ast.parse(Path(__file__).read_text('utf-8'));print('TRAIN_BATCH_COMPLETE',len(reports),flush=True)
if __name__=='__main__':main()
