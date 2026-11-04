"""User-requested small-detail merging, learned before the colour mapping.

Bilateral preparation is used ONLY to form training teacher targets. Runtime
still contains only the existing fifteen convolutions, residuals and nearest4.
"""
import argparse,time,json,copy,shutil
import numpy as np
import cv2
import library as l
from library import D,torch,save_json,sha
from refine import load_packed
from train import prepare_inputs,cut,objective,targets

IDS=['PI02','PI03','PI04','NI01','NI02','NI03','NI04','NI05','W-T256','R8-06','R8-07','B03']

def filtered_inputs(x):
    a=((x+1)*128).permute(0,2,3,1).cpu().numpy().round().clip(0,255).astype(np.uint8)
    out=[cv2.bilateralFilter(z,7,32,2.4,borderType=cv2.BORDER_REPLICATE) for z in a]
    return l.tx(np.stack(out))

def fit(e,inputs,nat,steps):
    folder=D/'models'/e['id'];report=json.loads((folder/'training.json').read_text('utf-8'))
    if report.get('detail_merge_complete'):print('MERGE_SKIP_COMPLETE',e['id'],flush=True);return
    l.guard();start=time.perf_counter();X,V=inputs;teacher=l.Teacher(e)
    FX,FV=filtered_inputs(X),filtered_inputs(V);Y,_=targets(teacher,FX,False);W,_=targets(teacher,FV,False)
    m=load_packed(folder);m.leak=.002
    def score():
        with torch.no_grad():
            z=torch.cat([cut(m(x)) for x in V.split(4)]);err=(z-cut(W)).abs().mean((1,2,3))*128
            a=float(err[:nat[1]].mean());b=float(err[nat[1]:].mean());return a+.25*b,a,b
    for p in m.parameters():p.requires_grad_(False)
    dc={};params=[]
    for n,w in m.w.items():
        if w.shape[-1]!=3:continue
        w.requires_grad_(True);params.append(w);s=(getattr(m,n+'_m')/131072).float()[:,None,None,None]
        dc[n]=l.HalfAway.apply(w/s).clamp(-127,127).sum((-1,-2))
        def tangent(g):
            z=g-g[:,:,1:2,1:2];z=z.clone();z[:,:,1,1]=0;return z
        w.register_hook(tangent)
    def project():
        with torch.no_grad():
            for n,target in dc.items():
                w=m.w[n];s=(getattr(m,n+'_m')/131072).float()[:,None,None,None]
                wi=l.HalfAway.apply(w/s).clamp(-127,127);wi[:,:,1,1]=0;centre=target-wi.sum((-1,-2));invalid=(centre<-127)|(centre>127)
                if invalid.any():w[invalid]=0;centre=torch.where(invalid,target,centre)
                w[:,:,1,1].copy_(centre*s[:,:,0,0])
    # Uniform-colour responses must remain identical; denoising must not
    # silently become a different random palette or channel mapping.
    colors=np.asarray([[r,g,b] for r in [0,85,170,255] for g in [0,85,170,255] for b in [0,85,170,255]],np.uint8)
    patches=l.tx(np.broadcast_to(colors[:,None,None,:],(64,8,8,3)).copy())
    with torch.no_grad():before_colours=m(patches).clone()
    initial=score();best=initial[0];state=copy.deepcopy(m.state_dict());selected=0;logs=[]
    opt=torch.optim.Adam(params,lr=.00005);rng=np.random.default_rng(2026110204)
    for step in range(steps+1):
        if step%50==0 or step==steps:
            sc,a,b=score();logs.append(dict(step=step,score=sc,natural_mae=a,palette_mae=b))
            if sc<best-1e-6:best=sc;state=copy.deepcopy(m.state_dict());selected=step
            if step%100==0 or step==steps:print('MERGE_TRAIN',e['id'],step,round(sc,3),'best',selected,flush=True)
        if step==steps:break
        if time.perf_counter()-start>900:raise TimeoutError('15 minute per-style detail-merge cap')
        ids=np.r_[rng.choice(nat[0],4,False),rng.choice(np.arange(nat[0],len(X)),2,False)];idx=torch.as_tensor(ids,device='cuda')
        opt.zero_grad(set_to_none=True);out=m(X[idx]);loss=objective(cut(out),cut(Y[idx]))
        if step%20==0:
            sx=torch.roll(X[idx],1,-1);st=teacher(filtered_inputs(sx));so=m(sx)
            loss+=.1*((cut(so)-cut(out))-(cut(st)-cut(Y[idx]))).abs().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(params,3);opt.step();project()
    m.load_state_dict(state);m.leak=0;m.eval()
    with torch.inference_mode():assert torch.equal(before_colours,m(patches)),'uniform palette drift'
    backup=folder/'before-detail-merge';backup.mkdir(exist_ok=True)
    for name in ['training.json','coefficients-padded.bin','graph.json','layout.json']:
        shutil.copy2(folder/name,backup/name)
    oldsha=sha(folder/'coefficients-padded.bin');l.export(m,folder);oracle=l.independent_oracle(folder);engine=l.PackedEngine(folder)
    with torch.inference_mode():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,style_id=e['id'],teacher_sha256=e['source_sha256'],actual_gradient_steps=steps,selected_step=selected,detail_merge=True),folder/'selected.pt')
    final=score();report.update(detail_merge_complete=True,detail_merge=dict(target_revision='precolour_bilateral7_sigma32_space2.4_training_only',gradient_steps=steps,selected_step=selected,initial_score=initial[0],final_score=final[0],records=logs,source_variation_is_intentional=True,uniform_rgb64_output_identical=True,dc_kernel_sums_preserved_on_integer_lattice=True,old_coefficients_sha256=oldsha,seconds=time.perf_counter()-start),selected_score=final[0],selected_natural_mae=final[1],selected_palette_mae=final[2],selected_score_target='frozen teacher applied to training-only edge-preserving simplified content',coefficient_sha256=sha(folder/'coefficients-padded.bin'),authoritative_checkpoint='selected.pt',independent_oracle=oracle,no_inference_filter=True)
    save_json(folder/'training.json',report);print('DETAIL_MERGE_FINISHED',e['id'],round(initial[0],3),'to',round(final[0],3),flush=True)
    del teacher,FX,FV,Y,W,m,engine;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*',default=IDS);p.add_argument('--steps',type=int,default=600);a=p.parse_args();l.setup(20261102);inputs,nat=prepare_inputs()
    for e in l.read_inventory():
        if e['id'] in a.ids:fit(e,inputs,nat,a.steps)
if __name__=='__main__':main()
