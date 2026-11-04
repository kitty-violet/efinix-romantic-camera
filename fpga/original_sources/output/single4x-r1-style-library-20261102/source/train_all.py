"""One bounded QAT attempt for every retained style, using frozen R2 teachers."""
import copy,json,time,sys,ast
import numpy as np
import oneblock as o
from oneblock import torch,F,D,save_json,sha
import train as data_module

def cut(x):return x[:,:,48:-48,48:-48]

def background(model,nodes):
    if isinstance(model,o.Model):
        mm=model.rgb_m;s=mm.float()/131072;w=o.HalfAway.apply(model.w['rgb']/s[:,None,None,None]).clamp(-127,127);b=o.HalfAway.apply(model.b['rgb']/s).double()[None,:,None,None];mm=mm[None,:,None,None]
    else:w,b,mm=model.ops['rgb']
    dot=F.conv2d(nodes['up1_high'][:,:5],w[:,:5]);return o.HalfAway.apply((dot.double()+b).clamp(-(2**31),2**31-1)*mm/131072).clamp(-128,127).float()/128

def noise_statistics(pred,base,teacher_base):
    delta=(cut(pred)-cut(base)).mean(1,keepdim=True);light=((cut(teacher_base)+1)/2).mean(1,keepdim=True)
    regions=[(light<.33).float(),((light>=.33)&(light<.66)).float(),(light>=.66).float()];terms=[]
    for region in regions:
        denom=region.sum().clamp_min(1)
        for sign in [-1,1]:
            z=(delta*sign).clamp_min(0)
            for v in [z,z.square(),((z-.025)/.05).clamp(0,1)]:terms.append((v*region).sum()/denom)
    return torch.stack(terms)

@torch.no_grad()
def teacher_targets(teacher,x,key):
    ys=[];bs=[];lat=[]
    for batch in x.split(4):
        if key in ['OLD_FILM','PIXEL_R39']:
            nodes=teacher(batch,True);ys.append(nodes['rgb']/128)
            if key=='OLD_FILM':bs.append(background(teacher,nodes))
            else:lat.append(nodes['res2_add'][:,:24]/128)
        else:ys.append(teacher(batch))
    return torch.cat(ys),torch.cat(bs) if bs else None,torch.cat(lat) if lat else None

def fit(entry,inputs,nat,steps=700):
    key=entry['id'];folder=D/'models'/key
    if (folder/'training.json').exists():
        j=json.loads((folder/'training.json').read_text('utf-8'));assert j['teacher_sha256']==entry['source_sha256'];assert sha(folder/'coefficients-padded.bin')==j['coefficient_sha256'];print('SKIP_VERIFIED_R1',key,flush=True);return
    o.guard();start=time.perf_counter();m,teacher,method=o.initialize(entry);m.leak=.002;m.qat=True
    X,V=inputs;Y,B,LT=teacher_targets(teacher,X,key);W,C,LV=teacher_targets(teacher,V,key)
    record=[];rng=np.random.default_rng(2026110210);params=[]
    for p in m.parameters():p.requires_grad_(key not in ['OLD_FILM','PIXEL_R39'])
    project=lambda:None
    if key=='OLD_FILM':
        m.w['rgb'].requires_grad_(True);m.b['rgb'].requires_grad_(True);m.b['down_pw'].requires_grad_(True)
        bm=torch.zeros_like(m.b['down_pw']);bm[8:14]=1;bm[20:26]=1;m.b['down_pw'].register_hook(lambda g:g*bm)
        params=[dict(params=[m.w['rgb']],lr=.00015),dict(params=[m.b['rgb']],lr=.003),dict(params=[m.b['down_pw']],lr=.02)]
        def film_project():
            with torch.no_grad():
                for first,c in [(20,4),(23,5)]:
                    floor=(-m.b['down_pw'][first]/120+.5).clamp(.5,10)
                    for i in range(3):
                        for row in [first+i,first-12+i]:
                            m.b['down_pw'][row]=-120*(floor-.5+i*.5)
                            m.w['down_pw'][row,3,0,0]=-120*(12-floor)/80;m.w['down_pw'][row,6,0,0]=120*(12-floor)/80
                m.w['rgb'][:,5].clamp_(.3,1.3);m.w['rgb'][:,6].clamp_(-.8,-.12);m.w['rgb'][:,7].clamp_(-1.5,-.4)
        project=film_project
    elif key=='PIXEL_R39':
        names=['stem','down_dw','res1_a_dw','res1_b_dw'];masks={}
        for n in names:
            w=m.w[n];w.requires_grad_(True);mask=torch.ones_like(w)
            if n=='stem':
                mask.zero_()
                for c in range(3):mask[c,c]=1
            elif n=='down_dw':mask[3:]=0
            masks[n]=mask
            def tangent(g,mask=mask):
                z=(g-g[:,:,1:2,1:2])*mask;z=z.clone();z[:,:,1,1]=0;return z
            w.register_hook(tangent)
        dc={n:o.HalfAway.apply(m.w[n]/(getattr(m,n+'_m').float()/131072)[:,None,None,None]).clamp(-127,127).sum((-1,-2)).detach() for n in names}
        previous={n:m.w[n].detach().clone() for n in names}
        def pixel_project():
            with torch.no_grad():
                for n in names:
                    w=m.w[n];s=(getattr(m,n+'_m').float()/131072)[:,None,None,None]
                    if n=='stem':
                        avg=torch.stack([w[c,c] for c in range(3)]).mean(0)
                        for c in range(3):w[c,c]=avg
                    if n=='down_dw':w[:3].copy_(w[:3].mean(0,keepdim=True).expand_as(w[:3]))
                    wi=o.HalfAway.apply(w/s).clamp(-127,127);wi[:,:,1,1]=0;centre=dc[n]-wi.sum((-1,-2));bad=(centre.abs()>127).any(1)
                    if bad.any():
                        w[bad]=previous[n][bad];wi=o.HalfAway.apply(w/s).clamp(-127,127);wi[:,:,1,1]=0;centre=dc[n]-wi.sum((-1,-2))
                    assert centre.abs().max()<=127;w[:,:,1,1].copy_(centre*s[:,:,0,0]);previous[n].copy_(w)
        project=pixel_project;params=[dict(params=[m.w[n] for n in names],lr=.00008)]
    else:params=[dict(params=m.parameters(),lr=.00012)]
    opt=torch.optim.Adam(params)
    def objective(x,y,b=None,latent=None):
        if key=='OLD_FILM':
            nodes=m(x,True);pred=nodes['rgb']/128;base=background(m,nodes);a=noise_statistics(pred,base,b);t=noise_statistics(y,b,b)
            stat=((a-t).square()/(t.detach()+.002)).mean()
            return 8*(cut(base)-cut(b)).abs().mean()+2*stat+.2*(F.avg_pool2d(cut(pred),32)-F.avg_pool2d(cut(y),32)).square().mean()
        nodes=m(x,True) if key=='PIXEL_R39' else None;pred=nodes['rgb']/128 if nodes is not None else m(x);a=cut(pred);t=cut(y)
        loss=(a-t).abs().mean()+.2*(a-t).square().mean()
        loss+=.15*((a[:,:,:,1:]-a[:,:,:,:-1])-(t[:,:,:,1:]-t[:,:,:,:-1])).abs().mean()
        loss+=.15*((a[:,:,1:]-a[:,:,:-1])-(t[:,:,1:]-t[:,:,:-1])).abs().mean()
        if key=='PIXEL_R39':loss+=.2*(nodes['res1_add'][:,:24,12:-12,12:-12]/128-latent[:,:,12:-12,12:-12]).abs().mean()
        if key=='INK':
            for sign in [-1,1]:loss+=.1*((sign*a-.55).clamp(0,.4).mean()-(sign*t-.55).clamp(0,.4).mean()).square()
        return loss
    @torch.no_grad()
    def score():
        out=[];bg=[]
        for x in V.split(4):
            if key=='OLD_FILM':
                nodes=m(x,True);out.append(nodes['rgb']/128);bg.append(background(m,nodes))
            else:out.append(m(x))
        pred=torch.cat(out);err=(cut(pred)-cut(W)).abs().mean((1,2,3))*128;natural=float(err[:nat[1]].mean());palette=float(err[nat[1]:].mean());value=natural+.25*palette;extra={}
        if key=='OLD_FILM':
            back=torch.cat(bg);base_mae=float((cut(back[:nat[1]])-cut(C[:nat[1]])).abs().mean()*128)
            a=noise_statistics(pred[:nat[1]],back[:nat[1]],C[:nat[1]]);t=noise_statistics(W[:nat[1]],C[:nat[1]],C[:nat[1]])
            distribution=float(((a-t).abs()/(t+.002)).mean());value=base_mae+10*distribution;extra=dict(base_mae=base_mae,noise_distribution_relative_error=distribution,noise_statistics=a.tolist(),teacher_noise_statistics=t.tolist())
        return value,natural,palette,extra
    initial=score();best=initial[0];state=copy.deepcopy(m.state_dict());selected=0;nonzero=0;maxgrad=0.;initial_state=copy.deepcopy(m.state_dict())
    for step in range(steps+1):
        if time.perf_counter()-start>1200:raise TimeoutError('20 minute per-style training cap')
        if step%100==0 or step==steps:
            value,n,c,extra=score();record.append(dict(step=step,score=value,natural_mae=n,palette_mae=c,**extra))
            if value<best-1e-7:best=value;state=copy.deepcopy(m.state_dict());selected=step
            print('R1_TRAIN',key,step,'score',round(value,4),'natural',round(n,4),'best',selected,flush=True)
        if step==steps:break
        if step%100==0:o.guard()
        ids=np.r_[rng.choice(nat[0],4,False),rng.choice(np.arange(nat[0],len(X)),2,False)];idx=torch.as_tensor(ids,device='cuda')
        opt.zero_grad(set_to_none=True);loss=objective(X[idx],Y[idx],B[idx] if B is not None else None,LT[idx] if LT is not None else None);loss.backward()
        trainable=[p for p in m.parameters() if p.requires_grad];g=float(torch.nn.utils.clip_grad_norm_(trainable,3));maxgrad=max(g,maxgrad);nonzero+=int(g>0);opt.step();project()
        if key=='INK':
            with torch.no_grad():
                m.w['rgb'].copy_(m.w['rgb'].mean(0,keepdim=True).expand_as(m.w['rgb']));m.b['rgb'].fill_(m.b['rgb'].mean());m.rgb_m.fill_(m.rgb_m.max())
    last_state=m.state_dict();updates=any(not torch.equal(last_state[k],initial_state[k]) for k in initial_state);m.load_state_dict(state);m.leak=0;m.eval();folder.mkdir(parents=True,exist_ok=True);o.export(m,folder)
    engine=o.Engine(folder)
    with torch.no_grad():
        for batch in V.split(4):assert torch.equal(m(batch),engine(batch))
    oracle=o.independent_oracle(folder);final=score()
    torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,style_id=key,gradient_steps=steps,selected_step=selected),folder/'selected.pt')
    save_json(folder/'training.json',dict(style_id=key,name=entry.get('current_display_name',entry['name']),teacher_sha256=entry['source_sha256'],teacher_folder=entry['source_folder'],source_r2_read_only=True,initialization=method,gradient_steps=steps,nonzero_gradient_steps=nonzero,any_training_weight_update=updates,selected_step=selected,initial_score=initial[0],selected_score=final[0],selected_natural_mae=final[1],selected_palette_mae=final[2],selected_extra=final[3],records=record,max_gradient_norm=maxgrad,coefficient_sha256=sha(folder/'coefficients-padded.bin'),cost=o.r2.costs(m.rows),independent_oracle=oracle,qat_vs_export_all_validation_exact=True,no_inference_filter=True,no_inference_flow=True,visual_acceptance=False,hardware_test=False,seconds=time.perf_counter()-start))
    print('R1_TRAIN_COMPLETE',key,round(initial[0],4),'to',round(final[0],4),'selected',selected,'sec',round(time.perf_counter()-start,1),flush=True)
    del m,teacher,engine,Y,B,LT,W,C,LV,opt,state,initial_state;torch.cuda.empty_cache()

def main():
    o.setup(20261102);o.guard();o.setup_portable();entries=o.inventory();data_module.D=D;inputs,nat=data_module.prepare_inputs();start=time.perf_counter()
    for e in entries:
        fit(e,inputs,nat)
        completed=[x['id'] for x in entries if (D/'models'/x['id']/'training.json').exists()];save_json(D/'training-progress.json',dict(completed=completed,total=16,complete=len(completed)==16,elapsed_seconds=time.perf_counter()-start))
    ast.parse(__import__('pathlib').Path(__file__).read_text('utf-8'));print('ALL_16_SINGLE_BLOCK_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
