"""Repair style-defining routes in an isolated S4R2 revision, not a new graph."""
from pathlib import Path
import json,copy,time,sys,shutil
import numpy as np
from PIL import Image
import library as l
from library import torch,F,save_json,sha
import train

BASE=l.D
D=BASE/'revisions/film-pixel-r2'

def empty_model():
    m=l.StyleModel().cuda()
    with torch.no_grad():
        for w in m.w.values():w.zero_()
        for b in m.b.values():b.zero_()
        for name,w in m.w.items():
            if w.shape[-1]==3:w[:,0,1,1]=1
            else:
                for c in range(min(w.shape[:2])):w[c,c,0,0]=1
    m.calibrate();m.qat=True;m.leak=.002
    return m

@torch.no_grad()
def cp(m,n,new,old,cin=None,cout=None):
    w=n.weight[old];b=n.bias[old];count=len(b) if cout is None else cout
    m.w[new].zero_();m.b[new].zero_()
    if cin is not None:w=w[:,:cin]
    m.w[new][:count,:w.shape[1]].copy_(w[:count]);m.b[new][:count].copy_(b[:count])
    getattr(m,new+'_m')[:count].copy_(getattr(n,old+'_m').flatten()[:count])

@torch.no_grad()
def row(m,name,out,terms,bias=0):
    m.w[name][out].zero_();m.b[name][out]=bias
    for c,v in terms.items():m.w[name][out,c,0,0]=v

@torch.no_grad()
def pixel_build(entry):
    old=l.teacher_parameters(entry['source']);m=empty_model()
    cp(m,old,'stem','stem');cp(m,old,'down_dw','down1__dw')
    # The original colour thresholds and hard terminal decision are retained.
    # Identity low-resolution stages are removed, never averaged into a colour
    # decoder. Palette reconstruction is AFTER the last hard decision.
    cp(m,old,'down_pw','down2__pw',cin=8)
    for dst,src in [('res1_a','res1_conv1'),('res1_b','res1_conv2'),('res2_a','res2_conv1'),('res2_b','res3_conv1')]:
        cp(m,old,dst+'_dw',src+'__dw');cp(m,old,dst+'_pw',src+'__pw')
    cp(m,old,'up1_low_dw','res3_conv2__dw')
    cp(m,old,'up1_low_pw','res3_conv2__pw',cout=8)
    cp(m,old,'up1_high','up2_high_dw');cp(m,old,'rgb','rgb_head')
    # All eight high-resolution depthwise filters are exact centre identities.
    m.w['up1_high'].zero_();m.w['up1_high'][:,0,1,1]=1;m.b['up1_high'].zero_();m.up1_high_m.fill_(2048)
    return m,old

@torch.no_grad()
def film_build(entry,gain=4.,floor=6.):
    old=l.teacher_parameters(entry['source']);m=empty_model()
    for c in range(3):m.w['stem'][c].copy_(old.weight['stem'][c]);m.b['stem'][c].copy_(old.bias['stem'][c])
    for c in range(3,8):m.w['stem'][c].copy_(old.weight['stem'][3]);m.b['stem'][c].copy_(old.bias['stem'][3])
    g=torch.tensor([[1,2,1],[2,4,2],[1,2,1]],device='cuda')/16
    m.w['down_dw'][:4,0]=g
    for c,(dy,dx,sign) in enumerate([(0,1,1),(1,0,1),(0,1,-1),(1,0,-1)],4):
        k=m.w['down_dw'][c,0];k.zero_();k[1,1]=-2*gain*sign;k[1+dy,1+dx]=gain*sign;k[1-dy,1-dx]=gain*sign
    m.w['down_pw'].zero_();m.b['down_pw'].zero_()
    for c in range(3):row(m,'down_pw',c,{c:1})
    row(m,'down_pw',3,{3:1},float(old.bias['down2__pw'][3]))
    row(m,'down_pw',4,{3:1},float(old.bias['down2__pw'][4]))
    row(m,'down_pw',5,{3:1});row(m,'down_pw',6,{3:1},-80);row(m,'down_pw',30,{3:-1},45)
    for c in range(8,12):row(m,'down_pw',c,{c-4:1})
    row(m,'down_pw',16,{4:1,5:-1});row(m,'down_pw',17,{6:1,7:-1})
    row(m,'res1_a_pw',18,{8:1,16:-1});row(m,'res1_a_pw',19,{10:1,17:-1})
    slope=(12-floor)/80
    for first,c in [(20,18),(23,19)]:
        for i in range(3):
            terms={c:120,5:-120*slope,6:120*slope}
            row(m,'res1_b_pw',first+i,terms,-120*(floor-.5+i*.5))
            row(m,'res1_b_pw',first-12+i,terms,-120*(floor-.5+i*.5))
    # Average ramp channels before their finite difference to avoid INT8
    # saturation of a raw three-way sum. Dot ramps remain undistributed.
    for c in range(20,26):
        m.w['res2_a_dw'][c].zero_();m.w['res2_a_dw'][c,0,1,:]=.25
    row(m,'res2_a_pw',26,{20:4,21:-8,22:4})
    row(m,'res2_a_pw',27,{23:4,24:-8,25:4})
    row(m,'res2_a_pw',28,{8:.5,9:-1,10:.5,11:.5,12:-1,13:.5})
    for name in ['res2_b_dw','up1_low_dw']:
        for c in [26,27]:m.w[name][c].zero_();m.w[name][c,0,1,:]=1
    m.w['res2_b_dw'][28].zero_();m.w['res2_b_dw'][28,0,0,1]=1
    m.w['up1_low_pw'].zero_();m.b['up1_low_pw'].zero_()
    for c in range(5):row(m,'up1_low_pw',c,{c:1})
    row(m,'up1_low_pw',5,{26:1});row(m,'up1_low_pw',6,{27:1,30:-3});row(m,'up1_low_pw',7,{28:4},-50)
    for c in [5,6]:m.w['up1_high'][c].zero_();m.w['up1_high'][c,0,1,:]=1
    # Preserve the content path instead of blurring it at every quarter-scale
    # stage. One quarter-grid cross plus native blur approximates the original
    # two decoder filters' footprint; event spreading remains independent.
    for name in ['res1_a_dw','res1_b_dw','res2_a_dw','res2_b_dw']:
        m.w[name][:5].zero_();m.w[name][:5,0,1,1]=1
    cross=torch.tensor([[0,.125,0],[.125,.5,.125],[0,.125,0]],device='cuda')
    m.w['up1_low_dw'][:5,0]=cross;m.w['up1_high'][:5,0]=g
    m.calibrate();cp(m,old,'rgb','rgb_head')
    return m,old

def background(m,nodes,name='rgb'):
    if isinstance(m,l.StyleModel):
        w=m.w[name];mult=getattr(m,name+'_m');scale=(mult/131072).float()
        wi=l.HalfAway.apply(w/scale[:,None,None,None]).clamp(-127,127)
        bi=l.HalfAway.apply(m.b[name]/scale).double()[None,:,None,None]
        a=nodes['up1_high'];mult=mult[None,:,None,None]
    else:
        name='rgb_head';scale=(getattr(m,'rgb_head_m')/131072).float().flatten();wi=l.HalfAway.apply(m.weight[name]/scale[:,None,None,None]).clamp(-127,127)
        bi=l.HalfAway.apply(m.bias[name]/scale).double()[None,:,None,None];a=nodes['up2_high_dw'];mult=getattr(m,'rgb_head_m')
    dot=F.conv2d(a[:,:5],wi[:,:5]);return l.HalfAway.apply((dot.double()+bi).clamp(-(2**31),2**31-1)*mult/131072).clamp(-128,127).float()/128

@torch.no_grad()
def event_summary(m,x):
    nodes=m(x,True);pred=nodes['rgb']/128 if isinstance(m,l.StyleModel) else nodes['rgb_head']/128
    b=background(m,nodes);z=train.cut(pred-b)*128
    return torch.stack([(z>10).float().mean(),(z< -10).float().mean(),z.clamp_min(0).mean(),(-z).clamp_min(0).mean()]).cpu().numpy()

def freeze(entry,m,report,V):
    key=entry['id'];folder=D/'models'/key;folder.mkdir(parents=True,exist_ok=True)
    m.leak=0;m.qat=True;m.eval();l.export(m,folder);engine=l.PackedEngine(folder)
    with torch.no_grad():
        for batch in V.split(4):assert torch.equal(engine(batch),m(batch))
    oracle=l.independent_oracle(folder)
    torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,style_id=key,actual_gradient_steps=report['gradient_steps'],selected_step=report['selected_step']),folder/'selected.pt')
    report.update(style_id=key,source=entry['source'],teacher_sha256=entry['source_sha256'],coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_oracle=oracle,qat_vs_export_all_validation_exact=True,pure_integer_forward=True,no_inference_filter=True,visual_acceptance=False,hardware_test=False)
    save_json(folder/'training.json',report);print('REPAIR_FROZEN',key,report['selected_step'],flush=True)

def fit_pixel(entry,inputs,nat):
    m,old=pixel_build(entry);X,V=inputs
    with torch.no_grad():
        targets=torch.cat([old(x,True)['down2.dw'][:,:3] for x in X.split(4)])
        val=torch.cat([old(x,True)['down2.dw'][:,:3] for x in V.split(4)])
    # Only the RGB sampling footprint can train. Hard palette, terminal
    # thresholds and centre-only upsampling decoder cannot learn to blur.
    for p in m.parameters():p.requires_grad_(False)
    for name in ['stem','down_dw']:m.w[name].requires_grad_(True)
    masks={n:torch.zeros_like(m.w[n]) for n in ['stem','down_dw']}
    for c in range(3):masks['stem'][c,c]=1;masks['down_dw'][c,0]=1
    def tangent(g,mask):
        z=(g-g[:,:,1:2,1:2])*mask;z=z.clone();z[:,:,1,1]=0;return z
    for n,mask in masks.items():m.w[n].register_hook(lambda g,mask=mask:tangent(g,mask))
    sums={n:l.HalfAway.apply(m.w[n]/(getattr(m,n+'_m').float()/131072)[:,None,None,None]).sum((-1,-2)).detach() for n in masks}
    last_valid={n:m.w[n].detach().clone() for n in masks}
    def project():
        with torch.no_grad():
            for n in masks:
                w=m.w[n];sc=(getattr(m,n+'_m').float()/131072)[:,None,None,None]
                # RGB kernels share the same footprint and preserve integer DC.
                if n=='stem':
                    avg=torch.stack([w[c,c] for c in range(3)]).mean(0)
                    for c in range(3):w[c,c].copy_(avg)
                else:w[:3].copy_(w[:3].mean(0,keepdim=True).expand_as(w[:3]))
                wi=l.HalfAway.apply(w/sc).clamp(-127,127);wi[:,:,1,1]=0;centre=sums[n]-wi.sum((-1,-2))
                invalid=(centre.abs()>127).any(1)
                if invalid.any():
                    w[invalid]=last_valid[n][invalid];wi=l.HalfAway.apply(w/sc).clamp(-127,127);wi[:,:,1,1]=0;centre=sums[n]-wi.sum((-1,-2))
                assert centre.abs().max()<=127
                w[:,:,1,1].copy_(centre*sc[:,:,0,0])
                last_valid[n].copy_(w)
    def measure():
        with torch.no_grad():
            o=torch.cat([m(x,True)['down_dw'][:,:3,12:-12,12:-12] for x in V[:nat[1]].split(4)])
            return float((o-val[:nat[1],:,12:-12,12:-12]).abs().mean())
    rng=np.random.default_rng(2026110206);initial=measure();best=initial;state=copy.deepcopy(m.state_dict());selected=0;logs=[]
    opt=torch.optim.Adam([m.w['stem'],m.w['down_dw']],lr=.00025)
    for step in range(801):
        if step%100==0:
            value=measure();logs.append(dict(step=step,quarter_rgb_mae=value))
            if value<best:best=value;state=copy.deepcopy(m.state_dict());selected=step
            print('PIXEL_REPAIR',step,round(value,3),'best',selected,flush=True)
        if step==800:break
        ids=torch.as_tensor(rng.choice(nat[0],6,False),device='cuda');o=m(X[ids],True)['down_dw'][:,:3,12:-12,12:-12];t=targets[ids,:,12:-12,12:-12]
        loss=(o-t).abs().mean()+.25*((o[:,:,:,1:]-o[:,:,:,:-1])-(t[:,:,:,1:]-t[:,:,:,:-1])).abs().mean()
        loss+=.25*((o[:,:,1:]-o[:,:,:-1])-(t[:,:,1:]-t[:,:,:-1])).abs().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([m.w['stem'],m.w['down_dw']],3);opt.step();project()
    m.load_state_dict(state)
    freeze(entry,m,dict(repair='preserve original palette and final binary gates; train only shared DC-preserving RGB sampling footprint',gradient_steps=800,selected_step=selected,initial_quarter_rgb_mae=initial,selected_quarter_rgb_mae=best,records=logs,hard_decoder_centre_only=True,original_bounded_sharpening_not_exactly_reproduced=True),V)
    del m,old;torch.cuda.empty_cache()

def fit_film(entry,inputs,nat):
    X,V=inputs;teacher=l.teacher_parameters(entry['source']);cal=X[torch.as_tensor(np.arange(0,nat[0],16)[:8],device='cuda')]
    target=event_summary(teacher,cal);records=[];best=float('inf');chosen=None
    for gain in [2.,4.,8.]:
        for floor in [2.,4.,6.,8.]:
            m,unused=film_build(entry,gain,floor);got=event_summary(m,cal)
            score=float(np.mean(np.abs(np.log((got+np.array([.002,.002,.2,.2]))/(target+np.array([.002,.002,.2,.2]))))))
            records.append(dict(gain=gain,floor=floor,score=score,measured=got.tolist(),target=target.tolist()))
            if score<best:best=score;chosen=(gain,floor)
            del m,unused
    print('FILM_ROUTE_SELECTED',chosen,records,flush=True)
    m,unused=film_build(entry,*chosen)
    with torch.no_grad():
        Y=torch.cat([background(teacher,teacher(x,True)) for x in X.split(4)])
        W=torch.cat([background(teacher,teacher(x,True)) for x in V.split(4)])
    for p in m.parameters():p.requires_grad_(False)
    m.w['rgb'].requires_grad_(True);m.b['rgb'].requires_grad_(True)
    mask=torch.zeros_like(m.w['rgb']);mask[:,:5]=1;m.w['rgb'].register_hook(lambda g:g*mask)
    opt=torch.optim.Adam([m.w['rgb'],m.b['rgb']],lr=.0002);rng=np.random.default_rng(2026110207)
    def measure():
        with torch.no_grad():
            o=torch.cat([train.cut(background(m,m(x,True))) for x in V[:nat[1]].split(4)])
            return float((o-train.cut(W[:nat[1]])).abs().mean()*128)
    first=measure();best=first;state=copy.deepcopy(m.state_dict());selected=0;logs=[]
    for step in range(501):
        if step%100==0:
            value=measure();logs.append(dict(step=step,base_mae=value))
            if value<best:best=value;state=copy.deepcopy(m.state_dict());selected=step
            print('FILM_REPAIR',step,round(value,3),'best',selected,flush=True)
        if step==500:break
        ids=torch.as_tensor(rng.choice(nat[0],6,False),device='cuda');o=train.cut(background(m,m(X[ids],True)));t=train.cut(Y[ids]);loss=(o-t).abs().mean()+.2*(o-t).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([m.w['rgb'],m.b['rgb']],3);opt.step()
    m.load_state_dict(state)
    freeze(entry,m,dict(repair='dedicated finite-window curvature events with bounded horizontal spreading; background-only QAT cannot suppress event coefficients',gradient_steps=500,selected_step=selected,initial_base_mae=first,selected_base_mae=best,records=logs,event_calibration_training_patches_only=records,event_gate=dict(curvature_gain=chosen[0],dark_floor=chosen[1],bright_centre=12),approximate_bar_support_native_pixels=30,no_inference_random_generator=True,original_phase_locations_not_claimed_equal=True),V)
    del m,teacher,unused;torch.cuda.empty_cache()

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=['PIXEL_R39','OLD_FILM']);a=p.parse_args()
    l.setup(20261102);l.guard();D.mkdir(parents=True,exist_ok=True)
    shutil.copy2(BASE/'inventory.json',D/'inventory.json')
    train.D=D;inputs,nat=train.prepare_inputs();entries={e['id']:e for e in l.read_inventory()}
    if 'PIXEL_R39' in a.ids:fit_pixel(entries['PIXEL_R39'],inputs,nat)
    if 'OLD_FILM' in a.ids:fit_film(entries['OLD_FILM'],inputs,nat)
    print('TWO_STYLE_REPAIR_COMPLETE',flush=True)

if __name__=='__main__':main()
