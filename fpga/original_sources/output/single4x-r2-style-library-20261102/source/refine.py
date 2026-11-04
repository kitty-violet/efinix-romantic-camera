"""Preserve pixel cells and choose genuine exported validation improvements."""
import json,time,copy,argparse,shutil
import numpy as np
import library as l
from library import D,torch,StyleModel,save_json,sha
from train import prepare_inputs,cut,objective,pixel_project,mono_project,targets

def load_packed(folder):
    desc,ps=l.decode(folder);m=StyleModel().cuda();m.residual=desc['residual']
    with torch.no_grad():
        for n,p in ps.items():
            s=torch.tensor(p['multiplier'],device='cuda').float()/131072
            m.w[n].copy_(torch.tensor(p['weight'],device='cuda').float()*s[:,None,None,None]);m.b[n].copy_(torch.tensor(p['bias'],device='cuda').float()*s)
            getattr(m,n+'_m').copy_(torch.tensor(p['multiplier'],device='cuda').double())
    m.qat=True;return m

@torch.no_grad()
def palette_initialize(entry):
    m=l.initialize(entry);old=l.teacher_parameters(entry['source'])
    def cp(new,src):
        m.w[new].copy_(old.weight[src]);m.b[new].copy_(old.bias[src]);getattr(m,new+'_m').copy_(getattr(old,src+'_m').flatten())
    cp('stem','stem');cp('down_dw','down1__dw')
    m.w['down_pw'].zero_();m.b['down_pw'].zero_();getattr(m,'down_pw_m').fill_(65536)
    for c in range(32):m.w['down_pw'][c,c%6,0,0]=1
    m.w['res1_a_dw'].zero_();m.w['res1_a_dw'][:,0,1,1]=1;m.b['res1_a_dw'].zero_();getattr(m,'res1_a_dw_m').fill_(2048)
    m.w['res1_a_pw'].zero_();m.w['res1_a_pw'][:,:8].copy_(old.weight['down2__pw'][:,:8]);m.b['res1_a_pw'].copy_(old.bias['down2__pw']);getattr(m,'res1_a_pw_m').copy_(old.down2__pw_m.flatten())
    cp('res1_b_dw','res1_conv1__dw');cp('res1_b_pw','res1_conv1__pw')
    cp('res2_a_dw','res1_conv2__dw');cp('res2_a_pw','res1_conv2__pw')
    cp('res2_b_dw','res2_conv1__dw');cp('res2_b_pw','res2_conv1__pw')
    cp('rgb','rgb_head');m.qat=True;del old
    return m

def refine(entry,inputs,nat,steps=800,dc=False,move_palette=False):
    key=entry['id'];folder=D/'models'/key;report=json.loads((folder/'training.json').read_text('utf-8'))
    if report.get('spatial_refinement_complete') and not dc:return
    teacher=l.Teacher(entry);X,V=inputs;use_latent=move_palette and key.startswith('R8-');Y,LT=targets(teacher,X,use_latent);W,_=targets(teacher,V,False)
    def score(m):
        with torch.no_grad():
            error=(torch.cat([cut(m(x)) for x in V.split(4)])-cut(W)).abs().mean((1,2,3))*128
            a=float(error[:nat[1]].mean());b=float(error[nat[1]:].mean());return a+.25*b,a,b
    final=load_packed(folder);initial=load_packed(folder/'initial')
    first=score(final);init=score(initial);m=initial if init[0]<first[0] else final
    start_source='initial' if m is initial else 'qat_attempt1'
    if use_latent:m=palette_initialize(entry);start_source='palette_threshold_moved_after_learnable_spatial_prefilter'
    if key=='PIXEL_R39':pixel_project(m)
    if key in ['INK','BW_LINE']:mono_project(m)
    before=score(m);best=before[0];state=copy.deepcopy(m.state_dict());selected=0;m.leak=.002
    for p in m.parameters():p.requires_grad_(False)
    for n,w in m.w.items():
        if w.shape[-1]==3:w.requires_grad_(True)
    dc_sums={}
    if dc:
        for n,w in m.w.items():
            if w.shape[-1]!=3:continue
            scale=(getattr(m,n+'_m')/131072).float()[:,None,None,None]
            dc_sums[n]=l.HalfAway.apply(w/scale).clamp(-127,127).sum((-1,-2))
            def tangent(g):
                z=g-g[:,:,1:2,1:2];z=z.clone();z[:,:,1,1]=0;return z
            w.register_hook(tangent)
    def project_dc():
        with torch.no_grad():
            for n,target in dc_sums.items():
                w=m.w[n];s=(getattr(m,n+'_m')/131072).float()[:,None,None,None]
                wi=l.HalfAway.apply(w/s).clamp(-127,127);wi[:,:,1,1]=0
                centre=target-wi.sum((-1,-2))
                # If an unconstrained step has no representable centre,
                # discard its off-centre update rather than wrap INT8.
                invalid=(centre<-127)|(centre>127)
                if invalid.any():
                    w[invalid]=0;centre=torch.where(invalid,target,centre)
                w[:,:,1,1].copy_(centre*s[:,:,0,0])
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.00006);rng=np.random.default_rng(2026110203);logs=[];start=time.perf_counter()
    for step in range(steps+1):
        if step%50==0 or step==steps:
            sc,a,b=score(m);logs.append(dict(step=step,score=sc,natural_mae=a,palette_mae=b))
            if sc<best-1e-6:best=sc;selected=step;state=copy.deepcopy(m.state_dict())
            if step%100==0:print('REFINE',key,step,round(sc,3),'best',selected,flush=True)
        if step==steps:break
        ids=np.r_[rng.choice(nat[0],4,False),rng.choice(np.arange(nat[0],len(X)),2,False)];idx=torch.as_tensor(ids,device='cuda')
        opt.zero_grad(set_to_none=True)
        if use_latent:
            nodes=m(X[idx],True);out=nodes['rgb']/128
            latent=(nodes['res1_a_dw'][:,:8,12:-12,12:-12]/128-LT[idx][:,:,12:-12,12:-12]).abs().mean()
        else:out=m(X[idx]);latent=0
        loss=objective(cut(out),cut(Y[idx]))+.5*latent
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],3);opt.step()
        if dc:project_dc()
        if key=='PIXEL_R39':pixel_project(m)
    m.load_state_dict(state);m.leak=0;m.eval();oldsha=sha(folder/'coefficients-padded.bin')
    backup=folder/'attempt1';backup.mkdir(exist_ok=True)
    for name in ['training.json','qat.pt','coefficients-padded.bin','graph.json','layout.json']:
        shutil.copy2(folder/name,backup/name)
    l.export(m,folder);oracle=l.independent_oracle(folder)
    torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,style_id=key,teacher_sha256=entry['source_sha256'],actual_gradient_steps=steps,selected_step=selected,refinement=True),folder/'selected.pt')
    engine=l.PackedEngine(folder)
    with torch.inference_mode():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    after=score(m);report.update(spatial_refinement_complete=True,refinement=dict(gradient_steps=steps,selected_step=selected,start_source=start_source,dc_lattice_preservation=dc,threshold_after_spatial_prefilter=use_latent,previous_score=first[0],initial_export_score=init[0],before_projection_score=before[0],after_score=after[0],records=logs,seconds=time.perf_counter()-start,previous_coefficients_sha256=oldsha),selected_score=after[0],selected_natural_mae=after[1],selected_palette_mae=after[2],coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_oracle=oracle,authoritative_checkpoint='selected.pt',pixel4x4_enforced=key=='PIXEL_R39')
    save_json(folder/'training.json',report);print('REFINE_DONE',key,round(first[0],3),'to',round(after[0],3),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=['INK','PIXEL_R39']);p.add_argument('--steps',type=int,default=800);p.add_argument('--dc',action='store_true');p.add_argument('--move-palette',action='store_true');a=p.parse_args();l.setup(20261102);inputs,nat=prepare_inputs()
    for entry in l.read_inventory():
        if entry['id'] in a.ids:refine(entry,inputs,nat,a.steps,a.dc,a.move_palette)
if __name__=='__main__':main()
