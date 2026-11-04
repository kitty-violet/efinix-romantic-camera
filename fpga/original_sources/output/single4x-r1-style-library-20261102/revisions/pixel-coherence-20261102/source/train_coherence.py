"""Fit only the existing low-resolution decision DW neighbourhood."""
import copy,time
from common import *
from diagnose import candidate
import train as data_module

def central(x):return x[:,:,12:-12,12:-12]

def main():
    o.setup(20261102);guard();data_module.D=D
    (X,V),nat=data_module.prepare_inputs();teacher=o.Engine(R2,2)
    @torch.no_grad()
    def targets(a):
        ys=[];bs=[]
        for x in a.split(4):
            q=teacher(x,True);ys.append(q['rgb']/128);bs.append((q['up1_low_dw'][:,:24]>63).float())
        return torch.cat(ys),torch.cat(bs)
    Y,B=targets(X);W,C=targets(V)
    m=candidate('gate100');m.leak=.002
    for p in m.parameters():p.requires_grad_(False)
    w=m.w['up1_low_dw'];w.requires_grad_(True)
    scale=(m.up1_low_dw_m.float()/131072)[:,None,None,None]
    original=w.detach().clone();logs=[];start=time.perf_counter()
    def tangent(g):
        z=torch.zeros_like(g);a=g[:24].mean(0,keepdim=True);a=a-a[:,:,1:2,1:2]
        z[:24]=a;return z
    w.register_hook(tangent)
    @torch.no_grad()
    def project():
        qi=w/scale
        corner=qi[:24,0][:,[0,0,2,2],[0,2,0,2]].mean().clamp(1,25)
        edge=qi[:24,0][:,[0,1,1,2],[1,0,2,1]].mean().clamp(1,25)
        total=corner+edge
        if total<16:corner=corner*16/total;edge=edge*16/total
        if total>28:corner=corner*28/total;edge=edge*28/total
        a=torch.stack([corner,edge,corner,edge,127-4*corner-4*edge,edge,corner,edge,corner]).reshape(1,1,3,3)
        # Quantized DC is exactly 127 for every output, at every update.
        q=o.HalfAway.apply(a).clamp(0,127);q[:,:,1,1]=127-q.sum((-1,-2))+q[:,:,1,1]
        assert 0<=q.min() and q.max()<=127
        a=a.clone();a[:,:,1,1]+=q[:,:,1,1]-o.HalfAway.apply(a[:,:,1,1])
        w[:24].copy_(a*scale[:24]);w[24:].copy_(original[24:])
    def loss(x,y,b):
        q=m(x,True);pred=q['rgb']/128
        qi=o.HalfAway.apply(w/scale).clamp(-127,127)
        raw=F.conv2d(F.pad(q['res1_add'],(1,1,1,1),mode='replicate'),qi,bias=o.HalfAway.apply(m.b['up1_low_dw']/scale[:,0,0,0]),groups=32)
        ce=F.binary_cross_entropy_with_logits(central(raw[:,:24])/512,central(b))
        a=pred[:,:,48:-48:4,48:-48:4];t=y[:,:,48:-48:4,48:-48:4]
        colour=(a-t).abs().mean()
        edge=((a[:,:,:,1:]-a[:,:,:,:-1])-(t[:,:,:,1:]-t[:,:,:,:-1])).abs().mean()
        edge+=((a[:,:,1:]-a[:,:,:-1])-(t[:,:,1:]-t[:,:,:-1])).abs().mean()
        return ce+.5*colour+.25*edge
    @torch.no_grad()
    def score():
        mae=[];edge=[]
        for x,y in zip(V[:nat[1]].split(4),W[:nat[1]].split(4)):
            a=m(x)[:,:,48:-48:4,48:-48:4];t=y[:,:,48:-48:4,48:-48:4]
            mae.append(float((a-t).abs().mean()*128))
            e=((a[:,:,:,1:]-a[:,:,:,:-1])-(t[:,:,:,1:]-t[:,:,:,:-1])).abs().mean()
            e+=((a[:,:,1:]-a[:,:,:-1])-(t[:,:,1:]-t[:,:,:-1])).abs().mean();edge.append(float(e*64))
        a=float(np.mean(mae));e=float(np.mean(edge));return a+.35*e,a,e
    best=score();state=copy.deepcopy(m.state_dict());selected=0;initial=best
    opt=torch.optim.Adam([w],lr=.025);rng=np.random.default_rng(2026110213)
    for step in range(501):
        if step%100==0:
            guard();s=score();q=o.HalfAway.apply(w[0,0]/scale[0,0]).tolist()
            logs.append(dict(step=step,score=s[0],colour_mae=s[1],grid_edge_mae=s[2],kernel=q))
            if s[0]<best[0]:best=s;state=copy.deepcopy(m.state_dict());selected=step
            print('COHERENCE_TRAIN',step,s,'best',selected,q,flush=True)
        if step==500:break
        assert time.perf_counter()-start<900
        ids=torch.as_tensor(rng.choice(nat[0],6,False),device='cuda')
        opt.zero_grad(set_to_none=True);z=loss(X[ids],Y[ids],B[ids]);z.backward();torch.nn.utils.clip_grad_norm_([w],3);opt.step();project()
    m.load_state_dict(state);m.leak=0;m.eval();folder=D/'models/PIXEL_COHERENT';folder.mkdir(parents=True,exist_ok=True)
    o.export(m,folder);engine=o.Engine(folder)
    with torch.no_grad():
        for x in V.split(4):assert torch.equal(m(x),engine(x))
        colours=np.concatenate([np.repeat(np.arange(256,dtype=np.uint8)[:,None],3,1),rng.integers(0,256,(64,3),dtype=np.uint8)])
        old=o.Engine(R1)
        for cs in np.array_split(colours,20):
            x=o.tx(np.broadcast_to(cs[:,None,None,:],(len(cs),8,8,3)).copy());assert torch.equal(engine(x),old(x))
    oracle=o.independent_oracle(folder)
    torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,style_id='PIXEL_COHERENT',gradient_steps=500,selected_step=selected),folder/'selected.pt')
    save(folder/'training.json',dict(gradient_steps=500,selected_step=selected,initial=initial,selected_score=best,records=logs,base_R1_sha256=sha(R1/'coefficients-padded.bin'),teacher_R2_sha256=sha(R2/'coefficients-padded.bin'),coefficient_sha256=sha(folder/'coefficients-padded.bin'),only_trainable_layer='up1_low_dw, existing first24 depthwise kernels, tied symmetric nonnegative DC127',operators_and_topology_unchanged=True,flat_RGB_320_cases_exactly_preserved=True,final_palette_weights_frozen=True,high_resolution_decoder_centre_only=True,qat_vs_bin_validation_exact=True,oracle=oracle,new_user_sources_not_training=True,hardware_modified=False))
    print('COHERENCE_MODEL_FROZEN',selected,best,flush=True)

if __name__=='__main__':main()
