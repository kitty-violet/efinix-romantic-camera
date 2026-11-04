"""Learn a conservative local geometry-support field in existing B1 channels.

Synthetic bounded objects are positive; soft light spots and unbounded shadows
are negatives. The auxiliary loss reads an EXISTING channel; no training head,
new inference operator, temporal state or external segmentation is exported.
Ambiguous painted objects remain an explicit limitation.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r15_structure_guard_probe import seed as guard_seed,fixtures
from r4_box_projector import BODY
import argparse

def samples(count,seed,size=128):
    rng=np.random.default_rng(seed);yy,xx=np.mgrid[:size,:size];images=[];labels=[];kinds=[]
    for i in range(count):
        kind=i%6;bg=rng.uniform(30,225,3)
        if i%3:bg[:]=bg.mean()
        cx=rng.uniform(40,88);cy=rng.uniform(40,88);rx=rng.uniform(14,25);ry=rng.uniform(14,25)
        fg=np.clip(bg+rng.choice([-1,1])*rng.uniform(65,150),5,250)
        if kind in [0,1]:
            mask=((xx-cx)/rx)**2+((yy-cy)/ry)**2<1 if kind==0 else (np.abs(xx-cx)<rx)&(np.abs(yy-cy)<ry)
            a=np.where(mask[...,None],fg,bg);a=cv2.GaussianBlur(a.astype(np.float32),(3,3),.65)
            support=(np.abs(xx-cx)<rx+16)&(np.abs(yy-cy)<ry+16)
        elif kind in [2,3]:
            g=np.exp(-.5*(((xx-cx)/rx)**2+((yy-cy)/ry)**2))
            amp=rng.uniform(55,170)*(1 if kind==2 else -1);a=bg+amp*g[...,None];support=np.zeros((size,size),bool)
        elif kind==4:
            angle=rng.uniform(-np.pi,np.pi);position=(xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
            mask=(position>0).astype(np.float32)
            if i%2:mask=cv2.GaussianBlur(mask,(9,9),2)
            a=bg+(fg-bg)*mask[...,None];support=np.zeros((size,size),bool)
        else:
            slope=rng.uniform(-.5,.5)*(xx-size/2)+rng.uniform(-.5,.5)*(yy-size/2)
            a=bg+slope[...,None];support=np.zeros((size,size),bool)
        # Camera variation belongs in training; no per-frame noise is injected
        # into the eventual inference. Fixed seeds and split fixtures are saved.
        a=np.clip(a+rng.normal(0,.6,(size,size,3))+.5,0,255).astype(np.uint8)
        images.append(a);labels.append(support[::4,::4].astype(np.float32));kinds.append(kind)
    return images,np.stack(labels),np.array(kinds)

@torch.no_grad()
def start():
    n=guard_seed();n.training_saturation_leak=.02
    lap=torch.full((3,3),-1/8,device='cuda');lap[1,1]=1
    for ch,v in enumerate([.25,.5,.25]):n.weight['stem'][7,ch]=lap*v
    n.bias['stem'][7]=-3;n.weight['down1__dw'][7].fill_(1);n.bias['down1__dw'][7]=0
    n.weight['down1__pw'][3,3:8,0,0]=.5
    n.weight['down2__dw'][3].fill_(1/9);n.bias['down2__pw'][31]=-3
    for base in BODY:
        n.weight[base+'__dw'][31].fill_(1/9);n.bias[base+'__dw'][31]=0
        n.weight[base+'__pw'][31,31,0,0]=1.5;n.bias[base+'__pw'][31]=0
    n.weight['up1_low_dw'][31].fill_(1/9)
    return n

def support(n,x):
    return n(x,True)['up1_low_pw'][:,3:4]/127

def main(args):
    setup(271111);d=OUT/args.dest;d.mkdir(exist_ok=True);disk_guard();n=start()
    n.quantized=args.warmup==0
    a,y,k=samples(384,271111);va,vy,vk=samples(96,271112)
    X=torch.stack([tensor(q,'cpu') for q in a]).cuda();Y=torch.from_numpy(y[:,None]).cuda()
    VX=torch.stack([tensor(q,'cpu') for q in va]).cuda();VY=torch.from_numpy(vy[:,None]).cuda()
    for p in n.parameters():p.requires_grad_(False)
    train=[];bounds={}
    def activate(key,rows,lr=.001):
        for kind,container in [('w',n.weight),('b',n.bias)]:
            p=container[key];mask=torch.zeros_like(p);mask[rows]=1
            if kind=='w' and key in getattr(n,'support_input_masks',{}):mask*=n.support_input_masks[key]
            p.requires_grad_(True)
            p.register_hook(lambda g,m=mask:g*m);train.append(dict(params=[p],lr=lr if kind=='w' else lr*10))
            bounds[(key,kind)]=(p.detach().clone(),mask)
    activate('stem',slice(3,8),.0015)
    activate('down1__dw',slice(3,8),.001)
    sr=getattr(n,'support_rows',31);dr=getattr(n,'support_down_rows',3)
    activate('down1__pw',dr,.00015)
    activate('down2__dw',dr,.00015)
    activate('down2__pw',sr,.0002)
    for base in BODY:activate(base+'__dw',sr,.0002);activate(base+'__pw',sr,.0002)
    activate('up1_low_dw',sr,.0002);activate('up1_low_pw',3,.0002)
    for group in train:group['lr']*=getattr(args,'lr_scale',1.)
    opt=torch.optim.Adam(train);rng=np.random.default_rng(271113);best=float('inf');beststep=0;logs=[];begin=time.perf_counter()
    checkpoint(n,d/'initial.pt',0,trained=False,kind='lighting-negative geometry support')
    # Crop the feature field to respect RF77. Fixture centres and their bounds
    # are within this validated region; no padding-location shortcut is scored.
    crop=(slice(None),slice(None),slice(10,-10),slice(10,-10))
    for step in range(args.steps+1):
        if time.perf_counter()-begin>720:raise TimeoutError('12 minute bounded support fit')
        if step==args.warmup and args.warmup:
            n.quantized=True
            for group in opt.param_groups:group['lr']*=.4
        if step%50==0 or step==args.steps:
            with torch.no_grad():
                preds=torch.cat([support(n,q) for q in VX.split(8)])
                pred=preds[crop];target=VY[crop];pos=target>.5;neg=~pos
                miss=float((1-pred)[pos].mean());false=float(pred[neg].mean());score=3*miss+2*false
                hit=float((pred[pos]>.5).float().mean());fpr=float((pred[neg]>.5).float().mean())
                # Select the directly tested geometry classification, not a
                # hidden amplitude loss that can favour a worse false alarm.
                amplitude_score=score;score=2*fpr+(1-hit)+.02*amplitude_score
            logs.append(dict(step=step,miss_amplitude=miss,false_amplitude=false,true_positive_rate=hit,false_positive_rate=fpr,score=score,seconds=time.perf_counter()-begin))
            if n.quantized and hit>=.6 and fpr<=.25 and score<best:best=score;beststep=step;checkpoint(n,d/'selected.pt',step,trained=step>0,kind='local structural support with illumination negatives and positive-retention gate')
            print('STRUCTURE_SUPPORT_FIT',step,'HIT',round(hit,3),'FPR',round(fpr,3),'SCORE',round(score,3),'BEST',beststep,flush=True)
        if step==args.steps:break
        ids=torch.as_tensor(rng.choice(len(X),8,replace=False),device='cuda');x=X[ids];target=Y[ids]
        opt.zero_grad(set_to_none=True);p=support(n,x);t=target[crop];v=p[crop];pos=t>.5;neg=~pos
        prob=.01+.98*v
        loss=-3*prob[pos].log().mean()-2*(1-prob[neg]).log().mean() if pos.any() else -(1-prob).log().mean()*2
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in n.parameters() if p.requires_grad],2);opt.step()
        with torch.no_grad():
            for (key,kind),(initial,mask) in bounds.items():
                p=n.weight[key] if kind=='w' else n.bias[key]
                delta=.35 if kind=='w' else 12
                p.copy_(initial+(p-initial).clamp(-delta,delta)*mask)
            if hasattr(n,'project_support_geometry'):n.project_support_geometry()
    checkpoint(n,d/'diagnostic-last.pt',args.steps,trained=True,kind='last support fit, not necessarily selected')
    eligible=any(z['step']>=args.warmup and z['true_positive_rate']>=.6 and z['false_positive_rate']<=.25 for z in logs)
    if not eligible:
        checkpoint(n,d/'diagnostic-last.pt',args.steps,trained=True,selection_failed=True)
        save_json(d/'training.json',dict(validation_logs=logs,selection_failed=True,MC_visual_accepted=False,reason='no nontrivial positive-retaining eligible quantized support'))
        print('STRUCTURE_SUPPORT_NO_ELIGIBLE_CANDIDATE',flush=True);return
    final=load_model(d/'selected.pt');raw=final.export();(d/'candidate.bin').write_bytes(raw);eng=IntegerEngine(raw);records=[]
    for name,a,positive in fixtures():
        nodes,_=eng(tensor(a)[None]*128,True);o=rgb(nodes['rgb_head']/128)
        assert torch.equal(final(tensor(a)[None])*128,nodes['rgb_head'])
        allowed=(nodes['up2_low_pw'][:,1]-nodes['up2_low_pw'][:,2]).clamp_min(0)
        fraction=float((allowed>1).float().mean());g=float((nodes['up1_low_pw'][:,3]>63).float().mean())
        panel([a,o],['已知平面或物体输入','学习结构支持 / 实际B1候选'],d/f'{name}.png',width=400)
        records.append(dict(fixture=name,positive=positive,extrusion_fraction=fraction,support_fraction=g))
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');nodes,_=eng(tensor(a)[None]*128,True);o=rgb(nodes['rgb_head']/128)
        Image.fromarray(o).save(d/f'{scene}_student.png')
        g=nodes['up1_low_pw'][0,3].cpu().numpy();g=cv2.resize(g,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_NEAREST)
        heat=np.clip(g*2,0,255).astype(np.uint8);heat=cv2.applyColorMap(heat,cv2.COLORMAP_TURBO)[...,::-1]
        panel([a,o,heat],['原图','实际B1 / 非最终MC','已有通道的结构支持诊断'],d/f'{scene}_comparison.jpg')
    save_json(d/'training.json',dict(steps=args.steps,selected_step=beststep,validation_logs=logs,counterexamples=records,
        train_samples=384,validation_samples=96,train_seed=271111,validation_seed=271112,auxiliary_loss_existing_channel='up1_low_pw:3',
        target='local bounded sharp objects positive, soft spots and broad shadows negative',
        known_ambiguity='a painted hard patch can be identical to a physical object; no reliable global geometry claim',
        minimum_validation_positive_hit_for_selection=.6,positive_loss_weight=3,negative_loss_weight=2,warmup_steps=args.warmup,
        validation_gate_pass=logs[beststep//50]['true_positive_rate']>=.6 and logs[beststep//50]['false_positive_rate']<=.25,
        inference_extra_operators=False,MC_visual_accepted=False,coefficient_sha256=sha(d/'candidate.bin'),seconds=time.perf_counter()-begin))
    print('STRUCTURE_SUPPORT_FIT_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600)
    p.add_argument('--warmup',type=int,default=0);p.add_argument('--dest',default='r18-learned-geometry-support-bce');main(p.parse_args())
