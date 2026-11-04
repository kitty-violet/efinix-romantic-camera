"""Fixed B1 pixel route initialization + QAT. No old style weights survive."""
from core import *
from pixel_targets import target,CONFIG,panel
import argparse

@torch.no_grad()
def pixel_start():
    n=B1().cuda()
    for p in n.parameters():p.zero_()
    for k in n.weight:getattr(n,k+'_m').fill_(2048)
    eye=torch.zeros((3,3),device='cuda');eye[1,1]=1
    avg=torch.zeros_like(eye);avg[1:,1:]=.25
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for j in range(3):n.weight['stem'][j,j,1,1]=.5;n.bias['stem'][j]=64
    for k in ['down1__dw','down2__dw']:n.weight[k].copy_(avg[None,None].expand_as(n.weight[k]))
    for j in range(3):n.weight['down1__pw'][j,j,0,0]=1
    for c in range(3):
        for i,t in enumerate((np.arange(5)+.5)*51):
            j=c*5+i;n.weight['down2__pw'][j,c,0,0]=1
            n.bias['down2__pw'][j]=-(t-CONFIG['transition_rgb']/2)/2
            n.weight['res1_conv1__pw'][j,j,0,0]=127/(CONFIG['transition_rgb']/2)
            n.weight['res1_conv2__pw'][c,j,0,0]=.2
    for base in ['res2_conv1','res2_conv2','res3_conv1','res3_conv2']:
        for c in range(3):n.weight[base+'__pw'][c,c,0,0]=1
    for base in ['up1_low_pw','up2_low_pw']:
        for c in range(3):n.weight[base][c,c,0,0]=1
    for c in range(3):n.weight['rgb_head'][c,c,0,0]=255/127;n.bias['rgb_head'][c]=-128
    for name in ['res1_add','res2_add','res3_add']:
        n.package['residual_encoding'][name]=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/120).clamp_min(64)
        m=(2**need.log2().ceil()).clamp(64,131071)
        getattr(n,k+'_m').copy_(m.double()[None,:,None,None])
    n.quantized=True;n.training_saturation_leak=.01
    return n

def checkpoint(n,path,step,**fields):
    torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
        qat=True,step=step,initialization='all coefficients reset; constructed RGB pool/step/nearest route',**fields),path)

def data():
    audit=json.loads((ROOT/'output/b1-ink-wash-20261026/audit.json').read_text('utf-8'))
    records=audit['train_records']
    # Local future-fit policy. Never change the source dataset or old packages.
    # Source saturation is screened before teacher/style output is consulted.
    quality=OUT/'data-quality/training-exposure-audit.json'
    selection=OUT/'data-quality/selection.json'
    exclusions=[]
    if quality.exists() and selection.exists():
        excluded={r['path'] for r in json.loads(quality.read_text('utf-8'))['records'] if r['exclude_from_future_local_fit']}
        rejected_sources=set(json.loads(selection.read_text('utf-8'))['exclude_training_video_names'])
        exclusions=[r for r in records if r['path'] in excluded or r['video'] in rejected_sources]
        records=[r for r in records if r['path'] not in excluded and r['video'] not in rejected_sources]
    independent={Path(r['video']).name for r in audit['independent_videos']}
    # Select by established split and source video, never by held-out visual success.
    train=[r for r in records if r['split']=='training']
    val=[r for r in records if r['split']=='validation']
    train=train[::max(1,len(train)//32)][:32];val=val[:8]
    assert not set(r['video'] for r in train)&set(r['video'] for r in val)
    assert not set(r['video'] for r in train+val)&independent
    a=[read(r['path']) for r in train];v=[read(r['path']) for r in val]
    assert all(z.shape[0]%4==0 and z.shape[1]%4==0 for z in a+v)
    save_json(OUT/'data_split.json',dict(training=train,validation=val,
        excluded_due_to_source_exposure=exclusions,
        exposure_policy=str(selection) if selection.exists() else None,
        independent_preview='outdoor and non-overexposed corridor; earlier station preview retired; development review, not untouched final test'))
    return a,v

def main(args):
    setup(271027);disk_guard();dest=OUT/'experiment';dest.mkdir(exist_ok=True)
    n=pixel_start();checkpoint(n,dest/'initial.pt',0,trained=False)
    raw=n.export();(dest/'initial.bin').write_bytes(raw)
    arrays,vals=data();rng=np.random.default_rng(271027)
    xs=[];ys=[]
    for a in arrays:
        h,w=a.shape[:2]
        for _ in range(6):
            y=int(rng.integers(0,(h-160)//4+1))*4;x=int(rng.integers(0,(w-160)//4+1))*4
            q=a[y:y+160,x:x+160].copy();xs.append(tensor(q,'cpu'));ys.append(tensor(target(q),'cpu'))
    X=torch.stack(xs).cuda();Y=torch.stack(ys).cuda()
    # Keep the spatial 4x grid exact through the existing nearest path.
    # Only colour thresholds, gate amplitude, and readout coefficients are fitted.
    active=['down2__pw','res1_conv1__pw','res1_conv2__pw','rgb_head']
    for p in n.parameters():p.requires_grad_(False)
    for k in active:
        n.weight[k].requires_grad_(True);n.bias[k].requires_grad_(True)
        mask=(n.weight[k].detach()!=0).float();bm=(mask.flatten(1).sum(1)>0).float()
        n.weight[k].register_hook(lambda g,m=mask:g*m)
        n.bias[k].register_hook(lambda g,m=bm:g*m)
    opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=.002)
    vx=torch.stack([tensor(a[192:352,448:608],'cpu') for a in vals]).cuda()
    vy=torch.stack([tensor(target(a[192:352,448:608]),'cpu') for a in vals]).cuda()
    logs=[];begin=time.perf_counter();best=float('inf');beststep=0
    with torch.no_grad():baseline=float((n(vx)-vy).abs().mean()*128)
    for step in range(args.steps+1):
        if time.perf_counter()-begin>600:raise TimeoutError('600s training budget')
        if step%25==0 or step==args.steps:
            with torch.no_grad():score=float((n(vx)-vy).abs().mean()*128)
            logs.append(dict(step=step,validation_mae=score,seconds=time.perf_counter()-begin))
            if score<best-1e-5:best=score;beststep=step;checkpoint(n,dest/'selected.pt',step,trained=step>0)
            print('STEP',step,'VAL',round(score,4),'BEST',beststep,flush=True)
        if step==args.steps:break
        ids=torch.as_tensor(rng.choice(len(X),8,replace=False),device='cuda')
        x,y=X[ids],Y[ids];opt.zero_grad(set_to_none=True);o=n(x)
        # RGB reconstruction plus differences across 4x-cell borders.
        l=F.smooth_l1_loss(o,y,beta=.06)
        dx=(o[:,:,:,4:]-o[:,:,:,:-4])-(y[:,:,:,4:]-y[:,:,:,:-4])
        dy=(o[:,:,4:]-o[:,:,:-4])-(y[:,:,4:]-y[:,:,:-4])
        loss=l+.12*(dx.abs().mean()+dy.abs().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in n.parameters() if p.requires_grad],2);opt.step()
    checkpoint(n,dest/'last.pt',args.steps,trained=True)
    save_json(dest/'training.json',dict(config=CONFIG,steps=args.steps,selected_step=beststep,
        initial_validation_mae=baseline,selected_validation_mae=best,logs=logs,
        loss='RGB smoothL1 + 4-pixel neighbour difference',trainable_fields=active,
        all_other_coefficients='fixed constructed route, still all ordinary B1 coefficients',
        inference_operators_added=False,old_style_coefficients_retained=False,seconds=time.perf_counter()-begin))
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=250);main(ap.parse_args())
