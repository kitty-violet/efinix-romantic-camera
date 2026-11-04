"""Bounded MC geometry capacity trial, distinct from flat pixel enlargement.

Synthetic aligned pairs test rounded-to-cuboid silhouette change and face tones.
They are NOT a general real-scene MC teacher. No target operation is exported.
"""
from core import *
from pixel_targets import panel
import argparse

@torch.no_grad()
def start(hinges=False):
    n=B1().cuda();eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    for p in n.parameters():p.zero_()
    for k in n.weight:getattr(n,k+'_m').fill_(2048)
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for c in range(8):n.weight['stem'][c,c%3,1,1]=.5;n.bias['stem'][c]=64
    for k,channels in [('down1__pw',16),('down2__pw',32)]:
        for c in range(channels):n.weight[k][c,c%3,0,0]=1
    if hinges:
        for c in range(3,32):
            n.weight['down2__pw'][c].zero_()
            n.weight['down2__pw'][c,(c-3)%3,0,0]=1
            n.bias['down2__pw'][c]=-((c-3)//3+1)*12
    for b in ['res1_conv1','res1_conv2','res2_conv1','res2_conv2','res3_conv1','res3_conv2']:
        for c in range(32):n.weight[b+'__pw'][c,c,0,0]=1
        # Small nonzero connections break duplicate-channel symmetry.
        n.weight[b+'__pw'].add_(torch.randn_like(n.weight[b+'__pw'])*.001)
    for k,channels in [('up1_low_pw',16),('up2_low_pw',8)]:
        for c in range(channels):n.weight[k][c,c if hinges else c%3,0,0]=1
    if hinges:
        offsets=[(0,1),(2,1),(1,0),(1,2),(0,0),(0,2),(2,0),(2,2)]
        for c in range(3,32):
            dy,dx=offsets[(c-3)%len(offsets)]
            n.weight['res1_conv1__dw'][c,0,dy,dx]=-1
    for c in range(3):n.weight['rgb_head'][c,c,0,0]=2;n.bias['rgb_head'][c]=-128
    if hinges:n.weight['rgb_head'][:,3:,0,0]=.005
    for k in ['res1_add','res2_add','res3_add']:n.package['residual_encoding'][k]=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    n.training_saturation_leak=.01
    return n

def samples(count,seed,size=96):
    rng=np.random.default_rng(seed);X=[];Y=[];masks=[]
    yy,xx=np.mgrid[:size,:size]
    for i in range(count):
        bg=rng.integers(35,225,3).astype(float);fg=rng.integers(25,230,3).astype(float)
        while np.linalg.norm(fg-bg)<130:fg=rng.integers(25,230,3).astype(float)
        cx=int(rng.integers(8,16))*4;cy=int(rng.integers(8,16))*4
        rx=int(rng.integers(3,7))*4;ry=int(rng.integers(3,7))*4
        circle=((xx-cx)/rx)**2+((yy-cy)/ry)**2<1
        square=(xx>=cx-rx)&(xx<cx+rx)&(yy>=cy-ry)&(yy<cy+ry)
        a=np.broadcast_to(bg,(size,size,3)).copy();b=a.copy()
        a[circle]=fg
        a=cv2.GaussianBlur(a,(5,5),.8,borderType=cv2.BORDER_REPLICATE)
        b[square]=fg
        top=square&(yy<cy-ry+max(4,ry//2));side=square&(xx>=cx+rx-max(4,rx//3))
        b[top]=fg*.88+255*.12;b[side]=fg*.80
        # Existing-light-like broad gradient, no texture. Targets use stepped shape.
        X.append(np.clip(a+.5,0,255).astype(np.uint8));Y.append(np.clip(b+.5,0,255).astype(np.uint8));masks.append(square)
    return X,Y,masks

def save_cp(n,path,step,qat):
    torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),qat=qat,
        step=step,kind='synthetic rounded-to-block geometry capacity probe, not production style'),path)

def main(args):
    setup(271029);dest=OUT/'mc-voxel'/args.name;dest.mkdir(parents=True,exist_ok=True);disk_guard()
    a,b,m=samples(256,271029);va,vb,vm=samples(16,271030)
    X=torch.stack([tensor(z,'cpu') for z in a]).cuda();Y=torch.stack([tensor(z,'cpu') for z in b]).cuda()
    VX=torch.stack([tensor(z,'cpu') for z in va]).cuda();VY=torch.stack([tensor(z,'cpu') for z in vb]).cuda()
    n=start(args.hinge_init);n.quantized=False;opt=torch.optim.Adam(n.parameters(),lr=.0002 if args.corner_weight>3 else .0006)
    vw=1+args.corner_weight*((VX-VY).abs().mean(1,keepdim=True)>.1)
    best=float('inf');beststep=None;logs=[];rng=np.random.default_rng(271031);begin=time.perf_counter()
    for step in range(1001):
        if time.perf_counter()-begin>600:raise TimeoutError('10-minute geometry capacity limit')
        if step==300:
            n.calibrate();n.quantized=True
            opt=torch.optim.Adam(n.parameters(),lr=.00004 if args.corner_weight>3 else .00012)
        if step%100==0:
            with torch.no_grad():
                ve=(n(VX)-VY).abs()*128
                mae=float(ve.mean());score=float((ve*vw).sum()/vw.expand_as(ve).sum())
            logs.append(dict(step=step,qat=n.quantized,validation_mae=mae,weighted_shape_error=score,seconds=time.perf_counter()-begin))
            if n.quantized and score<best:best=score;beststep=step;save_cp(n,dest/'selected.pt',step,True)
            print('MC_SHAPE_STEP',step,'QAT',n.quantized,'VAL',round(score,3),flush=True)
        if step==1000:break
        ids=torch.as_tensor(rng.choice(len(X),12,replace=False),device='cuda');x,y=X[ids],Y[ids]
        opt.zero_grad(set_to_none=True);o=n(x)
        shape=(x-y).abs().mean(1,keepdim=True)>.1
        loss=((o-y).abs()*(1+args.corner_weight*shape)).mean()
        dx=(o[:,:,:,1:]-o[:,:,:,:-1])-(y[:,:,:,1:]-y[:,:,:,:-1])
        dy=(o[:,:,1:]-o[:,:,:-1])-(y[:,:,1:]-y[:,:,:-1])
        loss=loss+.2*(dx.abs().mean()+dy.abs().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),1);opt.step()
    model=load_model(dest/'selected.pt');engine=IntegerEngine(model.export());(dest/'candidate.bin').write_bytes(model.export())
    scores=[]
    for j in range(4):
        z=rgb(engine(tensor(va[j])[None]*128)/128)
        assert torch.equal(engine(tensor(va[j])[None]*128),model(tensor(va[j])[None])*128)
        panel([va[j],vb[j],z],['构造输入：圆润外形','目标：方块外形与块面明暗','学生：纯B1整数'],dest/f'case_{j:02d}.png',width=384)
        bg=vb[j][0,0].astype(float);pred=np.linalg.norm(z.astype(float)-bg,axis=2)>45
        iou=float((pred&vm[j]).sum()/max(1,(pred|vm[j]).sum()));scores.append(dict(id=j,mask_iou=iou,
            target_mae=float(np.abs(z.astype(float)-vb[j]).mean())))
    for s in ['outdoor','indoor']:
        z=read(OUT/f'targets/{s}_source.png');out=rgb(engine(tensor(z)[None]*128)/128)
        Image.fromarray(out).save(dest/f'{s}_student.png')
        panel([z,out],['原图','MC构造试训学生 / 未达真实场景风格'],dest/f'{s}_comparison.jpg')
    save_json(dest/'report.json',dict(trained=True,steps=1000,selected_step=beststep,logs=logs,
        best_qat_weighted_shape_error=best,corner_weight=args.corner_weight,hinge_initialization=args.hinge_init,synthetic_mask_metrics=scores,
        scope='Only bounded synthetic capacity experiment. AI concept is not aligned supervision. Not an accepted/general real-scene MC style.',
        goal='Object silhouette cubification, separate from flat pixel route',
        real_scene_style_accepted=False,coefficient_sha256=sha(dest/'candidate.bin'),
        topology_changed=False,inference_extra_operators=False,seconds=time.perf_counter()-begin))
    print('MC_SHAPE_PROBE_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='capacity');p.add_argument('--corner-weight',type=float,default=3);p.add_argument('--hinge-init',action='store_true')
    main(p.parse_args())
