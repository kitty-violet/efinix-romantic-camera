"""Constrained QAT on held-out video groups, including RGB noise consistency."""
from core import *
from train_pixel import data,checkpoint
from pixel_targets import panel
import r3_palette_blocks as r3
import r4_box_projector as r4
import r5_block_faces as r5
import r6_five_tones as r6
import r7_shared_background as r7
import argparse

def main(args):
    setup(271042);dest=Path(args.folder);meta=json.loads((dest/'seed.json').read_text('utf-8'))
    n=load_model(dest/'initial.pt');n.training_saturation_leak=.01;n.train();disk_guard()
    if args.family=='r3':
        target=lambda a:r3.teacher(a,meta['config'])
        fields=['res3_conv2__pw','rgb_head'];palette_row=None
    elif args.family=='r4':
        target=lambda a:r4.teacher(a,meta['depth'],meta['face'],meta['preclean'],meta['chroma_blend'],meta['corners'],meta.get('soft_polarity',False))
        fields=['up1_low_pw','rgb_head'];palette_row=9
    elif args.family=='r5':
        target=r5.teacher;fields=['res3_conv1__pw','rgb_head'];palette_row=21
    elif args.family=='r6':
        target=r6.teacher;fields=['rgb_head'];palette_row=None
    else:
        target=r7.teacher;fields=['rgb_head'];palette_row=None
    arrays,vals=data();rng=np.random.default_rng(271042);X=[];Y=[]
    for a in arrays:
        t=target(a);h,w=a.shape[:2]
        for _ in range(5):
            y=int(rng.integers(0,(h-192)//4+1))*4;x=int(rng.integers(0,(w-192)//4+1))*4
            X.append(tensor(a[y:y+192,x:x+192],'cpu'));Y.append(tensor(t[y:y+192,x:x+192],'cpu'))
    X=torch.stack(X).cuda();Y=torch.stack(Y).cuda()
    VX=torch.stack([tensor(a[120:312,444:636],'cpu') for a in vals]).cuda()
    VY=torch.stack([tensor(target(a)[120:312,444:636],'cpu') for a in vals]).cuda()
    for p in n.parameters():p.requires_grad_(False)
    initial={};biasparams=[];wparams=[]
    for k in fields:
        n.weight[k].requires_grad_(True);n.bias[k].requires_grad_(True)
        mask=(n.weight[k].detach()!=0).float()
        if k=='up1_low_pw':mask[:palette_row]=0
        if args.family=='r5' and k=='res3_conv1__pw':mask[:21]=0
        bm=(mask.flatten(1).sum(1)>0).float()
        n.weight[k].register_hook(lambda g,m=mask:g*m);n.bias[k].register_hook(lambda g,m=bm:g*m)
        wparams.append(n.weight[k]);biasparams.append(n.bias[k])
        initial[k]=(n.weight[k].detach().clone(),n.bias[k].detach().clone())
    params=[dict(params=wparams,lr=.00015),dict(params=biasparams,lr=.01)]
    if args.family=='r3':
        n.bias['down2__pw'].requires_grad_(True);threshold_init=n.bias['down2__pw'].detach().clone()
        params.append(dict(params=[n.bias['down2__pw']],lr=1.))
    opt=torch.optim.Adam(params);crop=(slice(None),slice(None),slice(40,-40),slice(40,-40))
    # Fixed validation noise, never sampled from held-out demonstration videos.
    generator=torch.Generator(device='cuda').manual_seed(271044)
    VN=(VX+torch.randn(VX.shape,generator=generator,device='cuda')*.8/128).clamp(-1,127/128)
    best=float('inf');beststep=0;logs=[];begin=time.perf_counter()
    for step in range(args.steps+1):
        if time.perf_counter()-begin>900:raise TimeoutError('15 minute fit budget')
        if step%50==0 or step==args.steps:
            with torch.no_grad():
                vo=n(VX);vn=n(VN);mae=float((vo[crop]-VY[crop]).abs().mean()*128)
                noise=float((vn[crop]-vo[crop]).abs().mean()*128);score=mae+.35*noise
            logs.append(dict(step=step,validation_mae=mae,noise_mae=noise,selection_score=score,seconds=time.perf_counter()-begin))
            if score<best:best=score;beststep=step;checkpoint(n,dest/'selected.pt',step,trained=step>0,revision=args.family,teacher_config=meta)
            print('PALETTE_FIT',dest.name,step,'MAE',round(mae,3),'NOISE',round(noise,3),'BEST',beststep,flush=True)
        if step==args.steps:break
        ids=torch.as_tensor(rng.choice(len(X),6,replace=False),device='cuda');x,y=X[ids],Y[ids]
        noise=torch.randn_like(x)*.8/128;noisy=(x+noise).clamp(-1,127/128)
        opt.zero_grad(set_to_none=True);o=n(x);on=n(noisy)
        fit=(o[crop]-y[crop]).abs().mean()
        stable=(on[crop]-o[crop]).abs().mean()
        loss=fit+.35*stable;loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in n.parameters() if p.requires_grad],2);opt.step()
        with torch.no_grad():
            for k,(w,b) in initial.items():
                n.weight[k].copy_(torch.maximum(torch.minimum(n.weight[k],w+.06),w-.06))
                n.bias[k].copy_(torch.maximum(torch.minimum(n.bias[k],b+6),b-6))
            if args.family=='r3':n.bias['down2__pw'].copy_(torch.maximum(torch.minimum(n.bias['down2__pw'],threshold_init+63.5),threshold_init-63.5))
            if args.family=='r5':
                # Keep neutral colours neutral and retain visible four-pixel face contrast.
                hw=n.weight['rgb_head'];hb=n.bias['rgb_head'];iw,ib=initial['rgb_head']
                hw[:,:3,0,0]=hw[:,:3,0,0].mean(0)[None]
                hw[:,3:5]=iw[:,3:5]
                hw[:,5,0,0]=hw[:,5,0,0].mean().clamp(.24,.32)
                hw[:,6,0,0]=hw[:,6,0,0].mean().clamp(-.38,-.28)
                hb.copy_(ib+(hb-ib).mean())
                pw=n.weight['res3_conv1__pw'];pb=n.bias['res3_conv1__pw']
                for group in range(2):
                    start=21+group*4
                    pb[21+group]=63.5-127*pw[21+group,start:start+2,0,0].sum()-63.5*pw[21+group,29+group,0,0]
            if args.family=='r6':
                hw=n.weight['rgb_head'];hb=n.bias['rgb_head'];iw,ib=initial['rgb_head']
                hw[:,:4,0,0]=hw[:,:4,0,0].mean(0)[None]
                hw[:,4:6]=iw[:,4:6]
                hw[:,6,0,0]=hw[:,6,0,0].mean().clamp(.25,.35)
                hw[:,7,0,0]=hw[:,7,0,0].mean().clamp(-.4,-.3)
                hb.copy_(ib+(hb-ib).mean())
            if args.family=='r7':
                hw=n.weight['rgb_head'];hb=n.bias['rgb_head'];iw,ib=initial['rgb_head']
                hw[:,:5,0,0]=hw[:,:5,0,0].mean(0)[None]
                hw[:,5:7]=iw[:,5:7]
                hw[:,7,0,0]=1
                hb.copy_(ib+(hb-ib).mean())
    final=load_model(dest/'selected.pt');raw=final.export();(dest/'candidate.bin').write_bytes(raw);engine=IntegerEngine(raw)
    static=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');t=target(a);o=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(final(tensor(a)[None])*128,engine(tensor(a)[None]*128))
        Image.fromarray(o).save(dest/f'{scene}_trained.png')
        panel([a,t,o],['原图','目标 / 非B1','QAT后 / 纯B1整数'],dest/f'{scene}_trained_comparison.jpg')
        static.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=r3.geometry(o)))
    save_json(dest/'training.json',dict(family=args.family,steps=args.steps,selected_step=beststep,selection_score=best,
        logs=logs,static=static,coefficient_sha256=sha(dest/'candidate.bin'),
        training='constrained readout + r3 threshold bias; ordinary QAT only; geometry truth tables fixed',
        added_inference_operators=False,temporal_accepted=False,visual_accepted=False,seconds=time.perf_counter()-begin))
    print('PALETTE_FIT_COMPLETE',dest.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['r3','r4','r5','r6','r7'],required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--steps',type=int,default=400)
    main(p.parse_args())
