"""QAT readout/threshold calibration with explicit 3x face-region supervision."""
from core import *
from train_pixel import data,checkpoint
from pixel_targets import panel
from r3_palette_blocks import geometry
import r10_mass_cuboids as r10
import r11_planar_cuboids as r11
import r12_solid_cuboids as r12
import r13_region_cuboids as r13
import argparse

FAMILIES=dict(r10=r10,r11=r11,r12=r12,r13=r13)

def connection_loss(output,target):
    # Directly compare neighbours in the visible integer output, across1/2/4
    # native cells. Flat target neighbours must join; target face boundaries
    # must retain their contrast. No comparison to hidden/base image layers.
    output=output[:,:,::4,::4];target=target[:,:,::4,::4];loss=output.sum()*0;count=0
    for delta in [1,2,4]:
        for axis in [2,3]:
            a=[slice(None)]*4;b=a.copy();a[axis]=slice(delta,None);b[axis]=slice(None,-delta)
            pg=output[tuple(a)]-output[tuple(b)];tg=target[tuple(a)]-target[tuple(b)]
            flat=(tg.abs().amax(1,keepdim=True)<3/128).float()
            edge=(tg.abs().amax(1,keepdim=True)>12/128).float()
            loss+=(pg.abs()*flat+(pg-tg).abs()*edge).sum()/((flat+edge).sum()*3).clamp_min(1);count+=1
    return loss/count

def target_weight(a,module,config):
    t=module.teacher(a,config);f,h,d,_,_=module.fields(a,config)
    face=((d-f)>.03).astype(np.float32).repeat(4,0).repeat(4,1)
    return t,1+2*face

def main(args):
    setup(271071);dest=args.folder;meta=json.loads((dest/'seed.json').read_text('utf-8'));config=meta['config'];module=FAMILIES[args.family]
    n=load_model(dest/'initial.pt');n.training_saturation_leak=.01;n.train();disk_guard();arrays,vals=data()
    rng=np.random.default_rng(271071);X=[];Y=[];W=[]
    for a in arrays:
        t,w=target_weight(a,module,config);h,ww=a.shape[:2]
        for _ in range(5):
            yy=int(rng.integers(0,(h-192)//4+1))*4;xx=int(rng.integers(0,(ww-192)//4+1))*4
            X.append(tensor(a[yy:yy+192,xx:xx+192],'cpu'));Y.append(tensor(t[yy:yy+192,xx:xx+192],'cpu'))
            W.append(torch.tensor(w[yy:yy+192,xx:xx+192][None]))
    X,Y,W=[torch.stack(z).cuda() for z in [X,Y,W]]
    VX=torch.stack([tensor(a[120:312,444:636],'cpu') for a in vals]).cuda()
    labels=[target_weight(a,module,config) for a in vals]
    VY=torch.stack([tensor(t[120:312,444:636],'cpu') for t,w in labels]).cuda()
    VW=torch.stack([torch.tensor(w[120:312,444:636][None]) for t,w in labels]).cuda()
    for p in n.parameters():p.requires_grad_(False)
    head='rgb_head';iw=n.weight[head].detach().clone();ib=n.bias[head].detach().clone()
    n.weight[head].requires_grad_(True);n.bias[head].requires_grad_(True)
    n.weight[head].register_hook(lambda g:g*(iw!=0))
    threshold=dict(r10='down2__pw',r11='res1_conv2__pw',r12='res1_conv1__pw',r13='down2__pw')[args.family]
    tb=n.bias[threshold];ti=tb.detach().clone();tb.requires_grad_(True)
    mask=torch.zeros_like(tb);mask[:10 if args.family=='r12' else 15]=1;tb.register_hook(lambda g:g*mask)
    groups=[dict(params=[n.weight[head]],lr=.00012),dict(params=[n.bias[head]],lr=.006),dict(params=[tb],lr=.5)]
    spatial={}
    if args.family=='r13':
        for base in r13.BODY[:4]:
            key=base+'__dw';dw=n.weight[key];db=n.bias[key]
            spatial[key]=(dw.detach().clone(),db.detach().clone());dw.requires_grad_(True);db.requires_grad_(True)
            support=(dw!=0).float();rowmask=torch.zeros_like(db);rowmask[:31]=1
            dw.register_hook(lambda g,m=support:g*m);db.register_hook(lambda g,m=rowmask:g*m)
        groups += [dict(params=[n.weight[k] for k in spatial],lr=.0004),dict(params=[n.bias[k] for k in spatial],lr=.005)]
    opt=torch.optim.Adam(groups)
    crop=(slice(None),slice(None),slice(40,-40),slice(40,-40))
    generator=torch.Generator(device='cuda').manual_seed(271072)
    VN=(VX+torch.randn(VX.shape,generator=generator,device='cuda')*.8/128).clamp(-1,127/128)
    best=float('inf');beststep=0;logs=[];begin=time.perf_counter()
    for step in range(args.steps+1):
        if time.perf_counter()-begin>900:raise TimeoutError('15 minute fit limit')
        if step%50==0 or step==args.steps:
            with torch.no_grad():
                o=n(VX);on=n(VN);delta=(o[crop]-VY[crop]).abs()*128
                mae=float(delta.mean());face_mae=float((delta*(VW[crop]-1)).sum()/((VW[crop]-1).sum()*3).clamp_min(1))
                weighted=float((delta*VW[crop]).sum()/(VW[crop].sum()*3));noise=float((on[crop]-o[crop]).abs().mean()*128)
                conn=float(connection_loss(o[crop],VY[crop])*128) if args.family=='r13' else 0.
                score=weighted+.35*noise+.25*conn
            logs.append(dict(step=step,validation_mae=mae,face_mae=face_mae,weighted_mae=weighted,noise_mae=noise,connection_mae=conn,selection_score=score,seconds=time.perf_counter()-begin))
            if score<best:
                best=score;beststep=step;checkpoint(n,dest/'selected.pt',step,trained=step>0,revision=args.family,teacher_config=meta)
            print('CUBOID_FIT',dest.name,step,'MAE',round(mae,3),'FACE',round(face_mae,3),'CONNECT',round(conn,3),'NOISE',round(noise,3),'BEST',beststep,flush=True)
        if step==args.steps:break
        ids=torch.as_tensor(rng.choice(len(X),6,replace=False),device='cuda');x,y,w=X[ids],Y[ids],W[ids]
        noisy=(x+torch.randn_like(x)*.8/128).clamp(-1,127/128)
        opt.zero_grad(set_to_none=True);o=n(x);on=n(noisy)
        fit=((o[crop]-y[crop]).abs()*w[crop]).sum()/(3*w[crop].sum())
        stable=(on[crop]-o[crop]).abs().mean()
        conn=connection_loss(o[crop],y[crop]) if args.family=='r13' else o.sum()*0
        loss=fit+.35*stable+.25*conn;loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in n.parameters() if p.requires_grad],2);opt.step()
        with torch.no_grad():
            hw=n.weight[head];hb=n.bias[head]
            hw.copy_(torch.maximum(torch.minimum(hw,iw+.06),iw-.06));hb.copy_(torch.maximum(torch.minimum(hb,ib+5),ib-5))
            # Keep neutral colours neutral, retain the three-face signs/gains,
            # and never let validation improve by erasing the cuboid shading.
            hw[:,:3,0,0]=hw[:,:3,0,0].mean(0)[None];hw[:,3:5]=iw[:,3:5]
            hw[:,1,0,0]=hw[:,1,0,0].clamp_max(-.2);hw[:,2,0,0]=hw[:,2,0,0].clamp_min(.4)
            hb.copy_(ib+(hb-ib).mean())
            tb.copy_(torch.maximum(torch.minimum(tb,ti+31.5),ti-31.5))
            if args.family=='r12':
                for j in range(5):tb[2*j:2*j+2]=tb[2*j:2*j+2].mean()
                tb[10:]=ti[10:]
            else:
                for j in range(5):
                    ids2=[j,5+j,10+j];tb[ids2]=tb[ids2].mean()
                tb[15:]=ti[15:]
            for key,(w0,b0) in spatial.items():
                w=n.weight[key];b=n.bias[key];support=(w0!=0).float()
                w.copy_(torch.maximum(torch.minimum(w,w0+.0625),w0-.0625)*support)
                b.copy_(torch.maximum(torch.minimum(b,b0+8),b0-8))
                # Front/top/side use exactly the same regularised region.
                for j in range(5):
                    rows=[j,5+j,10+j];w[rows]=w[rows].mean(0)[None];b[rows]=b[rows].mean()
                b[31:]=b0[31:]
    final=load_model(dest/'selected.pt');raw=final.export();(dest/'candidate.bin').write_bytes(raw);engine=IntegerEngine(raw);static=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');t=module.teacher(a,config);o=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(final(tensor(a)[None])*128,engine(tensor(a)[None]*128))
        Image.fromarray(o).save(dest/f'{scene}_trained.png');panel([a,t,o],['原图','大块三面体目标','QAT后 / 纯B1整数'],dest/f'{scene}_trained_comparison.jpg')
        record=dict(scene=scene,target_mae=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o))
        if args.family=='r13':record['regions']=r13.region_stats(o)
        static.append(record)
    save_json(dest/'training.json',dict(family=args.family,steps=args.steps,selected_step=beststep,logs=logs,static=static,
        coefficient_sha256=sha(dest/'candidate.bin'),face_region_loss_weight=3,noise_std_rgb=.8,
        connection_loss_weight=.25 if args.family=='r13' else 0,trainable_spatial_consensus_layers=list(spatial),
        geometry='ordinary fixed-B1 morphology/cuboid volume; bounded threshold/readout QAT plus r13 tied spatial consensus kernels',
        extra_inference_operators=False,visual_accepted=False,temporal_accepted=False,seconds=time.perf_counter()-begin))
    print('CUBOID_FIT_COMPLETE',dest.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=list(FAMILIES),required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--steps',type=int,default=400);main(p.parse_args())
