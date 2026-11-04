"""6x6 target feasibility fit on unchanged B1; never claims exact lattice."""
from core import *
from train_pixel import data,checkpoint
from make_large_pixel_targets import six_target
from pixel_targets import panel

def main():
    setup(271028);disk_guard();dest=OUT/'flat-pixel-v2';dest.mkdir(exist_ok=True)
    model=load_model(OUT/'experiment/selected.pt');model.train();model.training_saturation_leak=.01
    arrays,vals=data();rng=np.random.default_rng(271028)
    X=[];Y=[]
    for a in arrays:
        t=six_target(a);h,w=a.shape[:2]
        for i in range(8):
            # 12 is divisible by both source lattice4 and target lattice6.
            y=int(rng.integers(0,(h-156)//12+1))*12;x=int(rng.integers(0,(w-156)//12+1))*12
            X.append(tensor(a[y:y+156,x:x+156],'cpu'));Y.append(tensor(t[y:y+156,x:x+156],'cpu'))
    X=torch.stack(X).cuda();Y=torch.stack(Y).cuda()
    vx=torch.stack([tensor(a[192:348,444:600],'cpu') for a in vals]).cuda()
    vy=torch.stack([tensor(six_target(a)[192:348,444:600],'cpu') for a in vals]).cuda()
    for p in model.parameters():p.requires_grad_(False)
    spatial=['stem','down1__dw','down2__dw','res2_conv1__dw','res2_conv2__dw','res3_conv1__dw','res3_conv2__dw']
    color=['down2__pw','res1_conv1__pw','res1_conv2__pw','rgb_head']
    # Colour fit retains source routing. Spatial fit may combine adjacent native cells;
    # this is an approximation trial, not a hidden input/output resize.
    for k in spatial+color:
        model.weight[k].requires_grad_(True)
        if k in color:
            mask=(model.weight[k].detach()!=0).float()
            model.weight[k].register_hook(lambda g,m=mask:g*m)
            model.bias[k].requires_grad_(True)
            bm=(mask.flatten(1).sum(1)>0).float();model.bias[k].register_hook(lambda g,m=bm:g*m)
    opt=torch.optim.Adam([dict(params=[model.weight[k] for k in spatial],lr=.00025),
        dict(params=[p for k in color for p in [model.weight[k],model.bias[k]]],lr=.0004)])
    with torch.no_grad():initial=float((model(vx)-vy).abs().mean()*128)
    logs=[];best=initial;beststep=0;checkpoint(model,dest/'selected.pt',0,trained=False,route='6x6 target approximation')
    begin=time.perf_counter()
    for step in range(401):
        if time.perf_counter()-begin>600:raise TimeoutError('600 second trial')
        if step%50==0:
            with torch.no_grad():score=float((model(vx)-vy).abs().mean()*128)
            logs.append(dict(step=step,validation_mae=score,seconds=time.perf_counter()-begin))
            if score<best:best=score;beststep=step;checkpoint(model,dest/'selected.pt',step,trained=True,route='6x6 target approximation')
            print('LARGE_STEP',step,round(score,3),'BEST',beststep,flush=True)
        if step==400:break
        ids=torch.as_tensor(rng.choice(len(X),8,replace=False),device='cuda');x,y=X[ids],Y[ids]
        opt.zero_grad(set_to_none=True);o=model(x)
        # Reconstruction is evaluated against the actual six-pixel labels.
        err=(o-y).abs().mean()
        err.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
    final=load_model(dest/'selected.pt');raw=final.export();(dest/'candidate.bin').write_bytes(raw)
    engine=IntegerEngine(raw);stats=[]
    for s in ['outdoor','indoor']:
        a=read(OUT/f'targets/{s}_source.png');t=six_target(a);out=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(engine(tensor(a)[None]*128),final(tensor(a)[None])*128)
        Image.fromarray(out).save(dest/f'{s}_student.png')
        panel([read(OUT/f'evaluation/{s}_student.png'),t,out],['原4×4学生','6×6目标 / 非B1','放大试训 / 纯B1整数'],dest/f'{s}_comparison.jpg')
        crop=(slice(192,384),slice(448,768))
        panel([t[crop],out[crop]],['6×6目标局部','试训学生局部 / 纯B1整数'],dest/f'{s}_detail.png')
        stats.append(dict(id=s,mae_target=float(np.abs(t.astype(float)-out).mean())))
    save_json(dest/'training.json',dict(initial_mae=initial,best_mae=best,best_step=beststep,logs=logs,
        steps=400,independent=stats,exact_six_grid_claim=False,no_inference_resize_or_filter=True,
        fixed_graph=True,coefficient_sha256=sha(dest/'candidate.bin')))
    print('LARGE_PIXEL_TRIAL_COMPLETE',flush=True)

if __name__=='__main__':main()
