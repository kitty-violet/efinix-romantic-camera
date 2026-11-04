"""Bounded full-RGB scene learning diagnostic, not general style acceptance.

Only existing training-video still is used. Review street/corridor remain unseen.
Manual registration, exclusion masks and geometry losses are TRAINING ONLY.
All 10819 B1 coefficients learn; no auxiliary support-only fitting.
"""
from core import *
from probe_mc_shapes import start
from train_pixel import checkpoint
import argparse

D=OUT/'r34-full-scene-fit'
SOURCE=WORK/'runs/b1-distillation-20261003/data/f30b208e3685_000034_source.png'
GENERATED=Path(r'C:\Users\Lenovo\.codex\generated_images\01a17b77-a1ea-7d51-8ec6-e071fcbdcf72\exec-c81e8a3e-2a78-45b9-9347-e3d42230dc60.png')

def prepare():
    D.mkdir(exist_ok=True);a=read(SOURCE)
    assert a.shape==(720,1280,3)
    split=json.loads((OUT/'data_split.json').read_text('utf-8'))
    assert str(SOURCE) in {r['path'] for r in split['training']}
    assert not any('121509' in str(c['source']) for c in json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))['clips'])
    raw=read(GENERATED);b=cv2.resize(raw,(1280,720),interpolation=cv2.INTER_AREA)
    # Four corresponding path-edge anchors, manually estimated, not exact GT.
    target=np.float32([[340,719],[768,719],[552,422],[637,422]])
    source=np.float32([[375,719],[780,719],[553,420],[637,420]])
    H=cv2.getPerspectiveTransform(target,source)
    b=cv2.warpPerspective(b,H,(1280,720),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    mask=np.ones((720,1280),np.float32)
    # Discard generated sky/clouds and distant people. Not inference segmentation.
    cv2.fillPoly(mask,[np.array([[420,0],[900,0],[755,310],[620,360],[490,288]],np.int32)],0)
    cv2.fillPoly(mask,[np.array([[518,358],[680,358],[680,446],[518,446]],np.int32)],0)
    mask[:40]=0;mask[-40:]=0;mask[:,:40]=0;mask[:,-40:]=0
    grad=np.maximum(np.max(np.abs(b.astype(float)-np.roll(b,1,axis=0)),2),
                    np.max(np.abs(b.astype(float)-np.roll(b,1,axis=1)),2))
    interior=(cv2.dilate(grad.astype(np.float32),np.ones((11,11),np.uint8))<4).astype(np.float32)*mask
    Image.fromarray(a).save(D/'source.png');Image.fromarray(b).save(D/'target_registered.png')
    Image.fromarray((mask*255).astype(np.uint8)).save(D/'training_mask.png')
    save_json(D/'teacher.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),generated=str(GENERATED),generated_sha256=sha(GENERATED),
        source_video='VID_20260925_121509.mp4',split='existing training source only',
        raw_generated_shape=list(raw.shape),registration_H=H.tolist(),anchors_target=target.tolist(),anchors_source=source.tolist(),
        alignment='manually estimated path alignment; geometry changes are intended; not exact dense correspondence',
        excluded='sky, distant people, 40px image border',coverage=float(mask.mean()),
        purpose='single training-scene capacity diagnostic; cannot establish general scene MC style',
        target_scope='large connected background blocks; clean planes; no material texture',
        extra_inference_operators=False))
    return a,b,mask,interior

def image_metrics(o,t,m,interior):
    d=np.abs(o.astype(float)-t)
    og=cv2.cvtColor(o,cv2.COLOR_RGB2GRAY);tg=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
    oe=cv2.Canny(og,25,70)>0;te=cv2.Canny(tg,25,70)>0
    om=oe&(m>.5);tm=te&(m>.5)
    dt_t=cv2.distanceTransform((~te).astype(np.uint8),cv2.DIST_L2,5)
    dt_o=cv2.distanceTransform((~oe).astype(np.uint8),cv2.DIST_L2,5)
    precision=float((dt_t[om]<=4).mean()) if om.any() else 0.
    recall=float((dt_o[tm]<=4).mean()) if tm.any() else 0.
    hp=o.astype(np.float32)-cv2.blur(o.astype(np.float32),(7,7))
    thp=t.astype(np.float32)-cv2.blur(t.astype(np.float32),(7,7))
    return dict(rgb_mae=float(d[m>.5].mean()),edge_precision_4px=precision,edge_recall_4px=recall,
        edge_f1_4px=2*precision*recall/(precision+recall+1e-8),
        plane_interior_highpass_rms=float(np.sqrt((hp[interior>.5]**2).mean())),
        target_plane_interior_highpass_rms=float(np.sqrt((thp[interior>.5]**2).mean())))

def loss_fn(o,t,m,interior):
    mae=((o-t).abs()*m).sum()/(m.sum()*3+1e-6)
    edges=0.
    for k in [4,12]:
        for axis in [2,3]:
            sl1=[slice(None)]*4;sl2=sl1.copy();sl1[axis]=slice(k,None);sl2[axis]=slice(None,-k)
            sl1=tuple(sl1);sl2=tuple(sl2)
            delta=(o[sl1]-o[sl2])-(t[sl1]-t[sl2]);valid=m[sl1]*m[sl2]
            edges+=(delta.abs()*valid).sum()/(valid.sum()*3+1e-6)
    oh=o-F.avg_pool2d(o,7,1,3);th=t-F.avg_pool2d(t,7,1,3)
    planar=((oh-th).abs()*interior).sum()/(interior.sum()*3+1e-6)
    return mae+.15*edges+.22*planar

def main(args):
    global D
    if args.hard_init:D=OUT/'r35-hard-scene-fit'
    setup(271180);disk_guard();a,b,mask,inside=prepare()
    shutil.copy2(__file__,D/'source_used.py')
    n=load_model(OUT/'r28-shared-cube-scene/initial.pt') if args.hard_init else start(hinges=True)
    n.quantized=bool(args.hard_init)
    n.training_saturation_leak=.02
    X=tensor(a)[None];Y=tensor(b)[None];M=torch.from_numpy(mask).cuda()[None,None];I=torch.from_numpy(inside).cuda()[None,None]
    rng=np.random.default_rng(271181);opt=torch.optim.Adam(n.parameters(),lr=.00015 if args.hard_init else .00035)
    begin=time.perf_counter();logs=[];best=float('inf');beststep=-1
    save_json(D/'RUNNING.json',dict(steps=args.steps,budget_seconds=600,training_scene=str(SOURCE),all_coefficients_trainable=True,
        scope='single-frame capacity diagnostic; not independent validation',outputs_are_raw_B1=True))
    for step in range(args.steps+1):
        if time.perf_counter()-begin>600:raise TimeoutError('600s bounded scene-capacity experiment')
        if step==200 and not args.hard_init:
            n.calibrate();n.quantized=True;opt=torch.optim.Adam(n.parameters(),lr=.00009)
        if step%100==0 or step==args.steps:
            with torch.no_grad():
                o=n(X);value=float(loss_fn(o,Y,M,I));out=rgb(o)
                metrics=image_metrics(out,b,mask,inside)
            logs.append(dict(step=step,qat=n.quantized,loss=value,seconds=time.perf_counter()-begin,**metrics))
            # Hard-init comparison is selected with shape precision/recall as
            # well as colour and planarity. Smooth colour fit alone cannot win.
            selection=value+(.35*(1-metrics['edge_f1_4px']) if args.hard_init else 0.)
            if n.quantized and selection<best:
                best=selection;beststep=step
                torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
                    qat=True,step=step,trained=True,
                    initialization='shared three-face hard geometry' if args.hard_init else 'fresh RGB hinge initialization',
                    purpose='single training-scene capacity diagnostic',independent_validation=False),D/'selected.pt')
            print('SCENE_FIT',json.dumps(logs[-1]),flush=True)
            Image.fromarray(out).save(D/f'fit_step_{step:04d}.png')
            save_json(D/'progress.json',dict(logs=logs,selected_step=beststep))
        if step==args.steps:break
        patches=[]
        for j in range(8):
            for attempt in range(30):
                yy=int(rng.integers(0,(720-192)//4+1))*4;xx=int(rng.integers(0,(1280-192)//4+1))*4
                if mask[yy+40:yy+152,xx+40:xx+152].mean()>.55:break
            patches.append((yy,xx))
        crop=lambda z:torch.cat([z[:,:,yy:yy+192,xx:xx+192] for yy,xx in patches])
        x,y,m,inte=map(crop,(X,Y,M,I));opt.zero_grad(set_to_none=True)
        o=n(x);loss=loss_fn(o[:,:,40:-40,40:-40],y[:,:,40:-40,40:-40],m[:,:,40:-40,40:-40],inte[:,:,40:-40,40:-40])
        if args.hard_init:
            # Distinguish long target edges from source microtexture.
            oc=o[:,:,40:-40,40:-40];yc=y[:,:,40:-40,40:-40];mc=m[:,:,40:-40,40:-40]
            for axis in (2,3):
                e1=[slice(None)]*4;e2=e1.copy();e1[axis]=slice(8,None);e2[axis]=slice(None,-8);e1=tuple(e1);e2=tuple(e2)
                td=yc[e1]-yc[e2];od=oc[e1]-oc[e2]
                weight=(td.abs().mean(1,keepdim=True)>.16).float()*mc[e1]*mc[e2]
                loss=loss+.3*((od-td).abs()*weight).sum()/(weight.sum()*3+1e-6)
        loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),1.);opt.step()
    selected=load_model(D/'selected.pt');raw=selected.export();(D/'candidate.bin').write_bytes(raw);engine=IntegerEngine(raw)
    with torch.inference_mode():
        actual=engine(X*128);assert torch.equal(actual,selected(X)*128)
        out=rgb(actual/128);Image.fromarray(out).save(D/'training_scene_integer.png')
        board([[a,b,out]],D/'training_scene_comparison.jpg',['训练场景','大块场景目标 / 非B1','实际B1整数 / 容量诊断'],width=640)
        for name,path in [('corridor',OUT/'data-quality/VID_20260924_231158_source.png'),('outdoor',OUT/'targets/outdoor_source.png')]:
            src=read(path);z=rgb(engine(tensor(src)[None]*128)/128)
            Image.fromarray(z).save(D/f'unseen_{name}_integer.png')
            board([[src,z]],D/f'unseen_{name}_comparison.jpg',['独立来源原图','实际B1整数 / 未定稿'],width=640)
    save_json(D/'report.json',dict(complete=True,steps=args.steps,selected_step=beststep,logs=logs,
        trained_parameters=sum(p.numel() for p in n.parameters()),source_sha256=sha(SOURCE),target_sha256=sha(D/'target_registered.png'),
        coefficient_sha256=sha(D/'candidate.bin'),native_resolution=[1280,720],ordinary_QAT_equals_decoded_BIN=True,
        metrics=image_metrics(out,b,mask,inside),hard_geometry_initialization=args.hard_init,seen_scene_fit_only=True,general_MC_style_pass=False,
        inference_extra_operators=False,seconds=time.perf_counter()-begin,script_sha256=sha(__file__),disk=disk_guard()))
    (D/'RUNNING.json').unlink();print('SCENE_FIT_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=800);p.add_argument('--hard-init',action='store_true');main(p.parse_args())
