"""Aligned local cuboid-contour teacher + constructed ordinary B1 coefficients.

This targets square local masses and shallow face cues, not semantic 3D remeshing.
All morphology is LABEL generation; the student uses existing DW/PW/ReLU only.
"""
from core import *
from pixel_targets import panel
from train_pixel import pixel_start,checkpoint,data

def teacher(a):
    h,w=a.shape[:2];low=a.astype(np.float32).reshape(h//4,4,w//4,4,3).mean((1,3))
    colour=np.zeros_like(low)
    kernel=np.ones((3,3),np.uint8)
    for c in range(3):
        for t in (np.arange(5)+.5)*51:
            gate=(low[...,c]>t).astype(np.uint8)
            closed=cv2.morphologyEx(gate,cv2.MORPH_CLOSE,kernel,borderType=cv2.BORDER_REPLICATE)
            colour[...,c]+=51*closed
    top=np.pad(colour,((1,0),(0,0),(0,0)),mode='edge')[:-1]
    right=np.pad(colour,((0,0),(0,1),(0,0)),mode='edge')[:,1:]
    light=np.maximum(colour-top,0).mean(2);dark=np.maximum(colour-right,0).mean(2)
    out=colour+.10*light[...,None]-.125*dark[...,None]
    return np.clip(out+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed():
    n=pixel_start()
    for p in n.parameters():p.zero_()
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    avg=torch.zeros_like(eye);avg[1:,1:]=.25
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['stem'][c,c,1,1]=.5;n.bias['stem'][c]=64
    for k in ['down1__dw','down2__dw']:n.weight[k].copy_(avg[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['down1__pw'][c,c,0,0]=1
    for c in range(3):
        for i,t in enumerate((np.arange(5)+.5)*51):
            j=c*5+i
            # Saturated ordinary INT8 PW produces the binary palette decisions.
            n.weight['down2__pw'][j,c,0,0]=127
            n.bias['down2__pw'][j]=-(t/2)*127
            for b in ['res1_conv1','res1_conv2']:
                n.weight[b+'__dw'][j,0].fill_(1/9)
                n.weight[b+'__pw'][j,j,0,0]=9
            n.bias['res1_conv2__pw'][j]=-8*127
            for duplicate in range(3):n.weight['res2_conv1__pw'][c+duplicate*3,j,0,0]=.2
    for c in range(9):n.weight['res2_conv2__pw'][c,c,0,0]=1
    for c in range(3):
        n.weight['res2_conv2__dw'][c+3,0,0,1]=-1
        n.weight['res2_conv2__dw'][c+6,0,1,2]=-1
    for b in ['res3_conv1','res3_conv2']:
        for c in range(9):n.weight[b+'__pw'][c,c,0,0]=1
    for c in range(9):n.weight['up1_low_pw'][c,c,0,0]=1
    for c in range(3):
        n.weight['up2_low_pw'][c,c,0,0]=1
        n.weight['up2_low_pw'][3,c+3,0,0]=1/3
        n.weight['up2_low_pw'][4,c+6,0,0]=1/3
        n.weight['rgb_head'][c,c,0,0]=255/127
        n.weight['rgb_head'][c,3,0,0]=.2
        n.weight['rgb_head'][c,4,0,0]=-.25
        n.bias['rgb_head'][c]=-128
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main():
    setup(271032);dest=OUT/'mc-voxel/local-blocks';dest.mkdir(parents=True,exist_ok=True);disk_guard()
    n=seed();checkpoint(n,dest/'initial.pt',0,trained=False,kind='constructed local square closing and face cues')
    arrays,vals=data();rng=np.random.default_rng(271032);X=[];Y=[]
    for a in arrays:
        t=teacher(a);h,w=a.shape[:2]
        for _ in range(6):
            y=int(rng.integers(0,(h-160)//4+1))*4;x=int(rng.integers(0,(w-160)//4+1))*4
            X.append(tensor(a[y:y+160,x:x+160],'cpu'));Y.append(tensor(t[y:y+160,x:x+160],'cpu'))
    X=torch.stack(X).cuda();Y=torch.stack(Y).cuda()
    vx=torch.stack([tensor(a[192:352,448:608],'cpu') for a in vals]).cuda()
    vy=torch.stack([tensor(teacher(a)[192:352,448:608],'cpu') for a in vals]).cuda()
    # Discrete geometry seed is kept intact; fit ordinary colour/face readout fields.
    for p in n.parameters():p.requires_grad_(False)
    keys=['rgb_head','up2_low_pw','res2_conv1__pw']
    for k in keys:
        n.weight[k].requires_grad_(True);n.bias[k].requires_grad_(True)
        mask=(n.weight[k].detach()!=0).float();bm=(mask.flatten(1).sum(1)>0).float()
        n.weight[k].register_hook(lambda g,m=mask:g*m);n.bias[k].register_hook(lambda g,m=bm:g*m)
    opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=.0003)
    logs=[];best=float('inf');beststep=0
    for step in range(251):
        if step%25==0:
            with torch.no_grad():score=float((n(vx)-vy).abs().mean()*128)
            logs.append(dict(step=step,validation_mae=score));print('LOCAL_BLOCK_STEP',step,round(score,3),flush=True)
            if score<best:best=score;beststep=step;checkpoint(n,dest/'selected.pt',step,trained=step>0,kind='local square mass/face-cue B1 candidate')
        if step==250:break
        ids=torch.as_tensor(rng.choice(len(X),8,replace=False),device='cuda');opt.zero_grad(set_to_none=True)
        o=n(X[ids]);loss=(o-Y[ids]).abs().mean();loss.backward();opt.step()
    n=load_model(dest/'selected.pt');raw=n.export();(dest/'candidate.bin').write_bytes(raw);engine=IntegerEngine(raw)
    records=[]
    for s in ['outdoor','indoor']:
        a=read(OUT/f'targets/{s}_source.png');t=teacher(a);o=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
        for name,z in [('target',t),('student',o)]:Image.fromarray(z).save(dest/f'{s}_{name}.png')
        panel([a,t,o],['原图','局部方块轮廓目标 / 非B1','局部方块候选 / 纯B1整数'],dest/f'{s}_comparison.jpg')
        crop=(slice(192,384),slice(448,768));panel([a[crop],t[crop],o[crop]],['原图局部','方块轮廓目标局部','学生局部 / 纯B1整数'],dest/f'{s}_detail.png')
        records.append(dict(id=s,target_mae=float(np.abs(o.astype(float)-t).mean())))
    save_json(dest/'report.json',dict(steps=250,selected_step=beststep,logs=logs,independent=records,
        scope='Local square opening/closing silhouette abstraction with shallow directional face cues. Not global semantic3D cuboid remeshing.',
        target_only_ops='CPU binary square closing plus face cue, never inference',
        inference='ordinary existing B1 DW/PW/ReLU/saturation coefficients only',
        material_matching=False,true_MC_3D_achieved=False,coefficient_sha256=sha(dest/'candidate.bin')))
    print('LOCAL_BLOCK_STUDENT_COMPLETE',flush=True)

if __name__=='__main__':main()
