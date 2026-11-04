"""Axis-projected shapes with a 3-face swept volume, in the fixed B1 graph.

All geometry and face shading are ordinary DW/PW coefficients. CPU operations
below are supervision/measurement only. This is an image-derived local cuboid
relief, without claiming scene reconstruction or material replacement.
"""
from core import *
from train_pixel import pixel_start,checkpoint
from pixel_targets import panel
from r3_palette_blocks import CHROMA_VALUES,CHROMA_THRESH,geometry
from r4_box_projector import source_fields,BODY
import argparse

CONFIGS={
    'bright-large16':dict(polarity='bright',projection=3,preclean=0),
    'dark-large16':dict(polarity='dark',projection=3,preclean=0),
    'bright-clean16':dict(polarity='bright',projection=2,preclean=1),
    'dark-clean16':dict(polarity='dark',projection=2,preclean=1),
    'bright-square8':dict(polarity='bright',projection=5,preclean=0),
    'dark-square8':dict(polarity='dark',projection=5,preclean=0),
}

def fields(a,c):
    y,co,cg=source_fields(a);height=y if c['polarity']=='bright' else 255-y
    fronts=[];horiz=[];diags=[];radius=c['projection'];start=radius+c['preclean'];sweeps=7-start
    for t in (np.arange(5)+.5)*51:
        m=(height>t).astype(np.uint8)
        for _ in range(c['preclean']):m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        bh=cv2.dilate(m,np.ones((1,2*radius+1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        bv=cv2.dilate(m,np.ones((2*radius+1,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        f=(bh&bv).astype(np.float32);h=f.copy();d=f.copy()
        for _ in range(sweeps):
            oldh=np.pad(h,((0,0),(1,0)),mode='edge')[:,:-1]
            oldd=np.pad(d,((0,1),(1,0)),mode='edge')[1:,:-1]
            h=np.maximum(h,oldh);d=np.maximum(d,oldd)
        fronts.append(f);horiz.append(h);diags.append(d)
    return np.mean(fronts,0),np.mean(horiz,0),np.mean(diags,0),co,cg

def teacher(a,c):
    f,h,d,co,cg=fields(a,c);extra=np.maximum(d-f,0);top=np.maximum(d-h,0)
    if c['polarity']=='bright':Y=255*(.9*d-.30*extra+.40*top)
    else:Y=20+255*(.9*(1-d)-.10*extra+.32*top)
    def chroma(z):
        out=np.full_like(z,CHROMA_VALUES[0])
        for t,step in zip(CHROMA_THRESH,np.diff(CHROMA_VALUES)):out+=step*(z>t)
        return out
    co,cg=chroma(co),chroma(cg)
    z=np.stack([Y+co-cg,Y+cg,Y-co-cg],2)
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed(c):
    assert c['projection']+c['preclean'] in [3,5]
    n=pixel_start()
    for p in n.parameters():p.zero_()
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    g=torch.tensor(np.outer([1,2,1],[1,2,1])/16,device='cuda',dtype=torch.float32)
    avg=eye.clone().zero_();avg[1:,1:]=.25
    for r in n.plan['nodes']:
        key=r['id'].replace('.','__')
        if key in n.weight and r['kind']=='dw3x3':n.weight[key].copy_(eye[None,None].expand_as(n.weight[key]))
    for j in range(3):n.weight['stem'][j,j].copy_(g*.5);n.bias['stem'][j]=64;n.weight['down1__pw'][j,j,0,0]=1
    for key in ['down1__dw','down2__dw']:n.weight[key].copy_(avg[None,None].expand_as(n.weight[key]))
    sign=1 if c['polarity']=='bright' else -1
    for j,t in enumerate((np.arange(5)+.5)*51/2):
        for channel in [2*j,2*j+1]:
            n.weight['down2__pw'][channel,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*126*sign
            n.bias['down2__pw'][channel]=float(63.5-126*t if sign==1 else 63.5+126*(127.5-t))
    for group,vector in enumerate([[.5,0,-.5],[-.25,.5,-.25]]):
        for j,t in enumerate(CHROMA_THRESH/2):
            channel=10+group*8+j
            n.weight['down2__pw'][channel,:3,0,0]=torch.tensor(vector,device='cuda')*126
            n.bias['down2__pw'][channel]=float(63.5-126*t)
    start=c['projection']+c['preclean']
    for i,base in enumerate(BODY):
        w=n.weight[base+'__pw'];b=n.bias[base+'__pw']
        if i<start:
            for ch in range(26):w[ch,ch,0,0]=1
            for ch in range(10):
                kernel=n.weight[base+'__dw'][ch,0];kernel.zero_()
                if i<c['preclean']:
                    kernel.fill_(1);n.bias[base+'__dw'][ch]=-4*127
                elif ch%2==0:kernel[1,:]=1
                else:kernel[:,1]=1
            if i==start-1:
                w.zero_();b.zero_()
                for j in range(5):
                    for branch in range(3):
                        row=branch*5+j;w[row,[2*j,2*j+1],0,0]=4;b[row]=-6*127
                for ch in range(16):w[15+ch,10+ch,0,0]=1
        else:
            for ch in range(31):w[ch,ch,0,0]=1
            for ch in range(5,15):
                kernel=n.weight[base+'__dw'][ch,0];kernel.zero_();kernel[1,1]=1
                kernel[1,0] = 1 if ch<10 else 0
                if ch>=10:kernel[2,0]=1
    # Final low-resolution sweep gives12/16px-wide actual volume faces.
    for ch in range(5,15):
        kernel=n.weight['up1_low_dw'][ch,0];kernel.zero_();kernel[1,1]=1
        if ch<10:kernel[1,0]=1
        else:kernel[2,0]=1
    for branch in range(3):n.weight['up1_low_pw'][branch,branch*5:(branch+1)*5,0,0]=.2
    for group in range(2):
        n.weight['up1_low_pw'][3+group,15+group*8:23+group*8,0,0]=torch.tensor(np.diff(CHROMA_VALUES)/(2*127),device='cuda')
        n.bias['up1_low_pw'][3+group]=float((CHROMA_VALUES[0]+127)/2)
    n.weight['up2_low_pw'][0,2,0,0]=1
    n.weight['up2_low_pw'][1,2,0,0]=1;n.weight['up2_low_pw'][1,0,0,0]=-1
    n.weight['up2_low_pw'][2,2,0,0]=1;n.weight['up2_low_pw'][2,1,0,0]=-1
    for ch in [3,4]:n.weight['up2_low_pw'][ch,ch,0,0]=1
    if sign==1:
        n.weight['rgb_head'][:,:3,0,0]=torch.tensor([.90,-.30,.40],device='cuda')*255/127
        offset=0.
    else:
        n.weight['rgb_head'][:,:3,0,0]=torch.tensor([-.90,-.10,.32],device='cuda')*255/127
        offset=20+.9*255
    n.weight['rgb_head'][:,3:5,0,0]=torch.tensor([[2,-2],[0,2],[-2,-2]],device='cuda')
    n.bias['rgb_head'].copy_(torch.tensor([-128+offset,-255+offset,126+offset],device='cuda'))
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def proofs(engine,c,dest):
    yy,xx=np.mgrid[:192,:224];stats=[];bright=c['polarity']=='bright';bg=20 if bright else 235;fg=200 if bright else 55
    for j,mask in enumerate([(xx>=80)&(xx<144)&(yy>=72)&(yy<136),((xx-112)/12)**2+((yy-104)/12)**2<1]):
        a=np.where(mask[...,None],fg,bg).astype(np.uint8).repeat(3,2)
        o=rgb(engine(tensor(a)[None]*128)/128);t=teacher(a,c)
        panel([a,t,o],['输入','有宽顶面/侧面的方块目标','实际固定B1整数输出'],dest/f'proof_{j}.png',width=448)
        stats.append(dict(id=j,target_mae=float(np.abs(t.astype(float)-o).mean())))
    return stats

def main(args):
    setup(271061);root=OUT/'r9-cuboid-volume';root.mkdir(exist_ok=True);disk_guard()
    for name in args.names or list(CONFIGS):
        c=CONFIGS[name];d=root/name;d.mkdir(exist_ok=True);n=seed(c);raw=n.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,config=c,kind='three-face local cuboid volume')
        engine=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            panel([a,t,o],['原图','大块三面体 / 目标','实际B1系数 / 局部方块体积'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
        save_json(d/'seed.json',dict(config=c,records=stats,proofs=proofs(engine,c,d),constructed=True,trained=False,
            footprint_native_radius=c['projection']*4,extrusion_width_pixels=(7-c['projection']-c['preclean'])*4,
            coefficient_sha256=sha(d/'initial.bin'),added_operators=False,global_3D_reconstruction=False))
        print('CUBOID_SEED_COMPLETE',name,json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='*');main(p.parse_args())
