"""Clean large colour masses, rectangular projection, then 16px cuboid faces."""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r3_palette_blocks import CHROMA_VALUES,CHROMA_THRESH,geometry
from r4_box_projector import source_fields,BODY
import r9_cuboid_volume as r9
import argparse

CONFIGS={f'{polarity}-{method}':dict(polarity=polarity,method=method,face_gain=3.)
    for polarity in ['bright','dark'] for method in ['project2','clean-project1']}

def fields(a,c):
    y,co,cg=source_fields(a)
    y,co,cg=[cv2.GaussianBlur(z,(3,3),0,borderType=cv2.BORDER_REPLICATE) for z in [y,co,cg]]
    height=y if c['polarity']=='bright' else 255-y;fs=[];hs=[];ds=[]
    for t in (np.arange(5)+.5)*51:
        m=(height>t).astype(np.uint8);radius=2
        if c['method']=='clean-project1':
            m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8);radius=1
        bh=cv2.dilate(m,np.ones((1,2*radius+1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        bv=cv2.dilate(m,np.ones((2*radius+1,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        f=(bh&bv).astype(np.float32);h=f.copy();d=f.copy()
        for _ in range(4):
            h=np.maximum(h,np.pad(h,((0,0),(1,0)),mode='edge')[:,:-1])
            d=np.maximum(d,np.pad(d,((0,1),(1,0)),mode='edge')[1:,:-1])
        fs.append(f);hs.append(h);ds.append(d)
    return np.mean(fs,0),np.mean(hs,0),np.mean(ds,0),co,cg

def teacher(a,c):
    f,h,d,co,cg=fields(a,c);extra=np.clip(c['face_gain']*(d-f),0,1);top=np.clip(c['face_gain']*(d-h),0,1)
    if c['polarity']=='bright':Y=.9*255*d-40*extra+70*top
    else:Y=20+.9*255*(1-d)-35*extra+65*top
    def chroma(z):
        out=np.full_like(z,CHROMA_VALUES[0])
        for t,step in zip(CHROMA_THRESH,np.diff(CHROMA_VALUES)):out+=step*(z>t)
        return out
    co,cg=chroma(co),chroma(cg);z=np.stack([Y+co-cg,Y+cg,Y-co-cg],2)
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed(c):
    n=r9.seed(dict(polarity=c['polarity'],projection=3,preclean=0))
    g=torch.tensor(np.outer([1,2,1],[1,2,1])/16,device='cuda',dtype=torch.float32)
    n.weight['down2__pw'].zero_();n.bias['down2__pw'].zero_()
    for ch,vector in enumerate([[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]]):
        n.weight['down2__pw'][ch,:3,0,0]=torch.tensor(vector,device='cuda')
        if ch:n.bias['down2__pw'][ch]=63.5
    # First residual PW has ReLU, so threshold masks are nonnegative before
    # the next DW. This avoids signed-mask cancellation in spatial sums.
    base=BODY[0];n.weight[base+'__dw'].zero_();n.bias[base+'__dw'].zero_()
    n.weight[base+'__dw'][:3,0].copy_(g[None].expand(3,-1,-1))
    n.weight[base+'__pw'].zero_();n.bias[base+'__pw'].zero_()
    sign=1 if c['polarity']=='bright' else -1
    for j,t in enumerate((np.arange(5)+.5)*51/2):
        for ch in [2*j,2*j+1]:
            n.weight[base+'__pw'][ch,0,0,0]=126*sign
            n.bias[base+'__pw'][ch]=float(63.5-126*t if sign==1 else 63.5+126*(127.5-t))
    for group in range(2):
        for j,t in enumerate(CHROMA_THRESH/2+63.5):
            ch=10+group*8+j;n.weight[base+'__pw'][ch,1+group,0,0]=126
            n.bias[base+'__pw'][ch]=float(63.5-126*t)
    for i,base in enumerate(BODY[1:3],1):
        dw=n.weight[base+'__dw'];pw=n.weight[base+'__pw'];dw.zero_();n.bias[base+'__dw'].zero_()
        pw.zero_();n.bias[base+'__pw'].zero_();dw[:,0,1,1]=1
        for ch in range(26):pw[ch,ch,0,0]=1
        for ch in range(10):
            dw[ch].zero_()
            if i==1 and c['method']=='clean-project1':dw[ch].fill_(1);n.bias[base+'__dw'][ch]=-4*127
            elif ch%2==0:dw[ch,0,1,:]=1
            else:dw[ch,0,:,1]=1
        if i==2:
            pw.zero_()
            for j in range(5):
                for branch in range(3):
                    row=branch*5+j;pw[row,[2*j,2*j+1],0,0]=4;n.bias[base+'__pw'][row]=-6*127
            for ch in range(16):pw[15+ch,10+ch,0,0]=1
    n.weight['up2_low_pw'][1].mul_(c['face_gain']);n.weight['up2_low_pw'][2].mul_(c['face_gain'])
    terms=[.9*255,-40,70] if sign==1 else [-.9*255,-35,65]
    n.weight['rgb_head'][:,:3,0,0]=torch.tensor(terms,device='cuda')/127
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main(args):
    setup(271064);root=OUT/'r12-solid-cuboids';root.mkdir(exist_ok=True);disk_guard()
    for name in args.names or list(CONFIGS):
        c=CONFIGS[name];d=root/name;d.mkdir(exist_ok=True);n=seed(c);raw=n.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,config=c,kind='large solid cuboid faces')
        engine=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            panel([a,t,o],['原图','整块矩形+16像素三面体目标','实际B1整数 / 大块体积'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
        save_json(d/'seed.json',dict(config=c,records=stats,constructed=True,trained=False,
            extrusion_width_pixels=16,coefficient_sha256=sha(d/'initial.bin'),inference_extra_operators=False,global_3D_reconstruction=False))
        print('SOLID_CUBOID_COMPLETE',name,json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='*');main(p.parse_args())
