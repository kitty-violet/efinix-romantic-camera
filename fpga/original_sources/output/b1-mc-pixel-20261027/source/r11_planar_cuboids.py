"""Merge image-scale colour masses before quantized 12px cuboid extrusion."""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r3_palette_blocks import CHROMA_VALUES,CHROMA_THRESH,geometry
from r4_box_projector import source_fields,BODY
import r9_cuboid_volume as r9
import argparse

CONFIGS={f'{polarity}-{method}':dict(polarity=polarity,morph=ops,method=method,face_gain=3.)
    for polarity in ['bright','dark'] for method,ops in [('open','ED'),('merge','MM'),('close','DE')]}

def fields(a,c):
    y,co,cg=source_fields(a)
    y,co,cg=[cv2.GaussianBlur(cv2.GaussianBlur(z,(3,3),0,borderType=cv2.BORDER_REPLICATE),(3,3),0,borderType=cv2.BORDER_REPLICATE) for z in [y,co,cg]]
    height=y if c['polarity']=='bright' else 255-y;fs=[];hs=[];ds=[]
    for t in (np.arange(5)+.5)*51:
        m=(height>t).astype(np.uint8)
        for op in c['morph']:
            if op=='E':m=cv2.erode(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
            elif op=='D':m=cv2.dilate(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
            else:m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        f=m.astype(np.float32);h=f.copy();d=f.copy()
        for _ in range(3):
            h=np.maximum(h,np.pad(h,((0,0),(1,0)),mode='edge')[:,:-1])
            d=np.maximum(d,np.pad(d,((0,1),(1,0)),mode='edge')[1:,:-1])
        fs.append(f);hs.append(h);ds.append(d)
    return np.mean(fs,0),np.mean(hs,0),np.mean(ds,0),co,cg

def teacher(a,c):
    f,h,d,co,cg=fields(a,c)
    extra=np.clip(c['face_gain']*(d-f),0,1);top=np.clip(c['face_gain']*(d-h),0,1)
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
    sign=1 if c['polarity']=='bright' else -1
    for i,base in enumerate(BODY):
        dw=n.weight[base+'__dw'];pw=n.weight[base+'__pw'];b=n.bias[base+'__pw']
        dw.zero_();n.bias[base+'__dw'].zero_();pw.zero_();b.zero_();dw[:,0,1,1]=1
        if i<2:
            dw[:3,0].copy_(g[None].expand(3,-1,-1))
            if i==0:
                for ch in range(3):pw[ch,ch,0,0]=1
            else:
                for j,t in enumerate((np.arange(5)+.5)*51/2):
                    for branch in range(3):
                        row=branch*5+j;pw[row,0,0,0]=126*sign
                        b[row]=float(63.5-126*t if sign==1 else 63.5+126*(127.5-t))
                for group in range(2):
                    for j,t in enumerate(CHROMA_THRESH/2+63.5):
                        row=15+group*8+j;pw[row,1+group,0,0]=126;b[row]=float(63.5-126*t)
        else:
            for ch in range(31):pw[ch,ch,0,0]=1
            if i<4:
                op=c['morph'][i-2];dw[:15].fill_(1)
                # res1_conv2 PW has no ReLU: its threshold masks are signed.
                # The first DW must account for negative pseudo-mask values.
                if i==2:n.bias[base+'__dw'][:15]=dict(E=-8,M=0,D=8)[op]*127
                elif op in ['E','M']:n.bias[base+'__dw'][:15]=(-8 if op=='E' else -4)*127
            else:
                for ch in range(5,15):
                    dw[ch].zero_();dw[ch,0,1,1]=1;dw[ch,0,1 if ch<10 else 2,0]=1
    n.weight['up2_low_pw'][1].mul_(c['face_gain']);n.weight['up2_low_pw'][2].mul_(c['face_gain'])
    if sign==1:terms=[.9*255,-40,70]
    else:terms=[-.9*255,-35,65]
    n.weight['rgb_head'][:,:3,0,0]=torch.tensor(terms,device='cuda')/127
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main(args):
    setup(271063);root=OUT/'r11-planar-cuboids';root.mkdir(exist_ok=True);disk_guard()
    for name in args.names or list(CONFIGS):
        c=CONFIGS[name];d=root/name;d.mkdir(exist_ok=True);n=seed(c);raw=n.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,config=c,kind='large planar three-face cuboids')
        engine=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            panel([a,t,o],['原图','先归并大色面 / 12像素三面体目标','实际B1整数 / 大块体积'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
        save_json(d/'seed.json',dict(config=c,records=stats,constructed=True,trained=False,
            extrusion_width_pixels=12,coefficient_sha256=sha(d/'initial.bin'),inference_extra_operators=False,global_3D_reconstruction=False))
        print('PLANAR_CUBOID_COMPLETE',name,json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='*');main(p.parse_args())
