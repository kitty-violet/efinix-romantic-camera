"""Large, clean region masses before fixed-B1 cuboid extrusion.

Opening/closing are coefficients in pre-existing DW/ReLU/saturation layers.
The positive/negative face hierarchy and wide swept side faces remain in B1.
"""
from core import *
from train_pixel import pixel_start,checkpoint
from pixel_targets import panel
from r3_palette_blocks import CHROMA_VALUES,CHROMA_THRESH,geometry
from r4_box_projector import source_fields,BODY
import r9_cuboid_volume as r9
import argparse

CONFIGS={f'{polarity}-{method}':dict(polarity=polarity,morph=ops,method=method) for polarity in ['bright','dark']
    for method,ops in [('opening5','EEDD'),('closing5','DDEE'),('open-close3','EDDE'),('merge3','MMDE')]}

def fields(a,c):
    y,co,cg=source_fields(a);height=y if c['polarity']=='bright' else 255-y
    fs=[];hs=[];ds=[]
    for t in (np.arange(5)+.5)*51:
        m=(height>t).astype(np.uint8)
        for op in c['morph']:
            if op=='E':m=cv2.erode(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
            elif op=='D':m=cv2.dilate(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
            elif op=='M':m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        f=m.astype(np.float32);h=f.copy();d=f.copy()
        for _ in range(7-len(c['morph'])):
            h=np.maximum(h,np.pad(h,((0,0),(1,0)),mode='edge')[:,:-1])
            d=np.maximum(d,np.pad(d,((0,1),(1,0)),mode='edge')[1:,:-1])
        fs.append(f);hs.append(h);ds.append(d)
    return np.mean(fs,0),np.mean(hs,0),np.mean(ds,0),co,cg

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
    # Reuse the signed three-face decoder; replace all low body coefficients.
    n=r9.seed(dict(polarity=c['polarity'],projection=3,preclean=0))
    n.weight['down2__pw'].zero_();n.bias['down2__pw'].zero_()
    sign=1 if c['polarity']=='bright' else -1
    for j,t in enumerate((np.arange(5)+.5)*51/2):
        for branch in range(3):
            row=branch*5+j
            n.weight['down2__pw'][row,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*126*sign
            n.bias['down2__pw'][row]=float(63.5-126*t if sign==1 else 63.5+126*(127.5-t))
    for group,vector in enumerate([[.5,0,-.5],[-.25,.5,-.25]]):
        for j,t in enumerate(CHROMA_THRESH/2):
            row=15+group*8+j;n.weight['down2__pw'][row,:3,0,0]=torch.tensor(vector,device='cuda')*126
            n.bias['down2__pw'][row]=float(63.5-126*t)
    for i,base in enumerate(BODY):
        dw=n.weight[base+'__dw'];pw=n.weight[base+'__pw']
        dw.zero_();n.bias[base+'__dw'].zero_();pw.zero_();n.bias[base+'__pw'].zero_()
        dw[:,0,1,1]=1
        for ch in range(31):pw[ch,ch,0,0]=1
        if i<len(c['morph']):
            op=c['morph'][i];dw[:15].fill_(1)
            if op in ['E','M']:n.bias[base+'__dw'][:15]=(-8 if op=='E' else -4)*127
        else:
            for ch in range(5,15):
                dw[ch].zero_();dw[ch,0,1,1]=1
                dw[ch,0,1 if ch<10 else 2,0]=1
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main(args):
    setup(271062);root=OUT/'r10-mass-cuboids';root.mkdir(exist_ok=True);disk_guard()
    for name in args.names or list(CONFIGS):
        c=CONFIGS[name];d=root/name;d.mkdir(exist_ok=True);n=seed(c);raw=n.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,config=c,kind='mass-first three-face local cuboids')
        engine=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            panel([a,t,o],['原图','大块归并+12像素三面体目标','实际B1整数 / 大块体积'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
        save_json(d/'seed.json',dict(config=c,records=stats,constructed=True,trained=False,
            extrusion_width_pixels=(7-len(c['morph']))*4,coefficient_sha256=sha(d/'initial.bin'),
            inference_extra_operators=False,global_3D_reconstruction=False))
        print('MASS_CUBOID_COMPLETE',name,json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='*');main(p.parse_args())
