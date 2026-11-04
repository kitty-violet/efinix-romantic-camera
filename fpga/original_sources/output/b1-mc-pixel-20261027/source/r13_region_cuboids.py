"""Neighbourhood consensus for both colour and shape, then shared cuboid faces."""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r3_palette_blocks import CHROMA_VALUES,CHROMA_THRESH,geometry
from r4_box_projector import source_fields,BODY
import r10_mass_cuboids as r10
import argparse

CONFIGS={f'{polarity}-{method}':dict(polarity=polarity,morph=ops,method=method,face_gain=3.)
    for polarity in ['bright','dark'] for method,ops in [('aligned-close','HVDE'),('aligned-open','HVED'),('vote-close','MMDE')]}

def consensus(m,ops):
    for op in ops:
        if op=='H':m=(cv2.boxFilter(m.astype(np.float32),-1,(3,1),normalize=False,borderType=cv2.BORDER_REPLICATE)>=2).astype(np.uint8)
        elif op=='V':m=(cv2.boxFilter(m.astype(np.float32),-1,(1,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=2).astype(np.uint8)
        elif op=='M':m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        elif op=='D':m=cv2.dilate(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
        elif op=='E':m=cv2.erode(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
    return m

def fields(a,c):
    y,co,cg=source_fields(a);height=y if c['polarity']=='bright' else 255-y;fs=[];hs=[];ds=[]
    for t in (np.arange(5)+.5)*51:
        f=consensus((height>t).astype(np.uint8),c['morph']).astype(np.float32);h=f.copy();d=f.copy()
        for _ in range(3):
            h=np.maximum(h,np.pad(h,((0,0),(1,0)),mode='edge')[:,:-1])
            d=np.maximum(d,np.pad(d,((0,1),(1,0)),mode='edge')[1:,:-1])
        fs.append(f);hs.append(h);ds.append(d)
    def colour(z):
        out=np.full_like(z,CHROMA_VALUES[0])
        for t,step in zip(CHROMA_THRESH,np.diff(CHROMA_VALUES)):
            out+=step*consensus((z>t).astype(np.uint8),c['morph'])
        return out
    return np.mean(fs,0),np.mean(hs,0),np.mean(ds,0),colour(co),colour(cg)

def teacher(a,c):
    f,h,d,co,cg=fields(a,c);e=np.clip(c['face_gain']*(d-f),0,1);t=np.clip(c['face_gain']*(d-h),0,1)
    Y=.9*255*d-40*e+70*t if c['polarity']=='bright' else 20+.9*255*(1-d)-35*e+65*t
    z=np.stack([Y+co-cg,Y+cg,Y-co-cg],2)
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed(c):
    n=r10.seed(c)
    for i,base in enumerate(BODY[:4]):
        dw=n.weight[base+'__dw'];b=n.bias[base+'__dw'];dw.zero_();dw[:,0,1,1]=1;b.zero_()
        op=c['morph'][i]
        for ch in range(31):
            dw[ch].zero_()
            if op=='H':dw[ch,0,1,:]=1;b[ch]=-127
            elif op=='V':dw[ch,0,:,1]=1;b[ch]=-127
            else:
                dw[ch].fill_(1)
                if op in ['M','E']:b[ch]=(-4 if op=='M' else -8)*127
    n.weight['up2_low_pw'][1].mul_(c['face_gain']);n.weight['up2_low_pw'][2].mul_(c['face_gain'])
    terms=[.9*255,-40,70] if c['polarity']=='bright' else [-.9*255,-35,65]
    n.weight['rgb_head'][:,:3,0,0]=torch.tensor(terms,device='cuda')/127
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def region_stats(a):
    low=a[::4,::4];q=(low.astype(np.int32)+12)//24;codes=q[...,0]+12*q[...,1]+144*q[...,2]
    areas=[]
    for k in np.unique(codes):
        _,_,s,_=cv2.connectedComponentsWithStats((codes==k).astype(np.uint8),connectivity=4)
        areas.extend(s[1:,cv2.CC_STAT_AREA].tolist())
    v=np.asarray(areas)
    return dict(component_count=len(areas),median_component_area_native_pixels=float(np.median(v)*16),
        tiny_component_count_fraction=float((v<=2).mean()),tiny_component_area_fraction=float(v[v<=2].sum()/v.sum()),
        components_over_256_pixels=int((v>=16).sum()))

def main(args):
    setup(271073);root=OUT/'r13-region-cuboids';root.mkdir(exist_ok=True);disk_guard()
    for name in args.names or list(CONFIGS):
        c=CONFIGS[name];d=root/name;d.mkdir(exist_ok=True);n=seed(c);raw=n.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,config=c,kind='connected shared-boundary cuboid regions')
        engine=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            panel([a,t,o],['原图','颜色+形状共同归并 / 三面体目标','实际B1整数 / 邻域连接'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o),regions=region_stats(o)))
        save_json(d/'seed.json',dict(config=c,records=stats,constructed=True,trained=False,
            extrusion_width_pixels=12,coefficient_sha256=sha(d/'initial.bin'),inference_extra_operators=False,global_3D_reconstruction=False))
        print('REGION_CUBOID_COMPLETE',name,json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='*');main(p.parse_args())
