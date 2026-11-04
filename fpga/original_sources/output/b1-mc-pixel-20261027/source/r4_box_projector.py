"""Local rectangular silhouette projection, entirely in fixed B1 coefficients.

Horizontal/vertical palette masks + shifted corner-polarity decisions use32
existing channels. Selector, intersection and mux use ordinary PW/ReLU only.
This changes bounded silhouettes; it does not infer a 3D scene or material.
"""
from core import *
from train_pixel import pixel_start,checkpoint
from pixel_targets import panel
from r3_palette_blocks import geometry
import argparse

CV=np.array([-96,-24,0,24,96],np.float32)
CT=(CV[:-1]+CV[1:])/2
BODY=['res1_conv1','res1_conv2','res2_conv1','res2_conv2','res3_conv1','res3_conv2']
CORNERS=[(0,0),(0,2),(2,0),(2,2)]

def source_fields(a):
    z=cv2.GaussianBlur(a.astype(np.float32),(3,3),0,borderType=cv2.BORDER_REPLICATE)
    h,w=a.shape[:2];z=z.reshape(h//4,4,w//4,4,3).mean((1,3));r,g,b=z.transpose(2,0,1)
    return (r+2*g+b)/4,(r-b)/2,(-r+2*g-b)/4

def teacher(a,depth=6,face=.12,preclean=0,chroma_blend=0.,corners=4,soft_polarity=False):
    y,co,cg=source_fields(a);outy=np.zeros_like(y)
    for t in (np.arange(3)+.5)*85:
        mask=(y>t).astype(np.uint8)
        for _ in range(preclean):
            mask=(cv2.boxFilter(mask.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        dark=1-mask
        h=cv2.dilate(mask,np.ones((1,2*depth+1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        v=cv2.dilate(mask,np.ones((2*depth+1,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        dh=cv2.dilate(dark,np.ones((1,2*depth+1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        dv=cv2.dilate(dark,np.ones((2*depth+1,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        pad=np.pad(mask,depth,mode='edge');hh,ww=mask.shape
        votes=sum(pad[depth+dy:depth+dy+hh,depth+dx:depth+dx+ww].astype(np.int16)
                  for dy,dx in [(-depth,-depth),(-depth,depth),(depth,-depth),(depth,depth)][:corners])
        s=(votes/corners).astype(np.float32) if soft_polarity else (votes>=2).astype(np.float32)
        pieces=[np.maximum((h&v).astype(np.float32)-s,0),np.maximum(s-(dh&dv).astype(np.float32),0)]
        for piece in pieces:
            gate=piece.repeat(4,0).repeat(4,1)
            if outy.shape!=gate.shape:outy=np.zeros_like(gate)
            # Match each existing final DW's mask-preserving face response.
            top=np.pad(gate,((1,0),(0,0)),mode='edge')[:-1]
            right=np.pad(gate,((0,0),(0,1)),mode='edge')[:,1:]
            shaded=np.maximum(gate-face*top+face*right-face,0)/(1-face)
            outy+=85*shaded
    def chroma(x):
        q=np.full_like(x,CV[0])
        for t,step in zip(CT,np.diff(CV)):q+=step*(x>t)
        return q
    co,cg=[((1-chroma_blend)*chroma(z)+chroma_blend*z).repeat(4,0).repeat(4,1) for z in [co,cg]]
    z=np.stack([outy+co-cg,outy+cg,outy-co-cg],2)
    return np.clip(z+.5,0,255).astype(np.uint8)

@torch.no_grad()
def seed(depth=6,face=.12,preclean=0,chroma_blend=0.,corners=4,soft_polarity=False):
    assert depth+preclean<=6
    width=4+corners;chroma_start=3*width;raw_start=chroma_start+8
    assert corners in [3,4] and (not chroma_blend or raw_start+2<=32)
    n=pixel_start()
    for p in n.parameters():p.zero_()
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    g=torch.tensor(np.outer([1,2,1],[1,2,1])/16,dtype=torch.float32,device='cuda')
    avg=torch.zeros_like(eye);avg[1:,1:]=.25
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['stem'][c,c].copy_(g*.5);n.bias['stem'][c]=64
    for k in ['down1__dw','down2__dw']:n.weight[k].copy_(avg[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['down1__pw'][c,c,0,0]=1
    for j,t in enumerate((np.arange(3)+.5)*127/3):
        for c in range(j*width,j*width+width):
            sign=-1 if c%width in [2,3] else 1
            n.weight['down2__pw'][c,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*127*sign
            n.bias['down2__pw'][c]=-float(t*127*sign)
    for group,vector in enumerate([[.5,0,-.5],[-.25,.5,-.25]]):
        for j,t in enumerate((CT+127)/2):
            c=chroma_start+group*4+j;n.weight['down2__pw'][c,:3,0,0]=torch.tensor(vector,device='cuda')*127
            n.bias['down2__pw'][c]=float((63.5-t)*127)
        if chroma_blend:
            n.weight['down2__pw'][raw_start+group,:3,0,0]=torch.tensor(vector,device='cuda')
            n.bias['down2__pw'][raw_start+group]=63.5
    for i,base in enumerate(BODY):
        for c in range(32):
            n.weight[base+'__pw'][c,c,0,0]=1
            if i<preclean and c<3*width:
                n.weight[base+'__dw'][c,0].fill_(1/9)
                # Threshold must precede the DW's guaranteed ReLU. Conv2 PW
                # has no ReLU and must never forward negative pseudo-masks.
                n.bias[base+'__dw'][c]=-63.5
                n.weight[base+'__pw'][c,c,0,0]=127
            elif preclean<=i<depth+preclean and c<3*width:
                d=c%width
                if d<4:
                    n.weight[base+'__dw'][c,0].zero_()
                    if d%2==0:n.weight[base+'__dw'][c,0,1,:]=1/3
                    else:n.weight[base+'__dw'][c,0,:,1]=1/3
                    n.weight[base+'__pw'][c,c,0,0]=3
                else:
                    n.weight[base+'__dw'][c,0].zero_();dy,dx=CORNERS[d-4]
                    n.weight[base+'__dw'][c,0,dy,dx]=1
    # B=intersection of projected bright masks; D=same for dark; S=corner majority.
    for j in range(3):
        b=j*width;k=j*3
        n.weight['up1_low_pw'][k,b:b+2,0,0]=4;n.bias['up1_low_pw'][k]=-6*127
        n.weight['up1_low_pw'][k+1,b+2:b+4,0,0]=4;n.bias['up1_low_pw'][k+1]=-6*127
        if soft_polarity:n.weight['up1_low_pw'][k+2,b+4:b+width,0,0]=1/corners
        else:n.weight['up1_low_pw'][k+2,b+4:b+width,0,0]=4;n.bias['up1_low_pw'][k+2]=-6*127
        # Exact binary mux as two existing ReLU hinges.
        n.weight['up2_low_pw'][2*j,k,0,0]=1;n.weight['up2_low_pw'][2*j,k+2,0,0]=-1
        n.weight['up2_low_pw'][2*j+1,k+2,0,0]=1;n.weight['up2_low_pw'][2*j+1,k+1,0,0]=-1
    for group in range(2):
        n.weight['up1_low_pw'][9+group,chroma_start+group*4:chroma_start+4+group*4,0,0]=torch.tensor(np.diff(CV)/(2*127),device='cuda')*(1-chroma_blend)
        n.bias['up1_low_pw'][9+group]=float((CV[0]+127)/2)*(1-chroma_blend)
        if chroma_blend:n.weight['up1_low_pw'][9+group,raw_start+group,0,0]=chroma_blend
        n.weight['up2_low_pw'][6+group,9+group,0,0]=1
    for c in range(6):
        n.weight['up2_high_dw'][c,0,0,1]=-face
        n.weight['up2_high_dw'][c,0,1,2]=face
        n.bias['up2_high_dw'][c]=-face*127
    n.weight['rgb_head'][:,:6,0,0]=85/(127*(1-face))
    n.weight['rgb_head'][:,6:,0,0]=torch.tensor([[2,-2],[0,2],[-2,-2]],device='cuda')
    n.bias['rgb_head'].copy_(torch.tensor([-128,-255,126],device='cuda'))
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def proof(engine,folder):
    rows=[];stats=[];yy,xx=np.mgrid[:128,:128]
    for j,(bg,fg,rx,ry) in enumerate([(25,225,20,20),(225,25,20,20),(25,225,16,24),(225,25,24,16)]):
        mask=((xx-64)/rx)**2+((yy-64)/ry)**2<1
        a=np.where(mask[...,None],np.full((128,128,3),fg),np.full((128,128,3),bg)).astype(np.uint8)
        z=rgb(engine(tensor(a)[None]*128)/128);t=teacher(a)
        pred=z.mean(2)>128 if fg>bg else z.mean(2)<128
        want=(xx>=64-rx)&(xx<64+rx)&(yy>=64-ry)&(yy<64+ry)
        iou=float((pred&want).sum()/max(1,(pred|want).sum()))
        stats.append(dict(id=j,foreground=fg,background=bg,radii=[rx,ry],rectangle_iou=iou))
        panel([a,t,z],['圆润输入','局部方块目标','实际B1系数输出'],folder/f'proof_{j:02d}.png',width=384)
    return stats

def main(args):
    setup(271041);dest=OUT/'r4-box-projector';dest.mkdir(exist_ok=True);disk_guard()
    for depth in args.depths:
        name=f'box-r{depth*4}'+(f'-clean{args.preclean}' if args.preclean else '')+(f'-mix{int(args.chroma_blend*100)}c{args.corners}' if args.chroma_blend or args.corners!=4 else '')+('-soft' if args.soft_polarity else '')
        d=dest/name;d.mkdir(exist_ok=True);net=seed(depth,args.face,args.preclean,args.chroma_blend,args.corners,args.soft_polarity);raw=net.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(net,d/'initial.pt',0,trained=False,depth=depth,face=args.face,preclean=args.preclean,chroma_blend=args.chroma_blend,corners=args.corners,soft_polarity=args.soft_polarity)
        eng=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,depth,args.face,args.preclean,args.chroma_blend,args.corners,args.soft_polarity);o=rgb(eng(tensor(a)[None]*128)/128)
            assert torch.equal(eng(tensor(a)[None]*128),net(tensor(a)[None])*128)
            for kind,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{kind}.png')
            panel([a,t,o],['原图',f'方块投影 r{depth*4}目标',f'方块投影 r{depth*4} 纯B1整数'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(t.astype(float)-o).mean()),geometry=geometry(o)))
        tests=proof(eng,d) if depth==6 else []
        save_json(d/'seed.json',dict(depth=depth,face=args.face,preclean=args.preclean,chroma_blend=args.chroma_blend,corners=args.corners,soft_polarity=args.soft_polarity,constructed=True,trained=False,records=stats,
            geometric_proofs=tests,coefficient_sha256=sha(d/'initial.bin'),inference_extra_operators=False,
            native_grid=4,semantic3D=False))
        print('BOX_SEED_DONE',depth,json.dumps(stats),'PROOF',json.dumps(tests),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depths',type=int,nargs='*',default=[2,4,6]);p.add_argument('--face',type=float,default=.12);p.add_argument('--preclean',type=int,default=0);p.add_argument('--chroma-blend',type=float,default=0.);p.add_argument('--corners',type=int,default=4);p.add_argument('--soft-polarity',action='store_true')
    main(p.parse_args())
