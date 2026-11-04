"""Six-tone boxes with a shared continuous background and tri-state decoder.

The32-channel allocation is20 mask branches,3 background probes,2 chroma,2
source-face fields. B/D intersections encode a tri-state signal per tone.
One existing PW saturation decodes each tone with a continuous confidence.
"""
from core import *
from train_pixel import pixel_start,checkpoint
from pixel_targets import panel
from r3_palette_blocks import geometry
import r4_box_projector as r4

FACE_SPECS=[(j,direction,dark) for j in range(5) for dark in [False,True] for direction in ['top','right']
            if not (j==0 and not dark and direction=='top')]
assert len(FACE_SPECS)==19

def teacher(a):
    y,co,cg=r4.source_fields(a);shape=y.shape;out=np.zeros_like(y);topface=np.zeros_like(y);rightface=np.zeros_like(y)
    bg=y/2
    for _ in range(2):bg=cv2.boxFilter(bg,-1,(3,3),normalize=True,borderType=cv2.BORDER_REPLICATE)
    pad=np.pad(bg,3,mode='edge');hh,ww=shape
    back=sum(pad[3+dy:3+dy+hh,3+dx:3+dx+ww] for dy,dx in [(-3,-3),(-3,3),(3,-3)])/3
    boxes=[]
    for j,t in enumerate((np.arange(5)+.5)*51):
        m=(y>t).astype(np.uint8)
        for _ in range(2):m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        n=1-m
        B=(cv2.dilate(m,np.ones((1,7),np.uint8),borderType=cv2.BORDER_REPLICATE)&cv2.dilate(m,np.ones((7,1),np.uint8),borderType=cv2.BORDER_REPLICATE)).astype(np.float32)
        D=(cv2.dilate(n,np.ones((1,7),np.uint8),borderType=cv2.BORDER_REPLICATE)&cv2.dilate(n,np.ones((7,1),np.uint8),borderType=cv2.BORDER_REPLICATE)).astype(np.float32)
        confidence=np.clip(.5-6*(back-t/2)/127,0,1)
        gate=np.where(B>D,1,np.where(B<D,0,confidence))
        out+=51*gate;boxes.append((B,D))
    for j,direction,dark in FACE_SPECS:
        z=boxes[j][int(dark)]
        old=np.pad(z,((1,0),(0,0)),mode='edge')[:-1] if direction=='top' else np.pad(z,((0,0),(0,1)),mode='edge')[:,1:]
        edge=np.maximum(z-old,0)
        if direction=='top':topface+=edge/9
        else:rightface+=edge/10
    shade=127*(.35*topface-.4*rightface)
    z=np.stack([out+co-cg+shade,out+cg+shade,out-co-cg+shade],2)
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed():
    n=pixel_start()
    for p in n.parameters():p.zero_()
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    g=torch.tensor(np.outer([1,2,1],[1,2,1])/16,device='cuda',dtype=torch.float32);avg=eye.clone().zero_();avg[1:,1:]=.25
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['stem'][c,c].copy_(g*.5);n.bias['stem'][c]=64;n.weight['down1__pw'][c,c,0,0]=1
    for k in ['down1__dw','down2__dw']:n.weight[k].copy_(avg[None,None].expand_as(n.weight[k]))
    for j,t in enumerate((np.arange(5)+.5)*51/2):
        for c in range(4*j,4*j+4):
            sign=-1 if c%4 in [2,3] else 1
            n.weight['down2__pw'][c,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*126*sign
            n.bias['down2__pw'][c]=float(-126*t*sign+(63 if sign<0 else 63.5))
    for c in [20,21,22]:n.weight['down2__pw'][c,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')
    n.weight['down2__pw'][23,:3,0,0]=torch.tensor([.5,0,-.5],device='cuda');n.bias['down2__pw'][23]=64
    n.weight['down2__pw'][24,:3,0,0]=torch.tensor([-.25,.5,-.25],device='cuda');n.bias['down2__pw'][24]=64
    for i,base in enumerate(r4.BODY[:5]):
        for c in range(25):
            n.weight[base+'__pw'][c,c,0,0]=1
            if c<20:
                if i<2:
                    n.weight[base+'__dw'][c,0].fill_(1/9);n.bias[base+'__dw'][c]=-63.5;n.weight[base+'__pw'][c,c,0,0]=127
                else:
                    n.weight[base+'__dw'][c,0].zero_()
                    if c%2==0:n.weight[base+'__dw'][c,0,1,:]=1/3
                    else:n.weight[base+'__dw'][c,0,:,1]=1/3
                    n.weight[base+'__pw'][c,c,0,0]=3
            elif c<23:
                if i<2:n.weight[base+'__dw'][c,0].fill_(1/9)
                else:
                    n.weight[base+'__dw'][c,0].zero_();dy,dx=r4.CORNERS[c-20];n.weight[base+'__dw'][c,0,dy,dx]=1
    # res3_conv2 computes signed intersections; up1 DW supplies rectification.
    w=n.weight['res3_conv2__pw'];b=n.bias['res3_conv2__pw']
    for j in range(5):
        for row,cols in [(2*j,[4*j,4*j+1]),(2*j+1,[4*j+2,4*j+3])]:w[row,cols,0,0]=4;b[row]=-6*127
    w[10,20:23,0,0]=1/3;w[11,23,0,0]=1;w[12,24,0,0]=1
    for i,(j,direction,dark) in enumerate(FACE_SPECS):
        c=13+i;source=4*j+(2 if dark else 0)
        w[c,[source,source+1],0,0]=4;b[c]=-6*127
        dy,dx=(0,1) if direction=='top' else (1,2)
        n.weight['up1_low_dw'][c,0,dy,dx]=-1
    # T encodes B-D as1 /64 /127. A large margin keeps unambiguous states fixed.
    for j,t in enumerate((np.arange(5)+.5)*51/2):
        n.weight['up1_low_pw'][j,2*j,0,0]=.5;n.weight['up1_low_pw'][j,2*j+1,0,0]=-.5;n.bias['up1_low_pw'][j]=64
        n.weight['up2_low_pw'][j,j,0,0]=12;n.weight['up2_low_pw'][j,5,0,0]=-6
        n.bias['up2_low_pw'][j]=float(6*t+63.5-12*64)
    for row,source in [(5,10),(6,11),(7,12)]:n.weight['up1_low_pw'][row,source,0,0]=1
    for i,(_,direction,_) in enumerate(FACE_SPECS):n.weight['up1_low_pw'][8 if direction=='top' else 9,13+i,0,0]=1/(9 if direction=='top' else 10)
    n.weight['up2_low_pw'][5,6,0,0]=1;n.weight['up2_low_pw'][6,7,0,0]=1
    n.weight['up2_low_pw'][7,8,0,0]=.35;n.weight['up2_low_pw'][7,9,0,0]=-.4;n.bias['up2_low_pw'][7]=64
    n.weight['rgb_head'][:,:5,0,0]=51/127;n.weight['rgb_head'][:,5:7,0,0]=torch.tensor([[2,-2],[0,2],[-2,-2]],device='cuda')
    n.weight['rgb_head'][:,7,0,0]=1;n.bias['rgb_head'].copy_(torch.tensor([-192,-320,64],device='cuda'))
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main():
    setup(271051);d=OUT/'r7-shared-background';d.mkdir(exist_ok=True);disk_guard();n=seed();raw=n.export()
    (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,kind='r7 shared continuous background tri-state decoder')
    engine=IntegerEngine(raw);stats=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');t=teacher(a);o=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(engine(tensor(a)[None]*128),n(tensor(a)[None])*128)
        for k,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{k}.png')
        panel([a,t,o],['原图','6级明暗 连续背景 方块目标','实际B1系数 / 非全局3D'],d/f'{scene}_comparison.jpg')
        stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
    save_json(d/'seed.json',dict(records=stats,constructed=True,trained=False,semantic3D=False,
        coefficient_sha256=sha(d/'initial.bin'),added_inference_operators=False))
    print('R7_SEED_COMPLETE',json.dumps(stats),flush=True)

if __name__=='__main__':main()
