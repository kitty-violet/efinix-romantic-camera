"""Five luminance tones with projected contours, natural chroma and broad faces.

Four7-channel luma groups + two chroma + two face fields occupy all32 channels.
DNF decoder reuses14 existing features and fits the existing8-channel output.
"""
from core import *
from train_pixel import pixel_start,checkpoint
from pixel_targets import panel
from r3_palette_blocks import geometry
import r4_box_projector as r4

def teacher(a):
    y,co,cg=r4.source_fields(a);out=np.zeros_like(y);d=4
    for t in (np.arange(4)+.5)*63.75:
        m=(y>t).astype(np.uint8)
        for _ in range(2):m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        neg=1-m
        h=cv2.dilate(m,np.ones((1,9),np.uint8),borderType=cv2.BORDER_REPLICATE)
        v=cv2.dilate(m,np.ones((9,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        dh=cv2.dilate(neg,np.ones((1,9),np.uint8),borderType=cv2.BORDER_REPLICATE)
        dv=cv2.dilate(neg,np.ones((9,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        pad=np.pad(m,4,mode='edge');hh,ww=m.shape
        S=sum(pad[4+dy:4+dy+hh,4+dx:4+dx+ww].astype(np.float32) for dy,dx in [(-4,-4),(-4,4),(4,-4)])/3
        B=(h&v).astype(np.float32);D=(dh&dv).astype(np.float32)
        out+=63.75*(np.maximum(B-S,0)+np.maximum(S-D,0))
    top=np.pad(y,((1,0),(0,0)),mode='edge')[:-1]
    right=np.pad(y,((0,0),(0,1)),mode='edge')[:,1:]
    shade=.15*np.maximum(y-top,0)-.175*np.maximum(y-right,0)
    z=np.stack([out+co-cg+shade,out+cg+shade,out-co-cg+shade],2)
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed():
    n=pixel_start()
    for p in n.parameters():p.zero_()
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    g=torch.tensor(np.outer([1,2,1],[1,2,1])/16,dtype=torch.float32,device='cuda');avg=eye.clone().zero_();avg[1:,1:]=.25
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['stem'][c,c].copy_(g*.5);n.bias['stem'][c]=64;n.weight['down1__pw'][c,c,0,0]=1
    for k in ['down1__dw','down2__dw']:n.weight[k].copy_(avg[None,None].expand_as(n.weight[k]))
    for j,t in enumerate((np.arange(4)+.5)*63.75/2):
        for c in range(7*j,7*j+7):
            sign=-1 if c%7 in [2,3] else 1
            n.weight['down2__pw'][c,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*126*sign
            n.bias['down2__pw'][c]=float(-t*126*sign+(63 if sign<0 else 63.5))
    for c,vector,bias in [(28,[.5,0,-.5],64),(29,[-.25,.5,-.25],64),(30,[.25,.5,.25],0),(31,[.25,.5,.25],0)]:
        n.weight['down2__pw'][c,:3,0,0]=torch.tensor(vector,device='cuda');n.bias['down2__pw'][c]=bias
    for i,base in enumerate(r4.BODY):
        for c in range(32):
            n.weight[base+'__pw'][c,c,0,0]=1
            if c<28:
                if i<2:
                    n.weight[base+'__dw'][c,0].fill_(1/9);n.bias[base+'__dw'][c]=-63.5;n.weight[base+'__pw'][c,c,0,0]=127
                else:
                    kind=c%7;n.weight[base+'__dw'][c,0].zero_()
                    if kind<4:
                        if kind%2==0:n.weight[base+'__dw'][c,0,1,:]=1/3
                        else:n.weight[base+'__dw'][c,0,:,1]=1/3
                        n.weight[base+'__pw'][c,c,0,0]=3
                    else:
                        dy,dx=r4.CORNERS[kind-4];n.weight[base+'__dw'][c,0,dy,dx]=1
        if i==5:
            for j in range(4):
                c=7*j+4;n.weight[base+'__pw'][c].zero_();n.weight[base+'__pw'][c,c:c+3,0,0]=1/3
            n.weight[base+'__dw'][30,0,0,1]=-1;n.weight[base+'__dw'][31,0,1,2]=-1
    # S is now available before decoder; three DNF terms suffice per tone.
    for j in range(4):
        b=j*7;k=j*3;S=b+4
        n.weight['up1_low_pw'][k,[b,b+1],0,0]=1;n.weight['up1_low_pw'][k,S,0,0]=-1;n.bias['up1_low_pw'][k]=-127
        n.weight['up1_low_pw'][k+1,S,0,0]=1;n.weight['up1_low_pw'][k+1,b+2,0,0]=-1
        n.weight['up1_low_pw'][k+2,S,0,0]=1;n.weight['up1_low_pw'][k+2,b+3,0,0]=-1
        n.weight['up2_low_pw'][j,k:k+3,0,0]=1
    for row,source in [(12,28),(13,29),(14,30),(15,31)]:n.weight['up1_low_pw'][row,source,0,0]=1
    for row,source in [(4,12),(5,13),(6,14),(7,15)]:n.weight['up2_low_pw'][row,source,0,0]=1
    n.weight['rgb_head'][:,:4,0,0]=63.75/127
    n.weight['rgb_head'][:,4:6,0,0]=torch.tensor([[2,-2],[0,2],[-2,-2]],device='cuda')
    n.weight['rgb_head'][:,6,0,0]=.30;n.weight['rgb_head'][:,7,0,0]=-.35
    n.bias['rgb_head'].copy_(torch.tensor([-128,-256,128],device='cuda'))
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main():
    setup(271050);d=OUT/'r6-five-tones';d.mkdir(exist_ok=True);disk_guard();n=seed();raw=n.export()
    (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,kind='r6 five luminance tones + continuous chroma')
    engine=IntegerEngine(raw);rows=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');t=teacher(a);o=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(engine(tensor(a)[None]*128),n(tensor(a)[None])*128)
        for k,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{k}.png')
        panel([a,t,o],['原图','5级明暗 方块目标','实际B1系数 / 非全局3D'],d/f'{scene}_comparison.jpg')
        rows.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
    save_json(d/'seed.json',dict(records=rows,constructed=True,trained=False,semantic3D=False,
        coefficient_sha256=sha(d/'initial.bin'),added_inference_operators=False))
    print('R6_SEED_COMPLETE',json.dumps(rows),flush=True)

if __name__=='__main__':main()
