"""Bound local box changes with an internal source-luma path, ordinary B1 only.

This ablation limits the contrast changed by a threshold transition. It neither
uses output blending outside B1 nor changes the operator graph. Not semantic3D.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r3_palette_blocks import geometry
import r4_box_projector as r4
import r7_shared_background as r7

FACE_SPECS=r7.FACE_SPECS[:-1]
assert len(FACE_SPECS)==18

def teacher(a,mix):
    y,co,cg=r4.source_fields(a);shape=y.shape;out=np.zeros_like(y)
    bg=y/2
    for _ in range(2):bg=cv2.boxFilter(bg,-1,(3,3),normalize=True,borderType=cv2.BORDER_REPLICATE)
    pad=np.pad(bg,3,mode='edge');hh,ww=shape
    back=sum(pad[3+dy:3+dy+hh,3+dx:3+dx+ww] for dy,dx in [(-3,-3),(-3,3),(3,-3)])/3
    boxes=[]
    for t in (np.arange(5)+.5)*51:
        m=(y>t).astype(np.uint8)
        for _ in range(2):m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        n=1-m
        B=(cv2.dilate(m,np.ones((1,7),np.uint8),borderType=cv2.BORDER_REPLICATE)&cv2.dilate(m,np.ones((7,1),np.uint8),borderType=cv2.BORDER_REPLICATE)).astype(np.float32)
        D=(cv2.dilate(n,np.ones((1,7),np.uint8),borderType=cv2.BORDER_REPLICATE)&cv2.dilate(n,np.ones((7,1),np.uint8),borderType=cv2.BORDER_REPLICATE)).astype(np.float32)
        confidence=np.clip(.5-6*(back-t/2)/127,0,1)
        out+=51*np.where(B>D,1,np.where(B<D,0,confidence));boxes.append((B,D))
    topface=np.zeros_like(y);rightface=np.zeros_like(y)
    for j,direction,dark in FACE_SPECS:
        z=boxes[j][int(dark)]
        old=np.pad(z,((1,0),(0,0)),mode='edge')[:-1] if direction=='top' else np.pad(z,((0,0),(0,1)),mode='edge')[:,1:]
        edge=np.maximum(z-old,0)/9
        if direction=='top':topface+=edge
        else:rightface+=edge
    shade=127*(.35*topface-.4*rightface)
    Y=mix*y+(1-mix)*out
    z=np.stack([Y+co-cg+shade,Y+cg+shade,Y-co-cg+shade],2)
    return np.clip(z+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed(mix):
    n=r7.seed()
    n.weight['down2__pw'][25,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')
    for base in r4.BODY[:5]:n.weight[base+'__pw'][25,25,0,0]=1
    n.weight['res3_conv2__pw'][31].zero_();n.bias['res3_conv2__pw'][31]=0
    n.weight['res3_conv2__pw'][31,25,0,0]=1
    n.weight['up1_low_dw'][31].zero_();n.weight['up1_low_dw'][31,0,1,1]=1
    n.weight['up1_low_pw'][8:10,31,0,0]=0
    for i,(_,direction,_) in enumerate(FACE_SPECS):n.weight['up1_low_pw'][8 if direction=='top' else 9,13+i,0,0]=1/9
    n.weight['up1_low_pw'][10,31,0,0]=1
    n.weight['up2_low_pw'][7].zero_();n.weight['up2_low_pw'][7,8,0,0]=.175
    n.weight['up2_low_pw'][7,9,0,0]=-.2;n.weight['up2_low_pw'][7,10,0,0]=mix
    n.bias['up2_low_pw'][7]=64*(1-mix)
    n.weight['rgb_head'][:,:5,0,0]=(1-mix)*51/127
    n.weight['rgb_head'][:,7,0,0]=2
    n.bias['rgb_head'].copy_(torch.tensor([-256+128*mix,-384+128*mix,128*mix],device='cuda'))
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main():
    setup(271052);root=OUT/'r8-bounded-projection';root.mkdir(exist_ok=True);disk_guard()
    for mix in [.25,.5,.75]:
        d=root/f'source-{int(mix*100):02d}';d.mkdir(exist_ok=True);n=seed(mix);raw=n.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,kind='bounded local projection',mix=mix)
        engine=IntegerEngine(raw);stats=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,mix);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(engine(tensor(a)[None]*128),n(tensor(a)[None])*128)
            for k,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{k}.png')
            panel([a,t,o],['原图',f'局部方块+原明暗{mix:.0%}目标','实际固定B1 / 非全局3D'],d/f'{scene}_comparison.jpg')
            stats.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
        save_json(d/'seed.json',dict(records=stats,mix=mix,constructed=True,trained=False,semantic3D=False,
            coefficient_sha256=sha(d/'initial.bin'),added_inference_operators=False))
        print('R8_SEED_COMPLETE',mix,json.dumps(stats),flush=True)

if __name__=='__main__':main()
