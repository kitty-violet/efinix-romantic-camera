"""Projector with continuous polarity and four-pixel-wide directional faces.

Geometry decisions move into res3; decoder channels then carry explicit face
fields. No extra nodes, material, postfilter, or recurrent state is introduced.
"""
from core import *
from pixel_targets import panel
from train_pixel import checkpoint
from r3_palette_blocks import geometry
import r4_box_projector as r4

def teacher(a):
    y,co,cg=r4.source_fields(a);Y=np.zeros_like(y);light=np.zeros_like(y);darken=np.zeros_like(y)
    d=3
    for t in (np.arange(3)+.5)*85:
        m=(y>t).astype(np.uint8)
        m=(cv2.boxFilter(m.astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)>=5).astype(np.uint8)
        q=1-m
        h=cv2.dilate(m,np.ones((1,7),np.uint8),borderType=cv2.BORDER_REPLICATE)
        v=cv2.dilate(m,np.ones((7,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        dh=cv2.dilate(q,np.ones((1,7),np.uint8),borderType=cv2.BORDER_REPLICATE)
        dv=cv2.dilate(q,np.ones((7,1),np.uint8),borderType=cv2.BORDER_REPLICATE)
        B=(h&v).astype(np.float32);D=(dh&dv).astype(np.float32)
        p=np.pad(m,3,mode='edge');hh,ww=m.shape
        S=sum(p[3+dy:3+dy+hh,3+dx:3+dx+ww].astype(np.float32) for dy,dx in [(-3,-3),(-3,3),(3,-3)])/3
        gate=np.maximum(B-S,0)+np.maximum(S-D,0);Y+=85*gate
        for field,polarity in [(B,False),(D,True)]:
            top=np.pad(field,((1,0),(0,0)),mode='edge')[:-1]
            right=np.pad(field,((0,0),(0,1)),mode='edge')[:,1:]
            et=np.maximum(field-top,0);er=np.maximum(field-right,0)
            lt=np.maximum(et+S-1,0) if polarity else np.maximum(et-S,0)
            rt=np.maximum(er+S-1,0) if polarity else np.maximum(er-S,0)
            light+=lt/3;darken+=rt/3
    def chroma(z):
        out=np.full_like(z,r4.CV[0])
        for t,step in zip(r4.CT,np.diff(r4.CV)):out+=step*(z>t)
        return .4*out+.6*z
    co,cg=chroma(co),chroma(cg);shade=127*(.28*light-.33*darken)
    o=np.stack([Y+co-cg+shade,Y+cg+shade,Y-co-cg+shade],2)
    return np.clip(o+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed():
    n=r4.seed(depth=3,face=0,preclean=1,chroma_blend=.6,corners=3,soft_polarity=True)
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    # First four body layers retain preclean1 + projection3. Use two remaining
    # existing residual convolutions for region decisions and masked face cues.
    for base in ['res3_conv1','res3_conv2']:
        n.weight[base+'__dw'].copy_(eye[None,None].expand_as(n.weight[base+'__dw']))
        n.bias[base+'__dw'].zero_();n.weight[base+'__pw'].zero_();n.bias[base+'__pw'].zero_()
    w=n.weight['res3_conv1__pw'];b=n.bias['res3_conv1__pw']
    for j in range(3):
        source=j*7;k=j*3
        for row,terms in [(k,[source,source+1]),(k+1,[source+2,source+3])]:
            w[row,terms,0,0]=4;b[row]=-6*127
        w[k+2,source+4:source+7,0,0]=1/3
        # Duplicate boxes supply four directional face decisions per threshold.
        for row,terms in [(9+j,[source,source+1]),(12+j,[source,source+1]),
                          (15+j,[source+2,source+3]),(18+j,[source+2,source+3])]:
            w[row,terms,0,0]=4;b[row]=-6*127
    for group in range(2):
        w[21+group,21+group*4:25+group*4,0,0]=torch.tensor(np.diff(r4.CV)/(2*127)*.4,device='cuda')
        w[21+group,29+group,0,0]=.6;b[21+group]=float((r4.CV[0]+127)/2)*.4
    for j in range(3):
        for c,dy,dx in [(9+j,0,1),(12+j,1,2),(15+j,0,1),(18+j,1,2)]:
            n.weight['res3_conv2__dw'][c,0,dy,dx]=-1
    w=n.weight['res3_conv2__pw'];b=n.bias['res3_conv2__pw']
    for c in [*range(9),21,22]:w[c,c,0,0]=1
    for j in range(3):
        S=j*3+2
        for row,edge,polarity in [(9+4*j,9+j,False),(10+4*j,12+j,False),
                                  (11+4*j,15+j,True),(12+4*j,18+j,True)]:
            w[row,edge,0,0]=1;w[row,S,0,0]=1 if polarity else -1
            if polarity:b[row]=-127
    # The guaranteed up1_low DW ReLU rectifies signed face channels.
    n.weight['up1_low_pw'].zero_();n.bias['up1_low_pw'].zero_()
    for j in range(3):
        k=j*3;n.weight['up1_low_pw'][2*j,k,0,0]=1;n.weight['up1_low_pw'][2*j,k+2,0,0]=-1
        n.weight['up1_low_pw'][2*j+1,k+2,0,0]=1;n.weight['up1_low_pw'][2*j+1,k+1,0,0]=-1
        n.weight['up1_low_pw'][8,[9+4*j,11+4*j],0,0]=1/3
        n.weight['up1_low_pw'][9,[10+4*j,12+4*j],0,0]=1/3
    for group in range(2):n.weight['up1_low_pw'][6+group,21+group,0,0]=1
    n.weight['up2_low_pw'].zero_();n.bias['up2_low_pw'].zero_()
    for j in range(3):n.weight['up2_low_pw'][j,[2*j,2*j+1],0,0]=1
    for row,source in [(3,6),(4,7),(5,8),(6,9)]:n.weight['up2_low_pw'][row,source,0,0]=1
    n.weight['rgb_head'].zero_();n.bias['rgb_head'].copy_(torch.tensor([-128,-255,126],device='cuda'))
    n.weight['rgb_head'][:,:3,0,0]=85/127
    n.weight['rgb_head'][:,3:5,0,0]=torch.tensor([[2,-2],[0,2],[-2,-2]],device='cuda')
    n.weight['rgb_head'][:,5,0,0]=.28;n.weight['rgb_head'][:,6,0,0]=-.33
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

def main():
    setup(271046);d=OUT/'r5-block-faces';d.mkdir(exist_ok=True);disk_guard();n=seed();raw=n.export()
    (d/'initial.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,kind='r5 local box faces')
    engine=IntegerEngine(raw);rows=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');t=teacher(a);o=rgb(engine(tensor(a)[None]*128)/128)
        assert torch.equal(engine(tensor(a)[None]*128),n(tensor(a)[None])*128)
        for k,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{k}.png')
        panel([a,t,o],['原图','4像素宽块面明暗目标','实际B1系数 / 非全局3D'],d/f'{scene}_comparison.jpg')
        rows.append(dict(scene=scene,mae_target=float(np.abs(o.astype(float)-t).mean()),geometry=geometry(o)))
    save_json(d/'seed.json',dict(records=rows,constructed=True,trained=False,native_face_width=4,
        semantic3D=False,coefficient_sha256=sha(d/'initial.bin'),added_inference_operators=False))
    print('R5_FACE_SEED_COMPLETE',json.dumps(rows),flush=True)

if __name__=='__main__':main()
