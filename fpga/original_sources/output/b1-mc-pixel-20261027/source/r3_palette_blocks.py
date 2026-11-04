"""Shared YCoCg palette and square contour morphology in fixed ordinary B1.

The CPU teacher exists only to supervise/measure. Native inference remains the
existing24 convolution graph and nearest stages. No absolute6-grid is claimed.
"""
from core import *
from train_pixel import pixel_start,checkpoint,data
from pixel_targets import panel
import argparse

LUMA=np.array([.25,.5,.25],np.float32)
CHROMA_VALUES=np.array([-112,-72,-36,-16,0,16,36,72,112],np.float32)
CHROMA_THRESH=(CHROMA_VALUES[1:]+CHROMA_VALUES[:-1])/2
CONFIGS={
 'flat-clean':dict(morph='I',blur=True,face=0.,levels_y=8,levels_c=9),
 'flat-large':dict(morph='OC3',blur=True,face=.05,levels_y=8,levels_c=9),
 'mc-close':dict(morph='C5',blur=True,face=.45,levels_y=6,levels_c=9),
 'mc-open':dict(morph='O5',blur=True,face=.45,levels_y=6,levels_c=9),
 'mc-balanced':dict(morph='OC3',blur=True,face=.30,levels_y=6,levels_c=9),
}
SEQUENCES={'I':'IIIII','C5':'DDEIE','O5':'EIEDD','OC3':'EDDIE'}
BODY=['res1_conv1','res1_conv2','res2_conv1','res2_conv2','res3_conv1']

def channels(config):
    ky=config['levels_y']-1;kc=config['levels_c']-1
    return ky,kc,[(0,ky),(ky,kc),(ky+kc,kc)]

def teacher(a,config):
    z=a.astype(np.float32)
    if config['blur']:z=cv2.GaussianBlur(z,(3,3),0,borderType=cv2.BORDER_REPLICATE)
    h,w=z.shape[:2];low=z.reshape(h//4,4,w//4,4,3).mean((1,3))
    R,G,B=low.transpose(2,0,1)
    vals=[(R+2*G+B)/4,(R-B)/2,(-R+2*G-B)/4]
    quant=[]
    for j,(v,k) in enumerate(zip(vals,[config['levels_y']-1,config['levels_c']-1,config['levels_c']-1])):
        q=np.zeros_like(v) if j==0 else np.full_like(v,CHROMA_VALUES[0])
        thresholds=(np.arange(k)+.5)*255/k if j==0 else CHROMA_THRESH
        increments=np.full(k,255/k) if j==0 else np.diff(CHROMA_VALUES)
        for threshold,increment in zip(thresholds,increments):
            gate=(v>threshold).astype(np.uint8)
            if j==0:
                for op in SEQUENCES[config['morph']]:
                    if op=='D':gate=cv2.dilate(gate,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
                    if op=='E':gate=cv2.erode(gate,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
            q+=increment*gate
        quant.append(q)
    Y,Co,Cg=quant
    top=np.pad(Y,((1,0),(0,0)),mode='edge')[:-1]
    right=np.pad(Y,((0,0),(0,1)),mode='edge')[:,1:]
    shade=config['face']*(np.maximum(Y-top,0)-1.2*np.maximum(Y-right,0))
    out=np.stack([Y+Co-Cg+shade,Y+Cg+shade,Y-Co-Cg+shade],2)
    return np.clip(out+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

@torch.no_grad()
def seed(config):
    n=pixel_start()
    for p in n.parameters():p.zero_()
    eye=torch.zeros(3,3,device='cuda');eye[1,1]=1
    avg=torch.zeros_like(eye);avg[1:,1:]=.25
    g=torch.tensor(np.outer([1,2,1],[1,2,1])/16,device='cuda',dtype=torch.float32)
    for r in n.plan['nodes']:
        k=r['id'].replace('.','__')
        if k in n.weight and r['kind']=='dw3x3':n.weight[k].copy_(eye[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['stem'][c,c].copy_((g if config['blur'] else eye)*.5);n.bias['stem'][c]=64
    for k in ['down1__dw','down2__dw']:n.weight[k].copy_(avg[None,None].expand_as(n.weight[k]))
    for c in range(3):n.weight['down1__pw'][c,c,0,0]=1
    ky,kc,ranges=channels(config)
    vectors=[[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]]
    for group,((start,k),vector) in enumerate(zip(ranges,vectors)):
        thresholds=(np.arange(k)+.5)*127/k if group==0 else (CHROMA_THRESH+127)/2
        for i,t in enumerate(thresholds):
            c=start+i;n.weight['down2__pw'][c,:3,0,0]=torch.tensor(vector,device='cuda')*127
            n.bias['down2__pw'][c]=float((63.5 if group else 0)*127-t*127)
    for base,op in zip(BODY,SEQUENCES[config['morph']]):
        for c in range(ky+2*kc):
            n.weight[base+'__pw'][c,c,0,0]=1
            if c<ky and op!='I':
                n.weight[base+'__dw'][c,0].fill_(1/9)
                n.weight[base+'__pw'][c,c,0,0]=9
                if op=='E':n.bias[base+'__pw'][c]=-8*127
        # Every erosion is deliberately placed at an existing PW with ReLU.
        if op=='E':assert next(r for r in n.plan['nodes'] if r['id']==base+'.pw')['activation']=='relu'
    for c in range(3):
        begin,k=ranges[c]
        if c==0:n.weight['res3_conv2__pw'][c,begin:begin+k,0,0]=1/k
        else:
            n.weight['res3_conv2__pw'][c,begin:begin+k,0,0]=torch.tensor(np.diff(CHROMA_VALUES)/(2*127),device='cuda')
            n.bias['res3_conv2__pw'][c]=float((CHROMA_VALUES[0]+127)/2)
    # Extra copies of quantized luminance become directional face fields.
    n.weight['res3_conv2__pw'][3,:ky,0,0]=1/ky
    n.weight['res3_conv2__pw'][4,:ky,0,0]=1/ky
    n.weight['up1_low_dw'][3,0,0,1]=-1
    n.weight['up1_low_dw'][4,0,1,2]=-1
    for c in range(5):
        n.weight['up1_low_pw'][c,c,0,0]=1;n.weight['up2_low_pw'][c,c,0,0]=1
    for c,terms in enumerate([[1,1,-1],[1,0,1],[1,-1,-1]]):
        n.weight['rgb_head'][c,:3,0,0]=torch.tensor(terms,device='cuda')*torch.tensor([255/127,2,2],device='cuda')
        n.weight['rgb_head'][c,3,0,0]=config['face']*255/127
        n.weight['rgb_head'][c,4,0,0]=-1.2*config['face']*255/127
    n.bias['rgb_head'].copy_(torch.tensor([-128,-255,126],device='cuda'))
    for k,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,k+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    n.training_saturation_leak=.01
    return n

def geometry(a):
    # Quantization bins are common across channels, so neutral surfaces stay neutral.
    low=a[::4,::4].astype(np.float32)
    q=np.rint(low/12).astype(np.int16)
    dx=np.max(np.abs(np.diff(low,axis=1)),2)>10
    dy=np.max(np.abs(np.diff(low,axis=0)),2)>10
    run=[]
    for mat in [dx,dy.T]:
        for col in mat.T:
            z=np.pad(col,(1,1));edges=np.diff(z.astype(int));run.extend((np.where(edges==-1)[0]-np.where(edges==1)[0]).tolist())
    # Four-neighbour equal-ish adjacent cell pairs quantify merging without assuming global6 phase.
    nearx=np.max(np.abs(q[:,1:]-q[:,:-1]),2)==0;neary=np.max(np.abs(q[1:]-q[:-1]),2)==0
    r,g,b=low.transpose(2,0,1);neutral=(np.maximum.reduce([r,g,b])-np.minimum.reduce([r,g,b]))<10
    return dict(mean_straight_edge_run_pixels=float(np.mean(run)*4) if run else 0,
        long_straight_edge_fraction=float(np.mean(np.array(run)>=3)) if run else 0,
        merged_neighbour_fraction=float((nearx.mean()+neary.mean())/2),neutral_fraction=float(neutral.mean()),
        isolated_edge_runs_fraction=float(np.mean(np.array(run)==1)) if run else 0)

def main(args):
    setup(271040);dest=OUT/'r3-shared-palette';dest.mkdir(exist_ok=True);disk_guard()
    names=args.names or list(CONFIGS)
    for name in names:
        config=CONFIGS[name];d=dest/name;d.mkdir(exist_ok=True);net=seed(config);raw=net.export()
        (d/'initial.bin').write_bytes(raw);checkpoint(net,d/'initial.pt',0,trained=False,config=config)
        engine=IntegerEngine(raw);records=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,config);o=rgb(engine(tensor(a)[None]*128)/128)
            assert torch.equal(net(tensor(a)[None])*128,engine(tensor(a)[None]*128))
            for kind,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{kind}.png')
            panel([a,t,o],['原图',name+' / 目标',name+' / 构造系数 纯B1整数'],d/f'{scene}_comparison.jpg')
            records.append(dict(scene=scene,mae_target=float(np.abs(t.astype(float)-o).mean()),geometry=geometry(o)))
        save_json(d/'seed.json',dict(config=config,records=records,coefficient_sha256=sha(d/'initial.bin'),
            constructed=True,trained=False,semantic3D_MC=False,strict6_grid=False,inference_extra_operators=False))
        print('SHARED_PALETTE_SEED',name,json.dumps(records),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--names',nargs='*');main(ap.parse_args())
