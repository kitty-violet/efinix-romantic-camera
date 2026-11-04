"""Larger local square layer contours while retaining the full chroma palette.

Y uses H/V projected masks; colour uses symmetric axis voting instead of
shrinking all chroma toward grey. Original curve-A readout is retained exactly.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from try_user_pixel_curves import seed as curve_seed,requantise
from r4_box_projector import BODY
from r16_pixel_layer_geometry import cell_audit
from r3_palette_blocks import CHROMA_VALUES,CHROMA_THRESH
from evaluate_r3_r4 import controls

CONFIGS={f'{p}-r{r*4}':dict(polarity=p,radius=r) for p in ['bright','dark'] for r in [2,4]}

@torch.no_grad()
def seed(c):
    curve=json.loads((OUT/'pixel-three-user-curves/A/curve.json').read_text('utf-8'))
    n,_=curve_seed(curve);basis=n.weight['res3_conv2__pw'][0,:8,0,0].clone();yb=float(n.bias['res3_conv2__pw'][0])
    for base in BODY:
        n.weight[base+'__dw'].zero_();n.weight[base+'__dw'][:,0,1,1]=1;n.bias[base+'__dw'].zero_()
        n.weight[base+'__pw'].zero_();n.bias[base+'__pw'].zero_()
    n.weight['down2__pw'].zero_();n.bias['down2__pw'].zero_();sign=1 if c['polarity']=='bright' else -1
    for j,t in enumerate(curve['thresholds']):
        for ch in [2*j,2*j+1]:
            n.weight['down2__pw'][ch,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*126
            n.bias['down2__pw'][ch]=float(63.5-126*t/2)
            n.weight['res1_conv1__pw'][ch,ch,0,0]=sign;n.bias['res1_conv1__pw'][ch]=127 if sign<0 else 0
    for group,vec in enumerate([[.5,0,-.5],[-.25,.5,-.25]]):
        for j,t in enumerate(CHROMA_THRESH):
            ch=16+8*group+j;n.weight['down2__pw'][ch,:3,0,0]=torch.tensor(vec,device='cuda')*126
            n.bias['down2__pw'][ch]=float(63.5-126*t/2);n.weight['res1_conv1__pw'][ch,ch,0,0]=1
    n.weight['res1_conv1__dw'][:,0,1,1]=127;n.bias['res1_conv1__dw'].fill_(-113*127)
    for depth,base in enumerate(BODY[1:5]):
        for ch in range(32):
            n.weight[base+'__pw'][ch,ch,0,0]=1
            if ch<16 and depth<c.get('radii',[c['radius']]*8)[ch//2]:
                n.weight[base+'__dw'][ch].zero_()
                if ch%2==0:n.weight[base+'__dw'][ch,0,1,:]=1
                else:n.weight[base+'__dw'][ch,0,:,1]=1
            elif ch>=16:
                n.weight[base+'__dw'][ch].zero_()
                if depth%2==0:n.weight[base+'__dw'][ch,0,1,:]=1
                else:n.weight[base+'__dw'][ch,0,:,1]=1
                n.bias[base+'__dw'][ch]=-127
    for j in range(8):
        n.weight['res3_conv2__pw'][j,2*j:2*j+2,0,0]=1;n.bias['res3_conv2__pw'][j]=-127
    for ch in range(16):n.weight['res3_conv2__pw'][8+ch,16+ch,0,0]=1
    for key in ['up1_low_dw','up2_low_dw','up2_high_dw']:
        n.weight[key].zero_();n.weight[key][:,0,1,1]=1;n.bias[key].zero_()
    n.weight['up1_low_pw'].zero_();n.bias['up1_low_pw'].zero_();n.weight['up2_low_pw'].zero_();n.bias['up2_low_pw'].zero_()
    n.weight['up1_low_pw'][0,:8,0,0]=basis*sign;n.bias['up1_low_pw'][0]=yb+(float(basis.sum())*127 if sign<0 else 0)
    for group in range(2):
        n.weight['up1_low_pw'][1+group,8+8*group:16+8*group,0,0]=torch.tensor(np.diff(CHROMA_VALUES)/(2*127),device='cuda')
        n.bias['up1_low_pw'][1+group]=float((CHROMA_VALUES[0]+127)/2)
    for ch in range(3):n.weight['up2_low_pw'][ch,ch,0,0]=1
    n.weight['rgb_head'][:,3:5].zero_();requantise(n);n.up1_low_pw_m[:,0]=256
    return n,curve

def main():
    setup(271122);root=OUT/'r21-colour-preserving-layers';root.mkdir(exist_ok=True);disk_guard();images={s:[] for s in ['outdoor','indoor']}
    for name,c in CONFIGS.items():
        d=root/name;d.mkdir(exist_ok=True);n,curve=seed(c);raw=n.export();(d/'initial.bin').write_bytes(raw)
        checkpoint(n,d/'initial.pt',0,trained=False,kind='curve A larger rectangular colour layers with preserved chroma',config=c)
        eng=IntegerEngine(raw);stats=[]
        for scene in images:
            a=read(OUT/f'targets/{scene}_source.png');o=rgb(eng(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,eng(tensor(a)[None]*128))
            Image.fromarray(o).save(d/f'{scene}_student.png');images[scene].append(o);stats.append(dict(scene=scene,**cell_audit(o)))
        gray=controls(dict(candidate=eng))['candidate'];delta=float(np.max(np.abs(np.array(gray['gray_curve_mean'])-curve['actual']['gray_curve_mean'])))
        assert delta==0,(name,delta)
        save_json(d/'seed.json',dict(config=c,curve='A',saturation=1.3,chroma_levels=9,gray=gray,curve_A_max_gray_delta=delta,
            records=stats,larger_shape_projection_pixels=c['radius']*4,native_phase=4,strict8_grid=False,
            added_inference_operators=False,coefficient_sha256=sha(d/'initial.bin')))
        print('COLOUR_PRESERVING_LAYER',name,json.dumps(stats),flush=True)
    for scene,zs in images.items():
        old=read(OUT/f'pixel-three-user-curves/A/{scene}_student.png');panel([old,*zs],['旧曲线A',*CONFIGS],root/f'{scene}_all.jpg',width=480)
        crop=(slice(72,232),slice(200,456)) if scene=='indoor' else (slice(120,280),slice(780,1036))
        canvas=Image.new('RGB',(1536,700),'#101721');dr=ImageDraw.Draw(canvas);f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        for j,(z,label) in enumerate(zip([old,*zs],['旧曲线A',*CONFIGS])):
            x=j%3*512;y=j//3*350;canvas.paste(Image.fromarray(z[crop]).resize((512,320),Image.Resampling.NEAREST),(x,y+30));dr.text((x+5,y+4),label,font=f,fill='white')
        canvas.save(root/f'{scene}_native_crop.png')
    print('COLOUR_PRESERVING_LAYERS_COMPLETE',flush=True)

if __name__=='__main__':main()
