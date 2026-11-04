"""Move complete RGB cells together, rather than growing independent layers.

The existing low-resolution DW shifts a second RGB sample. PW/ReLU implements
a shared luminance decision and a six-hinge RGB mux. Two axes form local square
sample-and-hold patches, entirely inside ordinary exported B1 coefficients.
This is content-dependent 4/8-cell aggregation, not an absolute 8-pixel phase.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from try_user_pixel_curves import seed as curve_seed,requantise
from r16_pixel_layer_geometry import cell_audit
from r4_box_projector import BODY
from evaluate_r3_r4 import controls

CONFIGS={'joint-dark':(-1,-1),'joint-light':(1,1),'joint-light-dark':(1,-1),'joint-dark-light':(-1,1)}

@torch.no_grad()
def seed(name,margin=1):
    c=json.loads((OUT/'pixel-three-user-curves/A/curve.json').read_text('utf-8'))
    n,_=curve_seed(c);ky=len(c['levels'])-1
    # Copy the existing exact curve-A palette readout into the first residual
    # PW, before any cell selection. Preserve each complete RGB triplet.
    basis=n.weight['res3_conv2__pw'][:3,:,0,0].clone();offset=n.bias['res3_conv2__pw'][:3].clone()
    head=n.weight['rgb_head'][:,:3,0,0].clone();hb=n.bias['rgb_head'].clone()
    W=head@basis;B=head@offset+hb+128
    for base in BODY:
        n.weight[base+'__dw'].zero_();n.weight[base+'__dw'][:,0,1,1]=1;n.bias[base+'__dw'].zero_()
        n.weight[base+'__pw'].zero_();n.bias[base+'__pw'].zero_()
    # The old spatial morphology also hardened its threshold masks. Removing
    # that morphology must not leave half-valued masks as extra colour bands.
    # A single existing DW/ReLU hardens the integer decisions before the RGB
    # palette decode. 114 is the uniform response transition of old EDDIE.
    active=ky+16
    n.weight['res1_conv1__dw'][:active,0,1,1]=127
    n.bias['res1_conv1__dw'][:active]=-113*127
    for key in ['up1_low_dw','up2_low_dw','up2_high_dw']:
        n.weight[key].zero_();n.weight[key][:,0,1,1]=1;n.bias[key].zero_()
    for key in ['up1_low_pw','up2_low_pw']:
        n.weight[key].zero_();n.bias[key].zero_()
    # Eight fields: RGB A, RGB B, luma A/B. All RGB are 0..127.
    for row in range(6):
        n.weight['res1_conv1__pw'][row,:,0,0]=W[row%3]/2
        n.bias['res1_conv1__pw'][row]=B[row%3]/2
    lw=torch.tensor([.25,.5,.25],device='cuda');yw=lw@W;yb=lw@B
    for row in [6,7]:
        n.weight['res1_conv1__pw'][row,:,0,0]=yw/2;n.bias['res1_conv1__pw'][row]=yb/2
    def comparison(base,axis,sign):
        for ch in [3,4,5,7]:
            n.weight[base+'__dw'][ch].zero_()
            n.weight[base+'__dw'][ch,0,1,2] = 1 if axis==1 else 0
            if axis==0:n.weight[base+'__dw'][ch,0,2,1]=1
        for ch in range(6):n.weight[base+'__pw'][ch,ch,0,0]=1
        n.weight[base+'__pw'][6,6,0,0]=127*sign
        n.weight[base+'__pw'][6,7,0,0]=-127*sign
        # Select A by default on ties; B wins only by the specified half-RGB
        # margin. This avoids arbitrary colour switches on equal-luma surfaces.
        # The compared lumas are integers: place the threshold BETWEEN integer
        # decisions without a half-open gate (which would subtract from both
        # RGB hinges and create spurious dark cells at a tie).
        n.bias[base+'__pw'][6]=127*(margin+1)
    def hinges(base):
        for ch in range(3):
            n.weight[base+'__pw'][ch,ch,0,0]=1;n.weight[base+'__pw'][ch,6,0,0]=1;n.bias[base+'__pw'][ch]=-127
            n.weight[base+'__pw'][3+ch,3+ch,0,0]=1;n.weight[base+'__pw'][3+ch,6,0,0]=-1
    comparison('res1_conv2',1,CONFIGS[name][0]);hinges('res2_conv1')
    for row in range(6):
        ch=row%3;n.weight['res2_conv2__pw'][row,[ch,ch+3],0,0]=1
    for row in [6,7]:
        n.weight['res2_conv2__pw'][row,:3,0,0]=lw;n.weight['res2_conv2__pw'][row,3:6,0,0]=lw
    comparison('res3_conv1',0,CONFIGS[name][1]);hinges('res3_conv2')
    # res3_conv2 PW has no ReLU. Its following existing DW/ReLU clips the
    # hinges before the up1 PW adds them; no unsigned-mask assumption is made.
    for ch in range(3):
        n.weight['up1_low_pw'][ch,[ch,ch+3],0,0]=1
        n.weight['up2_low_pw'][ch,ch,0,0]=1
    n.weight['rgb_head'].zero_();n.bias['rgb_head'].fill_(-128)
    for ch in range(3):n.weight['rgb_head'][ch,ch,0,0]=2
    requantise(n)
    return n,c

def teacher(a,c,name,margin=2):
    # Curve-A cell colours, then jointly select all three channels by one luma
    # decision. CPU implementation is for training/reference only.
    from r16_pixel_layer_geometry import teacher as cell_teacher
    o=cell_teacher(a,c,[('I',0)]*5)[::4,::4].copy().astype(np.float32)
    for axis,sign in enumerate(CONFIGS[name]):
        q=np.pad(o,((0,1),(0,1),(0,0)),mode='edge')
        b=q[:o.shape[0],1:1+o.shape[1]] if axis==0 else q[1:1+o.shape[0],:o.shape[1]]
        y=(o*np.array([.25,.5,.25])).sum(2);yb=(b*np.array([.25,.5,.25])).sum(2)
        use_b=sign*(yb-y)>margin;o=np.where(use_b[...,None],b,o)
    return o.astype(np.uint8).repeat(4,0).repeat(4,1)

def main():
    setup(271102);root=OUT/'r17-joint-rgb-cells';root.mkdir(exist_ok=True);disk_guard();images={s:[] for s in ['outdoor','indoor']}
    for name in CONFIGS:
        d=root/name;d.mkdir(exist_ok=True);n,c=seed(name);raw=n.export();(d/'initial.bin').write_bytes(raw)
        checkpoint(n,d/'initial.pt',0,trained=False,kind='whole RGB cell selection on curve A',config=name)
        eng=IntegerEngine(raw);records=[]
        for scene in images:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c,name);o=rgb(eng(tensor(a)[None]*128)/128)
            nodes,_=eng(tensor(a)[None]*128,True)
            assert torch.equal(n(tensor(a)[None])*128,nodes['rgb_head'])
            for gate in ['res1_conv2.pw','res3_conv1.pw']:
                g=nodes[gate][:,6].clamp_min(0)
                assert torch.all((g==0)|(g==127)),(gate,torch.unique(g).tolist())
            assert cell_audit(o)['max_within_cell_rgb_range']==0
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            images[scene].append(o);records.append(dict(scene=scene,mae=float(np.abs(o.astype(float)-t).mean()),**cell_audit(o)))
        gray=controls(dict(candidate=eng))['candidate']
        save_json(d/'seed.json',dict(config=name,curve='A',saturation=1.3,records=records,gray=gray,
            native_cell=4,content_dependent_4_and_8_aggregation=True,strict8_grid=False,
            added_inference_operators=False,whole_rgb_sample_selection=True,coefficient_sha256=sha(d/'initial.bin')))
        print('JOINT_RGB_CELLS',name,json.dumps(records),flush=True)
    for scene,zs in images.items():
        old=read(OUT/f'pixel-three-user-curves/A/{scene}_student.png')
        panel([old,*zs],['旧曲线A',*CONFIGS],root/f'{scene}_all.jpg',width=480)
        crop=(slice(72,232),slice(200,456)) if scene=='indoor' else (slice(120,280),slice(780,1036))
        canvas=Image.new('RGB',(1536,700),'#101721');dr=ImageDraw.Draw(canvas);f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        for j,(z,label) in enumerate(zip([old,*zs],['旧曲线A',*CONFIGS])):
            x=j%3*512;y=j//3*350;canvas.paste(Image.fromarray(z[crop]).resize((512,320),Image.Resampling.NEAREST),(x,y+30));dr.text((x+5,y+4),label,font=f,fill='white')
        canvas.save(root/f'{scene}_native_crop.png')
    print('JOINT_RGB_CELLS_COMPLETE',flush=True)

if __name__=='__main__':main()
