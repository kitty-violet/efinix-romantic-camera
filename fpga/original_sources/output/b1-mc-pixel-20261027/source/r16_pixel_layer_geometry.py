"""Curve A, independently test colour-cell geometry in the unchanged B1 graph.

No post pixelation. All candidates produce their RGB from the delivered binary.
The native phase is four pixels; content-dependent merging is not a fixed 8-grid.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from try_user_pixel_curves import seed as curve_seed,requantise
from r3_palette_blocks import BODY,geometry,CHROMA_VALUES,CHROMA_THRESH
from r13_region_cuboids import region_stats

CONFIGS={
    'cell': [('I',0)]*5,
    'axis-vote': [('M',1),('M',0),('M',1),('M',0),('I',0)],
    'axis-open': [('E',1),('D',1),('E',0),('D',0),('I',0)],
    'axis-close': [('D',1),('E',1),('D',0),('E',0),('I',0)],
    'axis-open-close': [('E',1),('E',0),('D',1),('D',0),('I',0)],
}

@torch.no_grad()
def seed(name):
    curve=json.loads((OUT/'pixel-three-user-curves/A/curve.json').read_text('utf-8'))
    n,_=curve_seed(curve);count=len(curve['levels'])-1+16
    for base,(op,axis) in zip(BODY,CONFIGS[name]):
        dw=n.weight[base+'__dw'];db=n.bias[base+'__dw'];pw=n.weight[base+'__pw'];pb=n.bias[base+'__pw']
        dw.zero_();db.zero_();dw[:,0,1,1]=1;pw.zero_();pb.zero_()
        for ch in range(count):
            pw[ch,ch,0,0]=1
            if op!='I':
                dw[ch].zero_()
                if axis:dw[ch,0,1,:]=1
                else:dw[ch,0,:,1]=1
                db[ch]=(-2*127 if op=='E' else -127 if op=='M' else 0)
    # Flat pixel route: one RGB per native cell; remove old one-cell relief.
    n.weight['rgb_head'][:,3:5].zero_()
    requantise(n)
    # Keep the intentionally finer luminance-basis quantisation of curve A.
    n.res3_conv2__pw_m[:,[0,3,4]]=256
    return n,curve

def teacher(a,c,ops):
    h,w=a.shape[:2];z=cv2.GaussianBlur(a.astype(np.float32),(3,3),0,borderType=cv2.BORDER_REPLICATE)
    r,g,b=z.reshape(h//4,4,w//4,4,3).mean((1,3)).transpose(2,0,1)
    def transform(v,ts,levels):
        out=np.full_like(v,levels[0])
        for t,inc in zip(ts,np.diff(levels)):
            m=(v>t).astype(np.uint8)
            for op,axis in ops:
                k=np.ones((1,3) if axis else (3,1),np.uint8)
                if op=='D':m=cv2.dilate(m,k,borderType=cv2.BORDER_REPLICATE)
                elif op=='E':m=cv2.erode(m,k,borderType=cv2.BORDER_REPLICATE)
                elif op=='M':m=(cv2.filter2D(m.astype(np.float32),-1,k,borderType=cv2.BORDER_REPLICATE)>=2).astype(np.uint8)
            out+=inc*m
        return out
    y=transform((r+2*g+b)/4,c['thresholds'],c['levels'])
    co=transform((r-b)/2,CHROMA_THRESH,CHROMA_VALUES)*1.3
    cg=transform((-r+2*g-b)/4,CHROMA_THRESH,CHROMA_VALUES)*1.3
    return np.clip(np.stack([y+co-cg,y+cg,y-co-cg],2)+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

def cell_audit(a):
    h,w=a.shape[:2];cells=a.reshape(h//4,4,w//4,4,3)
    # Every RGB channel must share exactly the same spatial cells.
    spread=cells.max((1,3)).astype(int)-cells.min((1,3)).astype(int)
    return dict(max_within_cell_rgb_range=int(spread.max()),nonconstant_cell_fraction=float(np.any(spread,2).mean()),
                geometry=geometry(a),regions=region_stats(a))

def main():
    setup(271101);root=OUT/'r16-pixel-colour-cells';root.mkdir(exist_ok=True);disk_guard();all_images={s:[] for s in ['outdoor','indoor']}
    for name,ops in CONFIGS.items():
        d=root/name;d.mkdir(exist_ok=True);n,c=seed(name);raw=n.export();(d/'initial.bin').write_bytes(raw)
        checkpoint(n,d/'initial.pt',0,trained=False,kind='curve A with shared axis colour-cell geometry',geometry=name)
        eng=IntegerEngine(raw);records=[]
        for scene in all_images:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c,ops);o=rgb(eng(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,eng(tensor(a)[None]*128))
            assert cell_audit(o)['max_within_cell_rgb_range']==0
            for key,z in [('target',t),('student',o)]:Image.fromarray(z).save(d/f'{scene}_{key}.png')
            all_images[scene].append(o);records.append(dict(scene=scene,mae=float(np.abs(o.astype(float)-t).mean()),**cell_audit(o)))
        save_json(d/'seed.json',dict(config=name,ops=ops,records=records,curve='A',saturation=1.3,
            native_cell=4,strict6or8_grid=False,added_inference_operators=False,coefficient_sha256=sha(d/'initial.bin')))
        print('PIXEL_LAYER_GEOMETRY',name,json.dumps(records),flush=True)
    for scene,images in all_images.items():
        a=read(OUT/f'targets/{scene}_source.png');old=read(OUT/f'pixel-three-user-curves/A/{scene}_student.png')
        panel([old,*images],['旧曲线A',*CONFIGS],root/f'{scene}_all.jpg',width=480)
        # Native cell corners in the ceiling region circled by the user.
        crop=(slice(72,232),slice(200,456)) if scene=='indoor' else (slice(120,280),slice(780,1036))
        tiles=[Image.fromarray(z[crop]).resize((512,320),Image.Resampling.NEAREST) for z in [old,*images]]
        canvas=Image.new('RGB',(1536,700),'#101721');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        for j,(im,label) in enumerate(zip(tiles,['旧曲线A',*CONFIGS])):
            x=(j%3)*512;y=(j//3)*350;canvas.paste(im,(x,y+30));draw.text((x+6,y+4),label,font=font,fill='white')
        canvas.save(root/f'{scene}_native_crop.png')
    print('PIXEL_LAYER_GEOMETRY_COMPLETE',flush=True)

if __name__=='__main__':main()
