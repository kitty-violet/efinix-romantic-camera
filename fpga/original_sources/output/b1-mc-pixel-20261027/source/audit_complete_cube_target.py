"""Explicit complete-cube teacher versus actual B1. Teacher is not inference.

The object mask/size below are known from synthetic construction. They are not
available to the deployed B1, so this does not claim real-scene reconstruction.
"""
from core import *
from pixel_targets import panel

def complete_cube(mask,front,depth,bg):
    ys,xs=np.nonzero(mask);x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
    im=Image.new('RGB',(mask.shape[1],mask.shape[0]),tuple(bg));dr=ImageDraw.Draw(im)
    fc=np.asarray(front,np.float32);top=tuple(np.clip(fc+42,0,255).astype(int));side=tuple(np.clip(fc-26,0,255).astype(int))
    dr.polygon([(x0,y0),(x1,y0),(x1+depth,y0-depth),(x0+depth,y0-depth)],fill=top)
    dr.polygon([(x1,y0),(x1,y1),(x1+depth,y1-depth),(x1+depth,y0-depth)],fill=side)
    dr.rectangle([x0,y0,x1-1,y1-1],fill=tuple(fc.astype(int)))
    return np.array(im)

def synthetic():
    yy,xx=np.mgrid[:256,:256];mask=(xx>=88)&(xx<152)&(yy>=80)&(yy<144)
    base=np.full((256,256,3),235,np.uint8);uniform=base.copy();uniform[mask]=55
    gradient=base.copy();v=np.clip(35+(xx-88)*1.4,0,255).astype(np.uint8);gradient[mask]=np.repeat(v[...,None],3,2)[mask]
    patch=uniform.copy();patch[96:128,104:136]=110
    large=(xx>=48)&(xx<176)&(yy>=64)&(yy<192);big=base.copy();big[large]=55
    for name,a,m,depth in [('uniform64',uniform,mask,32),('same_outline_lit64',gradient,mask,32),
        ('same_outline_patch64',patch,mask,32),('large128',big,large,64)]:
        yield name,a,m,depth

@torch.inference_mode()
def main():
    setup(271162);d=OUT/'mc-complete-cube-target-audit';d.mkdir(exist_ok=True)
    paths=dict(early_r9=OUT/'r9-cuboid-volume/dark-large16/initial.bin',
        current_r25=OUT/'exports/mc-geometry-only-r25/coefficients-padded.bin')
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()};records=[]
    for name,a,mask,depth in synthetic():
        target=complete_cube(mask,[55]*3,depth,[235]*3);images=dict(source=a,target=target);cases={}
        for k,n in engines.items():
            nodes,_=n(tensor(a)[None]*128,True);images[k]=rgb(nodes['rgb_head']/128)
            if k=='current_r25':
                v=nodes['up2_low_pw'][0].cpu().numpy()
                extra=np.maximum(v[1]-v[2],0);top=np.maximum(v[3]-v[4],0);side=np.maximum(extra-top,0)
                g=nodes['up1_low_pw'][0,3].cpu().numpy();f=nodes['up1_low_pw'][0,0].cpu().numpy()
                cases[k]=dict(allowed_extra_fraction=float((extra>1).mean()),allowed_top_fraction=float((top>1).mean()),
                    allowed_side_fraction=float((side>1).mean()),support_fraction=float((g>1).mean()))
                for field,z in [('support',g),('allowed_top',top),('allowed_side',side)]:
                    z=np.clip(z*2,0,255).astype(np.uint8)
                    Image.fromarray(z).resize((256,256),Image.Resampling.NEAREST).save(d/f'{name}_{field}.png')
        for k,z in images.items():Image.fromarray(z).save(d/f'{name}_{k}.png')
        panel([images[k] for k in ['source','target','early_r9','current_r25']],
            ['同一物体输入','修正目标：完整立方体 / 教师示意','早期R9实际B1','当前R25实际B1'],d/f'{name}_comparison.png',width=384)
        records.append(dict(name=name,front_size=int(mask.sum()**.5),target_depth=depth,
            target_projected_depth_vector=[depth,-depth],target_source='known synthetic object mask; not an inferred segmentation',
            metrics=cases))
    # A concrete RF witness for the *large*64px relocation target: the target
    # back face appears far above the input object. Its entire RF77 input is
    # identical to an empty background, so a fixed local B1 cannot distinguish
    # the two cases there. This does not rule out the smaller32px target.
    _,large,large_mask,depth=list(synthetic())[-1]
    empty=np.full_like(large,235);target=complete_cube(large_mask,[55]*3,depth,[235]*3)
    px,py=128,8
    pad=lambda a:np.pad(a,((38,38),(38,38),(0,0)),mode='edge')[py:py+77,px:px+77]
    assert np.array_equal(pad(large),pad(empty)) and not np.array_equal(target[py,px],empty[py,px])
    witness={}
    for k,n in engines.items():
        a=rgb(n(tensor(large)[None]*128)/128);b=rgb(n(tensor(empty)[None]*128)/128)
        assert np.array_equal(a[py,px],b[py,px])
        witness[k]=dict(large_actual=a[py,px].tolist(),empty_actual=b[py,px].tolist())
    save_json(d/'audit.json',dict(models={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},records=records,
        old_target='5 independent luminance level sets, 12/16 native-pixel directional sweep',
        corrected_target='one known object mask, one depth vector, three single-plane faces',
        actual_model_changed=False,teacher_used_at_inference=False,
        large_depth_rf_witness=dict(point_xy=[px,py],identical_77x77_inputs=True,target_large=target[py,px].tolist(),target_empty=empty[py,px].tolist(),actual=witness),
        limitations='This audit supplies explicit supervision, not a solved real-scene cuboid assignment. Large depth64 is outside the RF77 local relocation radius.'))
    print('COMPLETE_CUBE_AUDIT_DONE',json.dumps(records),flush=True)

if __name__=='__main__':main()
