"""Verify actual BIN geometry features cannot use additive RGB offsets."""
from core import *

@torch.inference_mode()
def main():
    setup(271153);root=OUT/'r25-geometry-only-support';p=root/'calibrated/candidate.bin'
    raw=p.read_bytes();params,_=decode(raw);w=np.array(params['stem'].weights).reshape(8,3,3,3)
    dc=w[3:8].sum((1,2));assert np.all(dc==0),dc.tolist()
    masks=[('down1.pw',range(3,8),range(3,8)),('down2.pw',range(23,32),range(3,8)),
        *[(b+'.pw',range(23,32),range(23,32)) for b in ['res1_conv1','res1_conv2','res2_conv1','res2_conv2','res3_conv1','res3_conv2']],
        ('up1_low_pw',[3],range(23,32))]
    for name,rows,cols in masks:
        q=np.array(params[name].weights);outside=np.ones(q.shape[1],bool);outside[list(cols)]=False
        assert np.all(q[list(rows)][:,outside]==0),name
    engines=dict(previous=IntegerEngine((OUT/'r20-multifeature-structure/calibrated/candidate.bin').read_bytes()),geometry_only=IntegerEngine(raw))
    yy,xx=np.mgrid[:128,:128];spot=np.exp(-((xx-64)**2+(yy-64)**2)/(2*18**2));a=np.repeat(np.rint(65+90*spot)[...,None],3,2).astype(np.uint8)
    b=np.where(((xx>=40)&(xx<88)&(yy>=40)&(yy<88))[...,None],150,65).astype(np.uint8).repeat(3,2)
    records=[]
    for kind,base in [('spot',a),('box',b)]:
        for key,engine in engines.items():
            fields=[]
            for offset in [[0,0,0],[20,20,20],[-10,25,5]]:
                image_array=(base.astype(np.int16)+np.array(offset)[None,None]).astype(np.uint8)
                assert image_array.min()>0 and image_array.max()<255
                nodes,_=engine(tensor(image_array)[None]*128,True);fields.append(nodes['up1_low_pw'][:,3])
            maximum=max(float((z-fields[0]).abs().max()) for z in fields[1:])
            if key=='geometry_only':assert maximum==0
            records.append(dict(fixture=kind,model=key,additive_rgb_offset_geometry_max_error=maximum))
    flat=[]
    for colour in [[40,40,40],[100,140,190],[220,160,80]]:
        a=np.broadcast_to(np.array(colour,np.uint8),(128,128,3)).copy();nodes,_=engines['geometry_only'](tensor(a)[None]*128,True)
        peak=float(nodes['up1_low_pw'][:,3].max());assert peak==0;flat.append(dict(rgb=colour,geometry_support_max=peak))
    save_json(root/'invariance-audit.json',dict(stem_integer_DC_per_input_channel=dc.tolist(),absolute_RGB_and_Y_shortcuts_blocked=True,
        additive_RGB_offset_records=records,constant_colour_fields=flat,coefficient_sha256=sha(p),
        statement='Structural support alone has this invariance. RGB output still changes with colour and luminance. This is not proof of full geometry reconstruction or MC style.',
        inference_extra_operators=False,MC_style_accepted=False))
    # Correct stage metadata of the earlier calibrated R24 file; weights and
    # delivered BIN are unchanged. It selected local step350, not step700.
    cp=OUT/'r24-hard-lighting-negatives-v2/calibrated/selected.pt';state=torch.load(cp,map_location='cpu',weights_only=True)
    source=OUT/'r24-hard-lighting-negatives-v2/selected.pt';state['step']=int(torch.load(source,map_location='cpu',weights_only=True)['step']);state['source_checkpoint']=str(source)
    torch.save(state,cp)
    print('GEOMETRY_SUPPORT_INVARIANCE_PASS',json.dumps(records),flush=True)

if __name__=='__main__':main()
