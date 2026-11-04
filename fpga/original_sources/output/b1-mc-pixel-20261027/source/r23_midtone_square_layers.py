"""Concentrate square-contour changes in A's middle colour layers.

Protect near-black silhouettes and white light cores from the stronger midtone
projection. No extra filtering; this is a coefficient-only mixed-radius variant.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r21_colour_preserving_layers import seed
from r16_pixel_layer_geometry import cell_audit
from evaluate_r3_r4 import controls

def main():
    setup(271124);d=OUT/'r23-midtone-square-layers';d.mkdir(exist_ok=True);disk_guard()
    c=dict(polarity='dark',radius=2,radii=[1,2,4,4,4,3,2,1]);n,curve=seed(c);raw=n.export();(d/'initial.bin').write_bytes(raw)
    checkpoint(n,d/'initial.pt',0,trained=False,kind='curve A square midtone layers, protected black and white edges',config=c)
    eng=IntegerEngine(raw);records=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');o=rgb(eng(tensor(a)[None]*128)/128)
        assert torch.equal(n(tensor(a)[None])*128,eng(tensor(a)[None]*128))
        Image.fromarray(o).save(d/f'{scene}_student.png');old=read(OUT/f'pixel-three-user-curves/A/{scene}_student.png')
        panel([a,old,o],['原图','曲线A / 原色层形状','曲线A / 中间色层方块化 / 实际B1'],d/f'{scene}_comparison.jpg',width=640)
        crop=(slice(72,232),slice(200,456)) if scene=='indoor' else (slice(120,280),slice(780,1036))
        canvas=Image.new('RGB',(1536,355),'#101721');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        for j,(z,label) in enumerate(zip([a,old,o],['原图局部 / 最近邻放大','原曲线A / 同一局部','新A候选 / 同一局部 / 实际B1'])):
            canvas.paste(Image.fromarray(z[crop]).resize((512,320),Image.Resampling.NEAREST),(j*512,35));draw.text((j*512+5,5),label,font=font,fill='white')
        canvas.save(d/f'{scene}_native_crop.png');records.append(dict(scene=scene,**cell_audit(o)))
    gray=controls(dict(candidate=eng))['candidate'];delta=float(np.max(np.abs(np.array(gray['gray_curve_mean'])-curve['actual']['gray_curve_mean'])))
    assert delta==0
    save_json(d/'seed.json',dict(config=c,records=records,curve='A',saturation=1.3,gray=gray,curve_A_max_gray_delta=delta,
        native_cell=4,shape_projection_pixels=[r*4 for r in c['radii']],strict8_grid=False,coefficient_sha256=sha(d/'initial.bin'),
        no_added_inference_operators=True,trained=False,visual_accepted=False))
    print('MIDTONE_SQUARE_LAYERS_COMPLETE',flush=True)

if __name__=='__main__':main()
