"""Complete six-scene still package; no training or video claim."""
import ast,json,time,shutil
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from .directional_paint import transform,surface,PRESETS,sector_samples
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OUT=ROOT/'directional-paint-20261004-v3';OLD=ROOT/'painting-parameters-20261004-v2'


def load(p):return np.array(Image.open(p).convert('RGB'))


def main():
    configure_workspace();set_reproducible(2026100492);cv2.setNumThreads(2);OUT.mkdir(exist_ok=False)
    names=['directional_paint.py','painting_parameters.py',Path(__file__).name]
    hashes={n:file_hash(Path(__file__).with_name(n)) for n in names}
    for n in names:ast.parse(Path(__file__).with_name(n).read_text(encoding='utf-8'))
    adapter=np.load(OLD/'color_adapter.npz')['lab_chroma_affine'];config=json.loads((OLD/'experiment.json').read_text())
    rows=[]
    for source in config['inputs']:
        key=source['id'];p=OLD/(key+'_student.png');assert file_hash(p)==source['input_sha256'];rgb=load(p)
        for suffix in ['student','teacher','source','poster','poster_coarse']:shutil.copy2(OLD/(key+'_'+suffix+'.png'),OUT/(key+'_'+suffix+'.png'))
        for preset in PRESETS:
            start=time.perf_counter();out,stages=transform(rgb,adapter,preset,True)
            assert out.shape==rgb.shape and out.dtype==np.uint8
            Image.fromarray(out).save(OUT/(key+'_'+preset+'.png'))
            if preset=='direction4':
                for name,a in stages.items():Image.fromarray(a).save(OUT/(key+'_'+name+'.png'))
            rows.append(dict(id=key,preset=preset,seconds=time.perf_counter()-start))
        print('DIRECTIONAL_PUBLISH',key,flush=True)
    checks={}
    for preset,cfg in PRESETS.items():
        tests={};flat=np.full((96,128,3),127,np.uint8)
        tests['constant_max_error']=int(np.abs(surface(flat,cfg).astype(int)-flat).max());assert tests['constant_max_error']==0
        step=flat.copy();step[:,:64]=40;step[:,64:]=200
        q=surface(step,cfg);tests['step_width']=int(((q[48,:,0]>65)&(q[48,:,0]<175)).sum());assert tests['step_width']<=2
        bars=[]
        for width in [1,2,3]:
            bar=np.full_like(flat,170);bar[12:84,64:64+width]=50
            c=cv2.pyrMeanShiftFiltering(cv2.medianBlur(bar,5),sp=8,sr=20,maxLevel=1)
            new=surface(bar,cfg)
            bars.append(dict(width=width,input_contrast=120,C_contrast=float(170-c[32:64,64:64+width].mean()),
                             new_contrast=float(170-new[32:64,64:64+width].mean())))
        tests['thin_bar_diagnostics']=bars
        z=np.repeat(np.clip(130+np.random.default_rng(21).normal(0,30,(96,128,1)),0,255).astype(np.uint8),3,2)
        tests['fine_std_before']=float(z[20:-20,20:-20,0].std());tests['fine_std_after']=float(surface(z,cfg)[20:-20,20:-20,0].std())
        coords,weights=sector_samples(cfg['radius']);tests['samples']=len(coords)
        assert np.allclose(weights.sum(1),1)
        checks[preset]=tests
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680)),('val_new_hallway_120',(0,0,320,280))]
    b=Image.new('RGB',(1600,len(cases)*316),'#17202a');d=ImageDraw.Draw(b)
    for j,(key,box) in enumerate(cases):
        for i,(suffix,title) in enumerate([('teacher','教师参考'),('poster','旧B'),('poster_coarse','旧C'),('direction4','本次：较清晰'),('direction6','本次：更粗')]):
            d.text((i*320+4,j*316+4),title,font=f,fill='white');b.paste(Image.open(OUT/(key+'_'+suffix+'.png')).crop(box),(i*320,j*316+34))
    b.save(OUT/'comparison.png')
    b.crop((320,0,1280,316)).save(OUT/'grass_B_C_new.png')
    (OUT/'source_code').mkdir()
    for n in names:shutil.copy2(Path(__file__).with_name(n),OUT/'source_code'/n)
    shutil.copy2(OLD/'color_adapter.npz',OUT/'color_adapter.npz')
    write_json(OUT/'experiment.json',dict(presets=PRESETS,code=hashes,inputs=config['inputs'],
        neural_weight_source=config['neural_weight_source'],neural_weight_sha256=config['neural_weight_sha256'],
        inherited_color_adapter_sha256=file_hash(OUT/'color_adapter.npz'),new_training=False,teacher_at_inference=False,
        random_texture=False,model_integration=False,video_tested=False,board_validated=False,
        method='RGB structure tensor, tangent ellipse, 8 overlapping sectors, variance-weighted means; discrete approximation'))
    write_json(OUT/'validation.json',dict(checks=checks,records=rows,AST_pass=True))
    print('DIRECTIONAL_PUBLISH_DONE',OUT,flush=True)


if __name__=='__main__':main()
