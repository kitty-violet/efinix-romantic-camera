"""Six-scene, teacher/previous/current matched still delivery and diagnostics."""
import ast,json,time,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .block_separation import transform_surface,local_contrast,steepen,block_activity,PRESETS
from .painting_parameters import as_lab
from .runtime import configure_workspace,file_hash,write_json

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OLD=ROOT/'directional-paint-20261004-v3';OUT=ROOT/'block-separation-20261004-v3'


def load(p):return np.array(Image.open(p).convert('RGB'))


def main():
    configure_workspace();cv2.setNumThreads(2);OUT.mkdir(exist_ok=False)
    previous=json.loads((OLD/'experiment.json').read_text(encoding='utf-8'))
    records=[];inputs=[]
    for source in previous['inputs']:
        key=source['id'];start=time.perf_counter()
        for suffix in ['teacher','direction4','surface']:
            shutil.copy2(OLD/(key+'_'+suffix+'.png'),OUT/(key+'_'+suffix+'.png'))
        assert file_hash(OLD/(key+'_student.png'))==source['input_sha256']
        base=load(OUT/(key+'_surface.png'))
        for preset in PRESETS:
            out,stages=transform_surface(base,preset,True)
            assert out.shape==(720,1280,3) and out.dtype==np.uint8
            Image.fromarray(out).save(OUT/(key+'_'+preset+'.png'))
            if key=='val_plaza_090' and preset=='balanced':
                for name,value in stages.items():Image.fromarray(value).save(OUT/(key+'_'+name+'.png'))
            records.append(dict(id=key,preset=preset,output_sha256=file_hash(OUT/(key+'_'+preset+'.png')),
                                clipped_zero_fraction=float((out==0).mean()),clipped_255_fraction=float((out==255).mean())))
        inputs.append(dict(id=key,surface_sha256=file_hash(OUT/(key+'_surface.png')),
                           teacher_comparison_only_sha256=file_hash(OUT/(key+'_teacher.png')),
                           previous_sha256=file_hash(OUT/(key+'_direction4.png')),neural_prediction=source))
        print('BLOCK_PUBLISH',key,round(time.perf_counter()-start,2),flush=True)
    checks={}
    # Distinguish implementation guarantees from subjective picture acceptance.
    flat=np.full((96,128),37.,np.float32)
    y=local_contrast(flat,8.,.42,(0,100))
    checks['contrast_constant_max_error']=float(np.abs(y-flat).max());assert checks['contrast_constant_max_error']<1e-4
    x=np.tile(np.linspace(0,100,128,dtype=np.float32),(96,1))
    y=local_contrast(x,8.,0.,(0,100));checks['contrast_zero_amount_error']=float(np.abs(y-x).max());assert checks['contrast_zero_amount_error']<1e-4
    ramp=np.zeros((96,128,3),np.float32);ramp[:,:,0]=x
    checks['planar_ramp_activity_max']=float(block_activity(ramp)[12:-12,12:-12].max());assert checks['planar_ramp_activity_max']<1e-3
    steps=np.zeros((96,128,3),np.float32);steps[:,:,0]=30;steps[:,64:,0]=70
    q=steepen(steps,2);checks['strong_step_shock_max_change']=float(np.abs(q-steps).max());assert checks['strong_step_shock_max_change']<.01
    q=steepen(ramp,2);checks['shock_range']=[float(q[:,:,0].min()),float(q[:,:,0].max())];assert q[:,:,0].min()>=0 and q[:,:,0].max()<=100
    # Compare the exact same tree coordinates, not auto-selected flattering patches.
    stats={}
    for suffix in ['teacher','direction4','balanced','cleanblocks']:
        lab=as_lab(load(OUT/('val_plaza_090_'+suffix+'.png')));q=lab[280:560,10:330]
        dx=np.diff(q[:,:,0],axis=1);dy=np.diff(q[:,:,0],axis=0)
        grad=np.r_[np.abs(dx).ravel(),np.abs(dy).ravel()]
        stats[suffix]=dict(tree_L_std=float(q[:,:,0].std()),neighbor_L_difference_p90=float(np.percentile(grad,90)),
                           neighbor_L_difference_p99=float(np.percentile(grad,99)),mean_lab=q.mean((0,1)).tolist())
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    crop=(0,230,420,690);sheet=Image.new('RGB',(1260,494),'#17202a');d=ImageDraw.Draw(sheet)
    for i,(suffix,title) in enumerate([('teacher','教师参考'),('direction4','上一版'),('balanced','本次：块面分离')]):
        d.text((i*420+6,5),title,font=font,fill='white');sheet.paste(Image.open(OUT/('val_plaza_090_'+suffix+'.png')).crop(crop),(i*420,34))
    sheet.save(OUT/'tree_comparison.png')
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_new_hallway_120',(0,0,320,280))]
    sheet=Image.new('RGB',(960,len(cases)*316),'#17202a');d=ImageDraw.Draw(sheet)
    for j,(key,box) in enumerate(cases):
        for i,(suffix,title) in enumerate([('teacher','教师参考'),('direction4','上一版'),('balanced','本次')]):
            d.text((i*320+5,j*316+5),title,font=font,fill='white');sheet.paste(Image.open(OUT/(key+'_'+suffix+'.png')).crop(box),(i*320,j*316+34))
    sheet.save(OUT/'other_scenes.png')
    sheet=Image.new('RGB',(1280,3*394),'#17202a');d=ImageDraw.Draw(sheet)
    for i,entry in enumerate(inputs):
        x=(i%2)*640;y=(i//2)*394;key=entry['id']
        d.text((x+5,y+5),key,font=font,fill='white')
        sheet.paste(Image.open(OUT/(key+'_balanced.png')).resize((640,360),Image.Resampling.LANCZOS),(x,y+34))
    sheet.save(OUT/'six_scene_overview.png')
    code=OUT/'source_code';code.mkdir()
    for name in ['block_separation.py','directional_paint.py','painting_parameters.py',Path(__file__).name]:
        p=Path(__file__).with_name(name);ast.parse(p.read_text(encoding='utf-8'));shutil.copy2(p,code/name)
    shutil.copy2(OLD/'color_adapter.npz',OUT/'color_adapter.npz')
    write_json(OUT/'experiment.json',dict(presets=PRESETS,inputs=inputs,previous_package=str(OLD),
        neural_weight_source=previous['neural_weight_source'],neural_weight_sha256=previous['neural_weight_sha256'],
        code={p.name:file_hash(p) for p in code.iterdir()},new_training=False,teacher_at_inference=False,
        original_detail_added=False,material_masks=False,per_frame_histograms=False,
        video_tested=False,model_integrated=False,board_validated=False))
    write_json(OUT/'validation.json',dict(AST_pass=True,checks=checks,records=records,matched_tree_diagnostics=stats,
                                       note='Diagnostics are not visual style acceptance or temporal validation.'))
    print('BLOCK_PUBLISH_DONE',OUT,flush=True)


if __name__=='__main__':main()
