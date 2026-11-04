"""Native-pixel diagnostic controls. Never an automatic cartoon/style pass.

Gaussian and global-affine results are explicitly NON-CANDIDATE controls.
They expose why either flatness or RGB similarity alone can reward failure.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .runtime import configure_workspace, write_json, file_hash


def gray(a):
    return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)


def local_std(a):
    a=gray(a).astype(np.float32)
    mean=cv2.boxFilter(a,-1,(9,9),borderType=cv2.BORDER_REFLECT_101)
    var=cv2.boxFilter(a*a,-1,(9,9),borderType=cv2.BORDER_REFLECT_101)-mean*mean
    return np.sqrt(np.maximum(var,0))


def edge(a):
    return cv2.Canny(gray(a),40,100,L2gradient=True)>0


def measure(a,t,flat):
    e,et=edge(a),edge(t);kernel=np.ones((5,5),np.uint8)
    de=cv2.dilate(e.astype(np.uint8),kernel)>0;dt=cv2.dilate(et.astype(np.uint8),kernel)>0
    precision=float((e&dt).sum()/max(1,e.sum()));recall=float((et&de).sum()/max(1,et.sum()))
    return dict(rgb_mae=float(np.abs(a.astype(np.float32)-t).mean()),
        edge_precision_2px=precision,edge_recall_2px=recall,
        edge_f1_2px=2*precision*recall/max(1e-12,precision+recall),
        local_std_in_teacher_flat_regions=float(local_std(a)[flat].mean()) if flat.any() else None,
        output_edge_pixels=int(e.sum()))


def main():
    root=configure_workspace();out=root/'runs/boundary-budget-diagnostic-20261004/negative_controls';out.mkdir(exist_ok=False)
    manifest=json.loads((root/'runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'))
    records=[];sources={};panels=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for r in manifest['validation']:
        sp,tp=Path(r['source']),Path(r['target']);s,t=[np.asarray(Image.open(p).convert('RGB')) for p in (sp,tp)]
        assert s.shape==t.shape==(720,1280,3)
        sources[str(sp)]=file_hash(sp);sources[str(tp)]=file_hash(tp)
        # Solve only a global RGB affine transform; it cannot group surfaces.
        x=np.c_[s[::4,::4].reshape(-1,3).astype(np.float64),np.ones(57600)]
        coef=np.linalg.lstsq(x,t[::4,::4].reshape(-1,3),rcond=None)[0]
        affine=np.clip(np.rint(s.astype(np.float64)@coef[:3]+coef[3]),0,255).astype(np.uint8)
        flat=(local_std(t)<3)&(local_std(s)>6)
        flat[:16]=False;flat[-16:]=False;flat[:,:16]=False;flat[:,-16:]=False
        candidates=dict(source=s,teacher=t,blur_sigma2=cv2.GaussianBlur(s,(0,0),2),global_color_only=affine)
        for key,sub in [('failed_b1_2000','b1-cartoon-budget-20261004/train/validation_002000'),('failed_prior_400','b1-constructed-recovery-20261004/train/validation_000400')]:
            p=root/'runs'/sub/(r['id']+'.png');candidates[key]=np.asarray(Image.open(p).convert('RGB'));sources[str(p)]=file_hash(p)
        metrics={k:measure(v,t,flat) for k,v in candidates.items()}
        assert metrics['teacher']['rgb_mae']==0
        if edge(t).any():assert metrics['teacher']['edge_f1_2px']==1
        records.append(dict(id=r['id'],flat_region_pixels=int(flat.sum()),flat_fraction=float(flat.mean()),metrics=metrics))
        if r['id']=='val_stairs_090':
            for xy in [(424,320),(150,430)]:
                xx,yy=xy;panel=Image.new('RGB',(320*4,294),'#18202a');d=ImageDraw.Draw(panel)
                for i,(key,label) in enumerate([('source','原片'),('teacher','教师：区域与边界'),('blur_sigma2','纯模糊：失败对照'),('failed_b1_2000','本轮B1：用户已否决')]):
                    d.text((320*i+5,4),label,font=font,fill='white');panel.paste(Image.fromarray(candidates[key][yy:yy+264,xx:xx+320]),(320*i,30))
                panels.append(panel)
    canvas=Image.new('RGB',(1280,588))
    for i,p in enumerate(panels):canvas.paste(p,(0,294*i))
    canvas.save(out/'blur_vs_regions.png')
    write_json(out/'report.json',dict(status='DIAGNOSTIC_ONLY_NO_AUTOMATIC_ACCEPTANCE',records=records,source_hashes=sources,
        measurements='Native 720. Canny40/100 L2,2px match tolerance; flat mask teacher9x9 std<3 and source std>6.',
        limitations=['Threshold-sensitive observations, not calibrated style acceptance or perceptual ranking.',
                    'Blur can improve flatness; color can improve RGB error. Neither is cartoonization.',
                    'Sharp regions can still be wrong or flicker; full video is necessary after spatial acceptance.',
                    'Familiar regressions, not blind tests. No filtering of model outputs.'],
        candidates_rejected_by_user=['failed_b1_2000','failed_prior_400']))
    print(json.dumps([dict(id=r['id'],flat_pixels=r['flat_region_pixels'],metrics=r['metrics']) for r in records],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
