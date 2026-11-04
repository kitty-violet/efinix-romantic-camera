"""Actual-byte whole-frame B1 review on the three named TRAIN diagnostics."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .b1_coherent_rgb_patch_profiles import coherent_profiles
from .audit_b1_rgb_edge_width import selected,measure
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('data','mining','base','checkpoint','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    specs=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[specs[i] for i in (1,5,6)]
    models={name:load(path)[1] for name,path in (('before',a.base),('after',a.checkpoint))}
    out=Path(a.out);out.mkdir(exist_ok=False);records=[]
    board=Image.new('RGB',(1280,1044),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,s in enumerate(specs):
        row=rows[s['row']['id']];src=m.t.r.read(row['path']);target=m.t.r.read(row['target']);labels=m.load_region(row['region'])
        xx,yy=s['xy'];w,h=s['wh'];box=(xx,yy,xx+w,yy+h)
        x=torch.from_numpy(src.transpose(2,0,1).copy())[None].float()/128-1
        t=torch.from_numpy(target.transpose(2,0,1).copy())[None].float()/128-1
        lab=torch.from_numpy(labels.astype(np.int64))[None];predictions={};scores={}
        for name,net in models.items():
            raw=net.export();assert len(raw)==20480
            nodes=net(x,all_nodes=True);actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            p=actual['rgb_head']/128;im=m.t.r.c.rgb_image(p[0]);im.save(out/f'{row["id"]}_{name}.png')
            predictions[name]=im;crop=lambda z:z[...,yy:yy+h,xx:xx+w]
            value,_=coherent_profiles(crop(p),crop(t),crop(lab))
            profiles=[];ar=np.asarray(im)
            for axis,ey,ex in selected(labels,target):
                ref=(target if axis==1 else target.transpose(1,0,2))[ey,ex-12:ex+12].astype(float)
                pred=(ar if axis==1 else ar.transpose(1,0,2))[ey,ex-12:ex+12].astype(float)
                profiles.append(measure(pred,ref))
            widths=[r['width10_90'] for r in profiles if r['width10_90'] is not None]
            scores[name]=dict(actual_all_nodes_error=0,crop_mae=float((crop(p)-crop(t)).abs().mean()*128),
                crop_coherent=float(value),whole_frame_straight_edge_width_median=float(np.median(widths)),
                low_contrast_fraction=float(np.mean([v['relative_contrast']<=.25 for v in profiles])))
        for j,(title,im) in enumerate((('真实原片',Image.fromarray(src)),('本轮固定目标',Image.fromarray(target)),
                                     ('此前实际B1',predictions['before']),('当前候选实际B1',predictions['after']))):
            board.paste(im.crop(box),(320*j,348*i+28));draw.text((320*j+3,348*i+3),title,font=font,fill='white')
        records.append(dict(id=row['id'],crop=s,full720_inference=True,training_scene=True,scores=scores))
        print('CRITICAL_ACTUAL_B1',row['id'],scores,flush=True)
    board.save(out/'critical_comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),records=records,base_sha256=file_hash(a.base),checkpoint_sha256=file_hash(a.checkpoint),
        source_sha256=file_hash(__file__),visual_acceptance=False,thin_structure_still_requires_visual_review=True))


if __name__=='__main__':main()
