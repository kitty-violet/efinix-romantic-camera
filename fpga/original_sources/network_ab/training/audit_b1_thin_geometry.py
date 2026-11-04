"""Test known thin shapes/corners through the actual fixed spatial decoder.

This injects target geometry, so is a representation diagnostic only. It
cannot be published as source prediction or a deployable network result.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,net=load(a.checkpoint);out=Path(a.out);out.mkdir(exist_ok=False)
    def render(code):
        truth=joint_interface_fields(code)[:,:14]
        nodes=fragment(net,F.pad(net.round(truth),(0,0,0,0,0,2)),'up1_low_pw')
        return nodes['up2_high_dw'][:,0]*2+16
    records=[]
    for orientation in ('horizontal','vertical','diagonal'):
        yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij')
        for width in (1,2,3,4,5,6,8,12):
            for phase in range(4):
                d=yy-48-phase if orientation=='horizontal' else xx-64-phase if orientation=='vertical' else (yy-48-(xx-64))/2**.5-phase
                mask=(d>=-width/2)&(d<width/2)
                code=torch.where(mask,2*81+40,12*81+40)[None]
                y=render(code)[0];contrast=(208-float(y[mask].min()))/160
                records.append(dict(orientation=orientation,width=width,phase=phase,retained_contrast=contrast,area_half_contrast=int((y<128).sum()),target_area=int(mask.sum())))
    d=m.load_data(a.data);byid={r['id']:r for r in d['training']};specs=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records']
    board=Image.new('RGB',(960,1050),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);rowrecords=[]
    for i,index in enumerate((1,5,6)):
        sp=specs[index];row=byid[sp['row']['id']];xx,yy=sp['xy'];w,h=sp['wh'];sl=np.s_[yy:yy+h,xx:xx+w]
        with np.load(row['native_luma_codes']) as z:code=torch.from_numpy(z['codes'][sl].astype(np.int64))[None]
        x=torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        target=(code//81*16+16).float();known=render(code);actual=net(x,all_nodes=True)['up2_high_dw'][:,0]*2+16
        images=[target,known,actual]
        for j,z in enumerate(images):
            im=Image.fromarray(z[0].clamp(0,255).byte().numpy(),'L').convert('RGB');board.paste(im,(320*j,350*i+30))
            draw.text((320*j+2,350*i+3),['原生目标亮度','已知几何·固定解码诊断','实际源图网络亮度'][j],font=font,fill='white')
        rowrecords.append(dict(id=row['id'],known_mae=float((known-target).abs().mean()),actual_mae=float((actual-target).abs().mean())))
    board.save(out/'known_geometry.png')
    write_json(out/'review.json',dict(records=records,train_rows=rowrecords,known_target_not_prediction=True,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))
    for width in (1,2,3,4,5,6,8,12):
        rows=[r for r in records if r['width']==width];print('THIN_GEOMETRY',width,min(r['retained_contrast'] for r in rows),max(r['retained_contrast'] for r in rows),flush=True)


if __name__=='__main__':main()
