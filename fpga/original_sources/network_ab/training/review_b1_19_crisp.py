"""Actual-byte before/after review, with explicit limitations of edge metrics."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_19_levels as n
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,write_json,file_hash


def luma(x):return (x[:,0:1]+2*x[:,1:2]+x[:,2:3])/4


def measure(p,t):
    lp,lt=luma(p),luma(t);values=[]
    for dim in (-1,-2):
        gp,gt=torch.diff(lp,dim=dim).abs(),torch.diff(lt,dim=dim).abs()
        near=F.max_pool2d(gp,5,1,2);mask=gt>12
        values.append(dict(strong_teacher_edges=int(mask.sum()),
            missing_edge_contrast=float(F.relu(gt-near)[mask].mean()),
            nearby_contrast_ratio=float((near[mask]/gt[mask].clamp_min(1)).clamp_max(2).mean())))
    chroma=lambda a:torch.cat(((a[:,0:1]-a[:,2:3])/4,(-a[:,0:1]+2*a[:,1:2]-a[:,2:3])/8),1)
    cp,ct=chroma(p),chroma(t);neutral=(ct.square().sum(1,keepdim=True)<25)&(lt>0)
    green=lambda a:a[:,1:2]-(a[:,0:1]+a[:,2:3])/2
    return dict(chroma_mae=float((cp-ct).abs().mean()),
        green_excess_relative_to_teacher=float((green(p)-green(t))[neutral].mean()) if neutral.any() else None,
        neutral_bright_pixels=int(neutral.sum()),edges=values)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True,type=Path);a=ap.parse_args()
    configure_workspace();data,selected,cache=c.load_data();out=a.package/'sharpness-review';out.mkdir(exist_ok=False)
    old=n.OUT/'qat/package/coefficients-padded.bin';new=a.package/'coefficients-padded.bin'
    nets=[c.IntegerB1(c.runtime_from_image(p.read_bytes())) for p in (old,new)]
    images={};records=[]
    for row in cache:
        if row['row']['split']=='train':continue
        src=row['source'][None]*128;t=row['target'][None]*128
        pred=[net(src) for net in nets]
        key=row['row']['id'];records.append(dict(id=key,before=measure(pred[0],t),after=measure(pred[1],t)))
        images[key]=[c.rgb_image(p[0]/128) for p in pred]+[c.rgb_image(t[0]/128)]
    board=Image.new('RGB',(1260,1248),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for row,(key,box,label) in enumerate(c.CROPS):
        for col,img in enumerate(images[key]):
            board.paste(img.crop(box),(420*col,312*row+32))
            d.text((420*col+4,312*row+4),('修改前：19层','修改后：19层','教师参考')[col],font=font,fill='white')
    board.save(out/'before_after_teacher.png')
    small=Image.new('RGB',(1260,624),'#17202a');small.paste(board.crop((0,312,1260,624)),(0,0));small.paste(board.crop((0,936,1260,1248)),(0,312));small.save(out/'building_people.png')
    write_json(out/'comparison.json',dict(before=str(old),after=str(new),before_sha256=file_hash(old),after_sha256=file_hash(new),
        reviewed_validation_not_blind=True,records=records,visual_accepted=False,
        metric_limit='Nearby edge contrast allows2px tolerance and is not proof of narrower or correctly shaped contours; signed green excess is only teacher-relative bright neutral regions.'))
    print('CRISP_REVIEW_COMPLETE',str(out),flush=True)
    for r in records:print(r['id'],r['before']['green_excess_relative_to_teacher'],r['after']['green_excess_relative_to_teacher'],r['before']['edges'],r['after']['edges'],flush=True)


if __name__=='__main__':main()
