"""TRAIN-only bounded colour aggregation reference; not model inference.

The four overlapping quadrants share one RGB-variance decision, avoiding
independent colour-channel boundaries. This is the classical rectangular
Kuwahara construction, not anisotropic Kuwahara or a deployable B1 operator.
Existing fixed labels are read-only. A candidate must pass visual review
before it can be considered as supervision.
"""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def bounded_quadrants(rgb,radius):
    z=rgb.astype(np.float64);means=[];variances=[]
    n=radius+1
    for anchor in ((0,0),(radius,0),(0,radius),(radius,radius)):
        mean=cv2.boxFilter(z,-1,(n,n),anchor=anchor,borderType=cv2.BORDER_REPLICATE)
        second=cv2.boxFilter(z*z,-1,(n,n),anchor=anchor,borderType=cv2.BORDER_REPLICATE)
        means.append(mean);variances.append(np.maximum(second-mean*mean,0).mean(2))
    means=np.stack(means);variance=np.stack(variances)
    # Float cancellation on uniform areas must not determine a checkerboard
    # tie-break. Treat errors below 1e-8 byte^2 as an exact tie.
    variance[variance<1e-8]=0
    choice=variance.argmin(0);y,x=np.indices(rgb.shape[:2])
    return means[choice,y,x].round().clip(0,255).astype(np.uint8)


def regression():
    colour=np.array([71,124,189],np.uint8);z=np.broadcast_to(colour,(96,112,3)).copy()
    for r in (3,7,11):
        assert np.array_equal(bounded_quadrants(z,r),z)
        edge=z.copy();edge[:,56:]=[204,93,51]
        assert np.array_equal(bounded_quadrants(edge,r),edge)
        altered=z.copy();altered[:30]=0;altered[66:]=255
        assert np.array_equal(bounded_quadrants(altered,r)[42:54],z[42:54])
    print('BOUNDED_QUADRANTS_PASS constant/joint_RGB_step/local_support',flush=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();cv2.setNumThreads(2);regression();out=Path(a.out);out.mkdir(exist_ok=False)
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    mined=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[mined[i] for i in (1,5,6)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);board=Image.new('RGB',(1600,1044),'#17202a');draw=ImageDraw.Draw(board)
    records=[]
    for i,s in enumerate(specs):
        row=byid[s['row']['id']];parent=m.t.r.read(row['parent_target']);target=m.t.r.read(row['target'])
        xx,yy=s['xy'];w,h=s['wh'];variants=[('现有目标',target),('区域合并前父图',parent)]+[(f'局部RGB方差 r{r}·仅参照',bounded_quadrants(parent,r)) for r in (3,7,11)]
        for j,(name,rgb) in enumerate(variants):
            Image.fromarray(rgb).save(out/(row['id']+f'_{j}.png'))
            board.paste(Image.fromarray(rgb[yy:yy+h,xx:xx+w]),(j*320,i*348+28));draw.text((j*320+3,i*348+3),name,font=font,fill='white')
        records.append(dict(id=row['id'],parent_sha256=file_hash(row['parent_target']),target_sha256=row['target_sha256']))
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(records=records,scope='local aggregation relative to supplied parent RGB only; full parent source RF not bounded here',
        no_original_target_modified=True,not_network_output=True,visual_acceptance=False,source_sha256=file_hash(__file__)))


if __name__=='__main__':main()
