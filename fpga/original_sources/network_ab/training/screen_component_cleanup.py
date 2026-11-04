"""Compare explicit spatial operators on signed C3 before palette decode.

These are new hardware operations, not concealed improvements to CNN weights.
No resize, temporal blending, or source-detail paste is used.
"""
import json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .screen_integer_diffusion import diffuse
from .train_luma_regions import ROOT,RUN,LEVELS
from .runtime import write_json,file_hash


def decode_components(a):
    c=a.astype(np.int32)-128;c[:,:,0]+=128
    y,co,cg=[np.asarray(levels)[np.abs(c[:,:,i,None]-np.asarray(levels)).argmin(-1)] for i,levels in enumerate(LEVELS)]
    return np.stack((y+co-cg,y+cg,y-co-cg),-1).clip(0,255).astype(np.uint8)


def cleanup(a,method):
    if method=='none':return a
    if method.startswith('diff'):
        _,t,n=method.split('_');return diffuse(a,int(t),int(n)).astype(np.uint8)
    _,k,n=method.split('_')
    for _ in range(int(n)):a=cv2.medianBlur(a,int(k))
    return a


def main():
    cv2.setNumThreads(2);base=ROOT/'runs'/RUN;out=base/'component-cleanup-screen-v2';out.mkdir(exist_ok=False)
    data=json.loads((base/'component-regions-fp32/data.json').read_text(encoding='utf-8'));src=base/'component-centers-qat1500-integer'
    methods=['none','diff_4_4','diff_8_8','median_3_2','median_5_2'];rows=[]
    for r in data['validation']:
        p=src/(r['id']+'_components.npy');raw=np.load(p);assert raw.shape==(720,1280,3)
        assert raw.min()>=-128 and raw.max()<=127
        a=(raw.astype(np.int32)+128).astype(np.uint8)
        assert np.array_equal(decode_components(a),np.asarray(Image.open(src/(r['id']+'.png'))))
        for method in methods:
            start=time.perf_counter();v=decode_components(cleanup(a,method));Image.fromarray(v).save(out/f'{r["id"]}_{method}.png')
            rows.append(dict(id=r['id'],source=str(p),source_sha256=file_hash(p),method=method,seconds=time.perf_counter()-start))
        print('C3_CLEANUP_SCREEN',r['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(2240,1192),'#18202a');draw=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
        r=next(r for r in data['validation'] if r['id']==key)
        paths=[(r['source'],'原片'),(base/'intermediate-target-screen'/f'{key}_0.008.png','区域目标')]+[(out/f'{key}_{m}.png',m) for m in methods]
        for i,(path,title) in enumerate(paths):draw.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='SPATIAL_OPERATOR_SCREEN_ONLY',records=rows,operator_source_sha256=file_hash(__file__),new_hardware_required=True,rtl=False,fit=False,temporal_acceptance=False,style_accepted=False))


if __name__=='__main__':main()
