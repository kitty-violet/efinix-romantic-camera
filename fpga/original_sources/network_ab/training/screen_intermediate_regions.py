"""Check 'slightly finer' between the old region algorithm and detail_a."""
import json,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .region_mechanism_probe import gaussian,l0_surface,outline
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json


@torch.no_grad()
def target(x,lam):
    x=gaussian(x,.65).clamp_min(0).pow(.65)
    return outline(l0_surface(x,lam).clamp(0,1).pow(1/.65),strength=.4)


def main():
    torch.set_num_threads(2);root=ROOT/'runs'/RUN;out=root/'intermediate-target-screen';out.mkdir(exist_ok=False)
    data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'));records=[]
    for r in data['validation']:
        x=torch.from_numpy(np.asarray(Image.open(r['source']).convert('RGB')).copy()).permute(2,0,1)[None].float()/255
        for lam in [.008,.012]:
            start=time.perf_counter();y=(target(x,lam)[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
            Image.fromarray(y).save(out/f'{r["id"]}_{lam}.png');records.append(dict(id=r['id'],lambda_value=lam,seconds=time.perf_counter()-start))
        print('INTERMEDIATE_TARGET',r['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1280,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        x,y=xy;paths=[ROOT/'runs/region-mechanisms-20261004/algorithm'/(key+'.png'),out/f'{key}_0.012.png',out/f'{key}_0.008.png',root/'detail_a'/(key+'.png')]
        for i,(path,title) in enumerate(zip(paths,['上版区域算法','中间目标 .012','中间目标 .008','本轮原目标 .004'])):
            d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='CPU_TARGET_SCREEN_NOT_INFERENCE',source_sha256=file_hash(__file__),records=records))

if __name__=='__main__':main()
