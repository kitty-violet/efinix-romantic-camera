"""CPU-only inspect known training samples, separate underfit from transfer."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .skip_filter_net import SkipFilterNet
from .train_luma_regions import ROOT,RUN,display
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .runtime import file_hash,write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    torch.set_num_threads(2);root=ROOT/'runs'/RUN;out=root/a.name;out.mkdir(exist_ok=False)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in s['binding']['code'].items())
    net=SkipFilterNet().eval();net.load_state_dict(s['model']);net.scales=s['scales']
    data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'));groups={}
    for r in data['pairs']:groups.setdefault(r['source'],[]).append(r)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);board=Image.new('RGB',(960,len(groups)*210),'#18202a');d=ImageDraw.Draw(board);records=[]
    for i,(source,rows) in enumerate(sorted(groups.items())):
        r=rows[len(rows)//2];x=as_tensor(read_image(r['current']))[None];start=time.perf_counter();y=to_rgb(display(net(x))[0]);t=read_image(r['current_target']);seconds=time.perf_counter()-start
        Image.fromarray(y).save(out/f'{i:02d}.png');records.append(dict(source=source,input=r['current'],input_sha256=file_hash(r['current']),target=r['current_target'],rgb_mae=float(np.abs(y.astype(float)-t).mean()),seconds=seconds))
        for j,(z,title) in enumerate([(read_image(r['current']),'训练原片'),(t,'训练目标'),(y,'网络输出')]):
            board.paste(Image.fromarray(z).resize((320,180)),(j*320,i*210+30));d.text((j*320+4,i*210+5),f'{i} '+title,font=font,fill='white')
        print('TRAINING_FIT_PROBE',i+1,seconds,flush=True)
    board.save(out/'comparison.png');write_json(out/'report.json',dict(status='TRAIN_SOURCE_FIT_DIAGNOSTIC',checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),records=records,not_blind=True))


if __name__=='__main__':main()
