"""Diagnostic only: isolate high-frequency information in the native skip.

The filters here are interventions, not deployable new model operations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .context_region_net import ContextNet
from .train_luma_regions import display,ROOT,RUN
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def forward(net,x,k):
    values={'input':x}
    for n in net.nodes:
        a=values[n['inputs'][0]]
        if n['kind']=='nearest_x2':v=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
        elif n['kind']=='residual_add':
            b=values[n['inputs'][1]]
            if n['id']=='mix_add' and k>1:
                b=F.avg_pool2d(F.pad(b,(k//2,)*4,mode='replicate'),k,stride=1)
            v=a+b
        else:v=net.layers[n['id'].replace('.','__')](a)
        values[n['id']]=v
    return values['rgb']


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100433)
    root=ROOT/'runs'/RUN;out=root/a.name;out.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert state['scales'] is None and all(file_hash(k)==v for k,v in state['binding']['code'].items())
    net=ContextNet().cuda().eval();net.load_state_dict(state['model'])
    data=json.loads((root/'luma/data.json').read_text(encoding='utf-8'));scores=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].cuda()
        base=net(x);assert (base-forward(net,x,1)).abs().max()==0
        for k in [1,3,5,9]:
            y=to_rgb(display(forward(net,x,k))[0]);Image.fromarray(y).save(out/f'{r["id"]}_k{k}.png')
            target=read_image(root/'detail_a'/(r['id']+'.png'))
            scores.append(dict(id=r['id'],k=k,target_rgb_mae=float(np.abs(y.astype(float)-target).mean())))
        print('SKIP_PROBE',r['id'],flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1600,894),'#18202a');d=ImageDraw.Draw(board)
    for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        x,y=xy;paths=[root/'detail_a'/(key+'.png')]+[out/f'{key}_k{k}.png' for k in [1,3,5,9]]
        for i,(path,title) in enumerate(zip(paths,['目标','原网络','仅跳连3x3','仅跳连5x5','仅跳连9x9'])):
            d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png')
    write_json(out/'report.json',dict(status='CAUSAL_DIAGNOSTIC_NOT_DEPLOYABLE',checkpoint=str(a.checkpoint),sha256=file_hash(a.checkpoint),records=scores))

if __name__=='__main__':main()
