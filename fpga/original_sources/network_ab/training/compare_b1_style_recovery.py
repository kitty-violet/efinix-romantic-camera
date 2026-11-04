"""Equal-source native720 style comparisons, including pre-expansion baselines."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_style_recovery as m
from . import train_b1_reconstruction as r
from .publish_b1_reconstruction_research import ActualBytes
from .audit_b1_reconstruction_research import phase_tests
from .engine import set_reproducible
from .runtime import configure_workspace,contract_guard,file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--phase',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(2026100585);contract_guard()
    if (m.OUT/'data.json').exists():rows=m.data()['validation']
    else:
        _,rows=r.data();old=json.loads((r.OUT.parent/'b1-deploy-preview-20261005-v1/data.json').read_text())
        for row in rows:
            target=next(x for x in old['validation'] if x['source_sha256']==row['sha256'])
            row.update(target=target['target'],target_sha256=target['target_sha256'])
    dest=Path(a.out);dest.mkdir(exist_ok=False)
    pictures={x['id']:[Image.fromarray(r.read(x['target']))] for x in rows};results={};labels=['认可风格目标']
    for spec in a.models:
        label,path=spec.split('=',1);path=Path(path)
        if path.suffix=='.bin':net=ActualBytes(path)
        else:
            state=torch.load(path,map_location='cpu',weights_only=True);net=r.initialize();net.load_state_dict(state['model'])
            net.quantized=bool(state.get('qat',False) or state.get('args',{}).get('qat',False))
        metrics=m.evaluate(net,rows)
        results[label]=dict(path=str(path),sha256=file_hash(path),validation=metrics)
        if a.phase:results[label]['phase']=phase_tests(net,rows)
        labels.append(label)
        for row in rows:
            im=r.c.rgb_image(net(r.tensor(r.read(row['path']))[None])[0][0]);im.save(dest/(row['id']+'_'+label+'.png'))
            pictures[row['id']].append(im)
        print('STYLE_COMPARE',label,metrics['mean'],flush=True)
        del net;torch.cuda.empty_cache()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    board=Image.new('RGB',(420*len(labels),314*len(r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,scene) in enumerate(r.c.CROPS):
        for j,(im,label) in enumerate(zip(pictures[key],labels)):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),scene+' | '+label,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'results.json',results)


if __name__=='__main__':main()
