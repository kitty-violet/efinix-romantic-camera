"""Select and bind the tested reconstruction bytes after the loss control."""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from . import train_b1_reconstruction as r
from . import train_b1_learned_colour as c
from .b1_trained_image import runtime_from_image,decode
from .probe_b1_cartoon_int8 import IntegerB1
from .engine import set_reproducible
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(r.SEED)
    _,val=r.data();paths={'mse':r.OUT/'package','source_gradient':r.OUT/'source-gradient-control/package'}
    stats={}
    for label,path in paths.items():
        review=json.loads((path/'review.json').read_text(encoding='utf-8'))
        assert review['independent_oracle_error']==0 and all(x['export_node_error']==0 for x in review['records'])
        mean=sum(x['mse_rgb255'] for x in review['records'])/len(review['records'])
        stats[label]=dict(psnr=10*math.log10(255**2/mean),mse=mean,
            edge_mae=sum(x['edge_mae'] for x in review['records'])/len(review['records']),
            mae=sum(x['mae'] for x in review['records'])/len(review['records']))
    chosen=max(stats,key=lambda x:stats[x]['psnr'])
    delivery=r.OUT/'delivery';delivery.mkdir(exist_ok=False)
    for name in ('coefficients-padded.bin','load_words.hex','review.json'):
        shutil.copy2(paths[chosen]/name,delivery/name)
    modelroot=r.OUT if chosen=='mse' else r.OUT/'source-gradient-control'
    for stage in ('fp32','qat'):
        shutil.copy2(modelroot/stage/'best.pt',delivery/(stage+'_best.pt'))
    raw=(delivery/'coefficients-padded.bin').read_bytes();decode(raw)
    restored=b''.join(int(line,16).to_bytes(8,'little') for line in (delivery/'load_words.hex').read_text().splitlines())
    assert restored==raw
    malformed={}
    for key,damaged in [('length',raw[:-1]),('crc',raw[:1300]+bytes([raw[1300]^1])+raw[1301:]),('padding',raw[:-1]+b'\x01')]:
        try:decode(damaged)
        except (ValueError,AssertionError):malformed[key]='rejected'
        else:raise AssertionError(key)
    out=r.OUT/'final-review';out.mkdir(exist_ok=False)
    net=IntegerB1(runtime_from_image(raw))
    imgs={}
    for row in val:
        a=r.read(row['path']);x=r.tensor(a)[None]
        pred=c.rgb_image(net(x*128)[0]/128)
        original=Image.fromarray(a)
        original.save(out/(row['id']+'_source720.png'));pred.save(out/(row['id']+'_integer.png'))
        imgs[row['id']]=[original,pred]
    board=Image.new('RGB',(840,314*len(c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for i,(key,box,label) in enumerate(c.CROPS):
        for j,im in enumerate(imgs[key]):
            board.paste(im.crop(box),(420*j,314*i+34))
            draw.text((420*j+4,314*i+5),label+' | '+('原图' if j==0 else '固定B1实际INT8重建'),font=font,fill='white')
    board.save(out/'comparison.png')
    path=r.OUT/'watch.html'
    text=path.read_text(encoding='utf-8').replace('src="review/','src="final-review/')
    path.write_text(text,encoding='utf-8')
    data=dict(selected=chosen,selection_rule='highest source-disjoint reviewed-set pooled reconstruction PSNR',
        candidates=stats,weights=str(delivery/'coefficients-padded.bin'),package_sha256=file_hash(delivery/'coefficients-padded.bin'),
        transport_bytes=len(raw),core_bytes=17792,hex_roundtrip=True,malformed=malformed,
        hardware_changed=False,actual_board=False,weights_kind='original-image reconstruction, NOT style weights',
        limited_training_not_mathematical_capacity_bound=True)
    write_json(r.OUT/'delivery.json',data)
    print('RECON_DELIVERY_SELECTED',json.dumps(data,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
