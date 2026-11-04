"""Read-only real-byte six-frame review, independent of local TRAIN fitting."""
import argparse,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);raw=net.export();assert len(raw)==20480
    (out/'coefficients-padded.bin').write_bytes(raw)
    rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');data=m.load_data(a.data)
    images={};records=[]
    for row in data['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
        nodes=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
        p=actual['rgb_head']/128;im=m.t.r.c.rgb_image(p[0]);im.save(out/(row['id']+'_actual720.png'));images[row['id']]=im
        d=(p-t)*128;rec=dict(id=row['id'],actual_all_nodes_error=0,mae=float(d.abs().mean()),rgb_bias=d.mean((0,2,3)).tolist())
        records.append(rec);print('FULL_COLOUR_REVIEW',rec,flush=True)
    rows={r['id']:r for r in data['validation']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(title,im) in enumerate((('原片',Image.open(rows[rid]['path'])),('固定目标',Image.open(rows[rid]['target'])),('真实B1整数·整图后裁',images[rid]))):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),records=records,checkpoint_sha256=file_hash(a.checkpoint),
        package_sha256=file_hash(out/'coefficients-padded.bin'),data_sha256=file_hash(a.data),
        contract_sha256=state['contract_sha256'],source_sha256=file_hash(__file__),full720_not_crop_inference=True,
        dataset_split='historically seen validation; not claimed blind',visual_acceptance=False,video_acceptance=False))


if __name__=='__main__':main()
