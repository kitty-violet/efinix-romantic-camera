"""Intervene on colour/contour fields to locate errors; NOT deliverable images."""
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_two_colour_geometry import latent
from .probe_b1_decoder_coverage import fragment
from .train_b1_region_formation import formation_loss
from .runtime import configure_workspace, write_json, file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model'])
    net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    rows=m.load_data()['validation'];records=[];images={}
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.from_numpy(m.load_region(row['region']).astype('int64'))[None].cuda()
        nodes=net(x,all_nodes=True);z=nodes['up1_low_pw'];truth=torch.from_numpy(latent(row['target'],row['region'])).cuda()[None].round().clamp(0,127)
        replay=fragment(net,z,'up1_low_pw')['rgb_head'];assert torch.equal(replay,nodes['rgb_head'])
        ims=[Image.fromarray(m.t.r.read(row['target']))]
        for name in ('actual','target_colour_only','target_contour_only','both_target'):
            code=z.clone()
            if name in ('target_colour_only','both_target'):code[:,:6]=truth[:,:6]
            if name in ('target_contour_only','both_target'):code[:,6:7]=truth[:,6:7]
            q=fragment(net,code,'up1_low_pw')['rgb_head']/128
            _,values=formation_loss(q,target,labels)
            records.append(dict(id=row['id'],intervention=name,mae=float((q-target).abs().mean()*128),**{k:float(v) for k,v in values.items()}))
            im=m.t.r.c.rgb_image(q[0]);ims.append(im)
        images[row['id']]=ims
    board=Image.new('RGB',(2100,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    titles=('离线目标','实际小网络输出','诊断：只注入目标两侧颜色','诊断：只注入目标轮廓','诊断：两者均注入，非预测')
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],titles)):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+6),label+' | '+title,font=font,fill='white')
    board.save(dest/'diagnostic.png')
    means={name:{k:sum(v[k] for v in records if v['intervention']==name)/len(rows) for k in records[0] if k not in ('id','intervention')} for name in ('actual','target_colour_only','target_contour_only','both_target')}
    write_json(dest/'results.json',dict(warm_sha256=file_hash(a.warm),records=records,means=means,fragment_replay_error=0,target_injection_diagnostic=True,visual_acceptance=False))
    print('TWO_COLOUR_PATH_AUDIT',means,flush=True)


if __name__=='__main__':main()
