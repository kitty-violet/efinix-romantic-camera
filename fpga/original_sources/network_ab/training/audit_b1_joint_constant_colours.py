"""Audit actual constant binary-field colours and inject TRAIN region codes.

Not a model output: target regions select diagnostic field patterns. No network
parameter is optimized. This separates colour coverage from encoder learning.
"""
import argparse
import copy
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .probe_b1_joint_decoder_latent import IDS, BOX, MARGIN
from .runtime import configure_workspace, contract_guard, write_json, file_hash


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='joint-constant-colour-audit');args=ap.parse_args()
    configure_workspace(); contract_guard(); torch.set_num_threads(2)
    dest=m.OUT/args.name; dest.mkdir(exist_ok=False)
    path=m.OUT/'joint-field-colour-qat/candidate_000500.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
    raw=net.export();bits=((torch.arange(32768)[:,None]>>torch.arange(15)[None])&1).float()
    palette=[]
    for bit in bits.split(512):
        z=torch.zeros(len(bit),16,1,1);z[:,:15,0,0]=bit*127
        p=fragment(net,z,'up1_low_pw')['rgb_head']
        assert torch.equal(p,p[:,:,:1,:1].expand_as(p))
        palette.append(p[:,:,0,0])
    palette=torch.cat(palette)
    data=m.load_data(m.OUT/'hierarchical-targets-v2/data.json');rows={r['id']:r for r in data['training']}
    pictures=[];records=[];x0,y0,x1,y1=BOX
    for key in IDS:
        row=rows[key];lab=m.load_region(row['region'])[y0:y1,x0:x1]
        target=m.t.r.read(row['target'])[y0:y1,x0:x1];source=m.t.r.read(row['path'])[y0:y1,x0:x1]
        labels,index=np.unique(lab.ravel(),return_inverse=True);count=np.bincount(index)
        rgb=np.stack([np.bincount(index,weights=target[:,:,c].ravel())/count for c in range(3)],1).astype(np.float32)
        centres=torch.tensor(rgb-128);distance=(centres[:,None]-palette[None]).square().mean(2)
        codes=distance.argmin(1);colour=palette[codes]+128
        assigned=colour[index].reshape(*lab.shape,3).numpy().round().clip(0,255).astype(np.uint8)
        # Same code everywhere in a known region. Sampling a categorical label
        # at the 4x4 cell centre avoids averaging incompatible region codes.
        grid=torch.tensor(index.reshape(lab.shape)[2::4,2::4].copy())
        z=torch.zeros(1,16,*grid.shape)
        z[0,:15]=bits[codes[grid]].permute(2,0,1)*127
        injected=fragment(net,z,'up1_low_pw')['rgb_head']/128
        src=torch.tensor(source.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        actual=net(src,all_nodes=True)['rgb_head']/128
        output=m.t.r.c.rgb_image(injected[0]);output.save(dest/(key+'_injected.png'))
        pictures.append([Image.fromarray(source),Image.fromarray(target),Image.fromarray(assigned),m.t.r.c.rgb_image(actual[0]),output])
        error=np.abs(assigned.astype(np.float32)-target)
        records.append(dict(id=key,constant_palette_mae=float(error.mean()),
            region_mean_rgb_mae=float(((colour-centres-128).abs().mean(1)*torch.tensor(count)).sum()/count.sum()),
            region_mean_rgb_max_error=float((colour-centres-128).abs().max()),
            used_region_codes=int(codes.unique().numel()),target_regions=len(labels)))
    assert net.export()==raw
    size=384-2*MARGIN;board=Image.new('RGB',(size*5,(size+40)*len(IDS)),'#17202a')
    draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    titles=['训练原图','修订目标','可达色·保留目标轮廓','真实网络预测','目标区域编码注入·非预测']
    for i,imgs in enumerate(pictures):
        for j,(im,title) in enumerate(zip(imgs,titles)):
            board.paste(im.crop((MARGIN,MARGIN,384-MARGIN,384-MARGIN)),(j*size,i*(size+40)+40))
            draw.text((j*size+4,i*(size+40)+5),title,font=font,fill='white')
    board.save(dest/'comparison.png')
    result=dict(patterns=32768,unique_constant_rgb=int(palette.unique(dim=0).shape[0]),records=records,
        checkpoint_sha256=file_hash(path),code_sha256=file_hash(__file__),coefficients_unchanged=True,
        source_prediction=False,target_masks_injected=True,training_only=True,deployable=False,visual_acceptance=False)
    write_json(dest/'results.json',result);print('CONSTANT_COLOUR_AUDIT',result,flush=True)


if __name__=='__main__':main()
