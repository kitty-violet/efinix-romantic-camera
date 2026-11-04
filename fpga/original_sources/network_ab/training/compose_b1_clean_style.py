"""Compose fixed colour/large-region contour rules into B1 coefficient rows."""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_sparse_outlines import outline_head
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.no_grad()
def tone(net):
    w=net.weight['rgb_head'];b=net.bias['rgb_head'];y=w.new_tensor([.299,.587,.114])
    matrix=1.08*(1.2*torch.eye(3,device=w.device)-.2*torch.ones(3,1,device=w.device)*y[None])
    cool=w.new_tensor([-2.,1.,8.])
    matrix=matrix-cool[:,None]*y[None]/127
    net.weight['rgb_head'].copy_(torch.einsum('oc,cihw->oihw',matrix,w))
    net.bias['rgb_head'].copy_(matrix@b+cool)
    scale=(w.abs().flatten(1).amax(1).clamp_min(1e-5)/112)
    net.rgb_head_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006231);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];initial=copy.deepcopy(net.state_dict());data=m.load_data()
    images={r['id']:[Image.fromarray(m.t.r.read(r['target']))] for r in data['validation']}
    for name in ('current','colour','colour-outline'):
        net.load_state_dict(initial)
        if name=='colour-outline':outline_head(net,3,threshold=1,retain_aa=True)
        if name!='current':tone(net)
        out=dest/name;out.mkdir();raw=net.export();(out/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw));records=[]
        for row in data['validation']:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None]
            ours=net(x,all_nodes=True);got,_=actual(x*128,True)
            error=max(float((z-got[k]).abs().max()) for k,z in ours.items());assert error==0
            im=m.t.r.c.rgb_image(got['rgb_head'][0]/128);im.save(out/(row['id']+'.png'));images[row['id']].append(im)
            records.append(dict(id=row['id'],node_error=error));del ours,got
        payload={k:v for k,v in state.items() if k in ('residual_encoding','field_node','fixed_sampling','classifier_mode')}
        payload.update(model=net.state_dict(),qat=True,parent=a.warm,step=state.get('step'),
            coverage_layout='3Y_2Co_2Cg_1outline' if name=='colour-outline' else '4Y_2Co_2Cg',
            fixed_tone=name!='current',outline_amount=3 if name=='colour-outline' else 0)
        torch.save(payload,out/'initial.pt');write_json(out/'review.json',dict(records=records,postfilter=False,
            fixed_rules_inside_coefficients=True,operators_added=0,independent_oracle_pending=True,visual_acceptance=False))
        print('CLEAN_STYLE_COMPOSED',name,flush=True);del actual;torch.cuda.empty_cache()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,title in enumerate(('离线目标','实际小网络','色彩系数修正','色彩＋大色块边界')):
            board.paste(images[key][j].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    for p in (Path(__file__),Path(__file__).with_name('probe_b1_sparse_outlines.py')):shutil.copy2(p,dest/p.name)
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),new_training=False,
        tone=dict(contrast=1.08,saturation=1.2,shadow_cool_rgb=[-2,1,8],neutral_white_preserved=True),
        outline_source='simplified luminance regions only',model_graph_unchanged=True,visual_acceptance=False))


if __name__=='__main__':main()
