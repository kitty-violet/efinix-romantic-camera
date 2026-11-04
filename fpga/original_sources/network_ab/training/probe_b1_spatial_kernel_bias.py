"""Coefficient-only probe of late spatial texture sensitivity.

Blend each existing up1_low_dw integer kernel towards a spatial box with the
EXACT same integer DC sum and bias. This is not a displayed-image blur or new
operation: each preview runs the exported byte package with hard colour levels.
It does not guarantee minimum areas or improvement on every spatial frequency.
"""
import argparse
import copy
import hashlib
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import write_json,file_hash


def integer_kernel_blend(code,alpha):
    flat=np.asarray(code,np.int64).reshape(-1,9)
    target=(1-alpha)*flat+alpha*flat.sum(1,keepdims=True)/9
    result=np.floor(target).astype(np.int64)
    remainder=flat.sum(1)-result.sum(1)
    for row,n in enumerate(remainder):
        if n:result[row,np.argsort(-(target[row]-result[row]),kind='stable')[:n]]+=1
    assert np.array_equal(result.sum(1),flat.sum(1))
    assert np.abs(result).max()<=127
    return result.reshape(np.asarray(code).shape)


def regression():
    rng=np.random.default_rng(19);x=rng.integers(-127,128,(1024,1,3,3))
    assert np.array_equal(integer_kernel_blend(x,0),x)
    for alpha in (.25,.6,1.):
        out=integer_kernel_blend(x,alpha)
        assert np.array_equal(out.sum((1,2,3)),x.sum((1,2,3)))
    print('SPATIAL_KERNEL_DC_REGRESSION PASS',flush=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();regression();torch.set_num_threads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    original=copy.deepcopy(net.state_dict());dest=Path(a.out);dest.mkdir(exist_ok=False)
    rows=m.load_data()['validation'];images={};records=[]
    for alpha in (0.,.25,.6):
        net.load_state_dict(original);key='up1_low_dw'
        scale=(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]
        old=net.round(net.weight[key]/scale).clamp(-127,127).cpu().numpy().astype(np.int64)
        new=integer_kernel_blend(old,alpha)
        if alpha:net.weight[key].copy_(torch.from_numpy(new).float()*scale)
        folder=dest/('original' if alpha==0 else f'blend{int(alpha*100):02d}');folder.mkdir()
        raw=net.export();(folder/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');errors=[]
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_=actual(x,True);reference=net(x/128,all_nodes=True)
            error=max(float((nodes[k]-reference[k]).abs().max()) for k in nodes);assert error==0
            decision=nodes['up2_low_dw'][:,:15];assert ((decision==0)|(decision==127)).all()
            im=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);images[row['id'],alpha]=im
            im.save(folder/(row['id']+'_actual_B1.png'));errors.append(dict(id=row['id'],all_node_error=error))
        if alpha:
            save={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
            save.update(model=copy.deepcopy(net.state_dict()),step=0,manual_kernel_blend=alpha,
                exact_integer_dc_preserved=True,parent=a.checkpoint,requires_retraining=True)
            torch.save(save,folder/'initial.pt')
        records.append(dict(alpha=alpha,package_sha256=hashlib.sha256(raw).hexdigest(),records=errors,
            changed_integer_weights=int((old!=new).sum()),dc_preserved=True))
        print('SPATIAL_KERNEL_PROBE',alpha,'six full frames all nodes=0',flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by_id[key]['target'])]+[images[key,v] for v in (0.,.25,.6)]
        for j,(im,title) in enumerate(zip(pics,('区域目标','原真实B1','权重内空间整理25%','权重内空间整理60%'))):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.checkpoint,parent_sha256=file_hash(a.checkpoint),records=records,
        manually_modified_coefficients=True,postfilter=False,added_operators=0,
        global_minimum_area_guaranteed=False,visual_acceptance=False,actual_board=False))


if __name__=='__main__':main()
