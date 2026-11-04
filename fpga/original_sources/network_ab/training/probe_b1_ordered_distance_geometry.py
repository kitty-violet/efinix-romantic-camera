"""Compare area and native signed-distance fields in the fixed radix decoder.

This deliberately re-tests distance fields after the exact-complement and
30 ordered-plane changes. Earlier15-plane/200-colour SDF failures still stand.
Ground-truth injection is a representation diagnostic, not source inference.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits, GROUPS30
from .train_b1_ternary_regions import region_codes
from .audit_b1_binary_boundary_phase import contour_metrics
from .runtime import file_hash, write_json
from . import train_b1_minimum_area as m


def distance_fields(code, gain):
    """Native-pixel-centre distance, then the actual4x4 footprint average.

    Clip after averaging. Nested cumulative masks retain their ordering.
    Constant predicates explicitly reach the exact0/127 endpoints.
    """
    masks=code_bits(code).cpu().numpy(); result=[]
    for example in masks:
        fields=[]
        for bit in example:
            bit=bit.astype(np.uint8)
            if bit.all(): d=np.full(bit.shape,128/gain,np.float32)
            elif not bit.any(): d=np.full(bit.shape,-128/gain,np.float32)
            else:
                d=cv2.distanceTransform(bit,cv2.DIST_L2,5)-cv2.distanceTransform(1-bit,cv2.DIST_L2,5)
                d-=.5*np.sign(d)
            fields.append(d)
        result.append(np.stack(fields))
    d=torch.from_numpy(np.stack(result)).to(code.device)
    return (63.5+gain*F.avg_pool2d(d,4)).clamp(0,127)


def fields(code, mode):
    mode=mode.split('-')[0]
    return F.avg_pool2d(code_bits(code)*127,4) if mode=='area' else distance_fields(code,float(mode[3:]))


@torch.no_grad()
def set_ramp(net, gain):
    assert gain%2==1 and 1<=gain<=127
    key='up2_low_dw';mult=131071
    net.weight[key].zero_();net.weight[key][:15,0,1,1]=gain*mult/131072
    net.bias[key].zero_();net.bias[key][:15]=-63.5*(gain-1)*mult/131072
    getattr(net,key+'_m').fill_(mult)


def regression():
    c=torch.arange(1215)[:,None,None].expand(-1,4,4)
    for gain in (4,8,16):
        f=distance_fields(c,gain)
        assert torch.equal(f,code_bits(c)[:,:,:1,:1]*127)
    c=torch.zeros(4,32,32,dtype=torch.long)
    for i in range(4): c[i,:,12+i:]=1214
    z=distance_fields(c,8)
    assert all(not torch.equal(z[i],z[i+1]) for i in range(3))
    for lo,hi in GROUPS30: assert (z[:,lo:hi-1]>=z[:,lo+1:hi]).all()
    inv=distance_fields(1214-c,8)
    assert (z+inv-127).abs().max()<1e-4
    print('ORDERED_DISTANCE_GEOMETRY_PASS 1215constant/order/complement/native_phase',flush=True)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--refine-ramp',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1);regression()
    state,net=load(a.checkpoint);raw=net.export();dest=Path(a.out);dest.mkdir(exist_ok=False)
    setup=state['binary_coverage720'];mat=torch.tensor(setup['matrix']).float();bias=torch.tensor(setup['bias']).float()
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    pk=tuple(map(tuple,(lev@mat.T+bias+128).clamp(0,255).tolist()))
    modes=('area-g7','sdf4-g15','sdf8-g11','sdf8-g15') if a.refine_ramp else ('area','sdf4','sdf8','sdf16');phase=[];shots={}
    constant=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    expected=fragment(net,constant,'up1_low_dw')['rgb_head'].clone()
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij')
    for mode in modes:
        set_ramp(net,int(mode.split('-g')[-1]) if '-g' in mode else 7)
        assert torch.equal(fragment(net,constant,'up1_low_dw')['rgb_head'],expected)
        for pair,(ca,cb) in enumerate((((5,4,4),(9,4,4)),((5,3,4),(6,6,3)),((2,4,5),(11,4,3)))):
            aa=(mat@torch.tensor(ca).float()+bias).clamp(-128,127);bb=(mat@torch.tensor(cb).float()+bias).clamp(-128,127)
            for slope in (.2,.5,1.):
                for p in range(4):
                    truth_x=64+slope*(torch.arange(96)-48)+p-.5
                    codes=torch.where(xx>truth_x[:,None],cb[0]*81+cb[1]*9+cb[2],ca[0]*81+ca[1]*9+ca[2])[None]
                    z=F.pad(net.round(fields(codes,mode)),(0,0,0,0,0,2))
                    rgb=fragment(net,z,'up1_low_dw')['rgb_head'][0]
                    metrics=contour_metrics(rgb,aa,bb,truth_x)
                    low,high=torch.minimum(aa,bb)[:,None,None],torch.maximum(aa,bb)[:,None,None]
                    metrics['component_overshoot']=float(torch.maximum(F.relu(rgb-high),F.relu(low-rgb)).max())
                    phase.append(dict(mode=mode,pair=pair,slope=slope,phase=p,**metrics))
        summary={k:float(np.mean([r[k] for r in phase if r['mode']==mode])) for k in metrics}
        print('ORDERED_DISTANCE_PHASE',mode,summary,flush=True)
    rows={r['id']:r for r in m.load_data(a.data)['validation']};records=[]
    for rid in dict.fromkeys(r[0] for r in m.t.r.c.CROPS):
        row=rows[rid];code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy()).float()-128
        for mode in modes:
            set_ramp(net,int(mode.split('-g')[-1]) if '-g' in mode else 7)
            f=fields(code,mode)
            for lo,hi in GROUPS30: assert (f[:,lo:hi-1]>=f[:,lo+1:hi]-1e-4).all()
            z=F.pad(net.round(f),(0,0,0,0,0,2));rgb=fragment(net,z,'up1_low_dw')['rgb_head'][0]
            im=m.t.r.c.rgb_image(rgb/128);im.save(dest/f'{rid}_{mode}.png');shots[rid,mode]=im
            records.append(dict(id=rid,mode=mode,mae=float((rgb-target).abs().mean())))
    board=Image.new('RGB',(2100,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(name,im) in enumerate([('original target',Image.open(rows[rid]['target']))]+[(mode+' target injection',shots[rid,mode]) for mode in modes]):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+3),label+' | '+name,font=font,fill='white')
    board.save(dest/'comparison.png');net.load_state_dict(state['model']);assert net.export()==raw
    write_json(dest/'review.json',dict(source_prediction=False,not_deployable=True,coefficients_unchanged=not a.refine_ramp,
        ramp_variants=a.refine_ramp,records=records,phase=phase,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))


if __name__=='__main__':main()
