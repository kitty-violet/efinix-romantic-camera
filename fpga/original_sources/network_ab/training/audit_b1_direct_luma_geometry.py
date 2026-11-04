"""Separate ideal contour width from source-prediction error in14Y decoding."""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .probe_b1_direct_luma_planes import install
from .probe_b1_monotone_pair_decoder import make_board
from .b1_joint_interface_targets import joint_interface_fields
from .audit_b1_binary_boundary_phase import contour_metrics
from .train_b1_ternary_regions import region_codes
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);net=copy.deepcopy(base);install(net);out=Path(a.out);out.mkdir(exist_ok=False)
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    pal=(levels@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])).clamp(-128,127);pk=tuple(map(tuple,(pal+128).tolist()))
    records=[];yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij')
    for name,model in (('radixY',base),('direct14Y',net)):
        for low,high in ((3,5),(5,9),(2,11)):
            ca=81*low+40;cb=81*high+40
            for slope in (.2,.5,1.):
                for phase in range(4):
                    xpos=64+slope*(torch.arange(96)-48)+phase-.5
                    code=torch.where(xx>xpos[:,None],cb,ca)[None];fields=joint_interface_fields(code,8)
                    p=fragment(model,F.pad(model.round(fields),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head'][0]
                    metrics=contour_metrics(p,pal[ca],pal[cb],xpos)
                    metrics['overshoot']=float(torch.maximum(F.relu(p-pal[cb,:,None,None]),F.relu(pal[ca,:,None,None]-p)).max())
                    records.append(dict(mode=name,levels=[low,high],slope=slope,phase=phase,**metrics))
    boards=[{},{},{}];images=[]
    for row in m.load_data(a.data)['validation']:
        target=m.t.r.read(row['target']);boards[0][row['id']]=Image.fromarray(target)
        codes=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None]
        truth=joint_interface_fields(codes,8);z=F.pad(net.round(truth),(0,0,0,0,0,2))
        for j,(name,model) in enumerate((('radixY',base),('direct14Y',net)),1):
            p=fragment(model,z,'up1_low_dw')['rgb_head'];im=m.t.r.c.rgb_image(p[0]/128);boards[j][row['id']]=im
            im.save(out/f'{name}_{row["id"]}_known.png');images.append(dict(id=row['id'],mode=name,known_target_mae=float(np.abs(np.array(im).astype(float)-target).mean())))
    means={}
    for name in ('radixY','direct14Y'):
        rr=[r for r in records if r['mode']==name];means[name]={key:sum(float(r[key]) for r in rr)/len(rr) for key in rr[0] if key not in ('mode','levels','slope','phase')}
    make_board(boards,['目标','已知共同界面·6位Y','已知共同界面·14独立Y'],out/'known_comparison.png')
    write_json(out/'review.json',dict(geometry=records,means=means,images=images,source_prediction=False,visual_acceptance=False,source_sha256=file_hash(__file__)))
    print('DIRECT_LUMA_GEOMETRY',means,flush=True)


if __name__=='__main__':main()
