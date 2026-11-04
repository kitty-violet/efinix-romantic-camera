"""Compare known-target geometry encodings, not source inference improvement."""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .b1_joint_interface_targets import joint_interface_fields,regression
from .probe_b1_ordered_distance_geometry import distance_fields
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .train_b1_ternary_regions import region_codes
from .audit_b1_colour_fringe_origin import two_colour_support,fringe
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);regression();state,net=load(a.checkpoint);data=m.load_data(a.data);out=Path(a.out);out.mkdir(exist_ok=False);raw=net.export()
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();setup=state['binary_coverage720']
    pal=(lev@torch.tensor(setup['matrix']).T+torch.tensor(setup['bias'])+128).clamp(0,255).numpy();pk=tuple(map(tuple,pal.tolist()))
    shots={};records=[]
    for row in data['validation']:
        code=region_codes(row['target'],row['region'],pk).astype(np.int64);c=torch.from_numpy(code)[None]
        old=distance_fields(c,8);joint,stats=joint_interface_fields(c,with_stats=True)
        ca,cb,ok=two_colour_support(code,pal);target=m.t.r.read(row['target']).astype(np.float32)
        for name,field in (('independent_sdf',old),('shared_interface',joint)):
            p=fragment(net,F.pad(net.round(field),(0,0,0,0,0,2)),'up1_low_dw')['rgb_head'][0].permute(1,2,0).numpy()+128
            rec=dict(id=row['id'],mode=name,mae=float(np.abs(p-target).mean()),fringe=fringe(p,ca,cb,ok),**stats[0]);records.append(rec)
            for rid,box,title in m.t.r.c.CROPS:
                if rid==row['id']:shots[name,rid,title]=Image.fromarray(p.astype(np.uint8)).crop(box)
            print('JOINT_INTERFACE_DIAGNOSTIC',rec,flush=True)
    board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);by={r['id']:r for r in data['validation']}
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by[rid]['target']).crop(box),shots['independent_sdf',rid,title],shots['shared_interface',rid,title]]
        for j,(im,label) in enumerate(zip(pics,('目标','已知独立距离·诊断','已知共用界面·诊断'))):
            board.paste(im,(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png');assert net.export()==raw
    write_json(out/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),
        target_injection_only=True,weights_unchanged=True,source_predictions_not_improved=True,visual_acceptance=False))


if __name__=='__main__':main()
