"""Keep ordinal coverage until the existing720p DW requantizer.

Pack two adjacent monotonic masks in each of eight existing channels. Final
0/1/2 band decisions happen at720p, not by adding any operator or resolution.
"""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    s=torch.load(a.warm,map_location='cpu',weights_only=True);assert s.get('manual_ordinal_majority')
    net=m.t.r.initialize();net.load_state_dict(s['model']);net.quantized=True;net.package['residual_encoding']=s['residual_encoding']
    old=net.weight['rgb_head'][:,:3,0,0].clone()
    net.weight['up2_low_dw'].zero_();net.bias['up2_low_dw'].zero_();net.weight['up2_low_dw'][:15,0,1,1]=1;net.up2_low_dw_m.fill_(2048)
    net.weight['up2_low_pw'].zero_();net.bias['up2_low_pw'].zero_();net.up2_low_pw_m.fill_(1024)
    pairs=((0,1),(2,3),(4,5),(6,6),(7,8),(9,10),(11,12),(13,14))
    for j,pair in enumerate(pairs):
        for src in pair:net.weight['up2_low_pw'][j,src,0,0]+=.5
    # Exact integer kernel [1,2,1]^2 with Q17 multiplier129; output codes0..2.
    kernel=old.new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    net.weight['up2_high_dw'][:,0]=kernel*(129/131072);net.bias['up2_high_dw'].zero_();net.up2_high_dw_m.fill_(129)
    net.weight['rgb_head'].zero_()
    for j,(channel,gain) in enumerate(((0,14),(0,14),(0,15),(0,9),(1,12),(1,12),(2,10),(2,10))):
        net.weight['rgb_head'][:,j,0,0]=old[:,channel]*gain
    scale=net.weight['rgb_head'].abs().flatten(1).amax(1)/120
    net.rgb_head_m.copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
    rows=m.load_data()['validation'];records=[]
    for row in rows:
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True);code=nodes['up2_high_dw']
        assert torch.all((code>=0)&(code<=2)&(code==code.round()))
        records.append(dict(id=row['id'],final_integer_levels=[int(code.min()),int(code.max())]))
    export(net,rows,dest/'package');m.t.preview(net,rows,dest/'preview');s.update(model=net.state_dict(),parent=a.warm,late720_palette=True)
    torch.save(s,dest/'initial.pt');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,parent_sha256=file_hash(a.warm),
        change='pairwise ordinal coverages, 720p existing DW quantization',added_operators=0,postfilter=False,
        global_minimum_area_guaranteed=False,visual_acceptance=False))
    print('LATE720_PALETTE_COMPLETE',flush=True)


if __name__=='__main__':main()
