"""Spatial majority of sector-choice bits in existing DW layers."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    s=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(s['model']);net.quantized=True
    net.package['residual_encoding']=s['residual_encoding']
    for key,channels in (('res1_conv2__dw',(12,13)),('res2_conv2__dw',(6,))):
        for c in channels:
            net.weight[key][c,0]=1;net.bias[key][c]=-4*127;getattr(net,key+'_m')[:,c]=2048
    gen=torch.Generator(device='cuda').manual_seed(61006);z=torch.zeros(16,32,32,32,device='cuda')
    z[:,:16]=torch.randint(0,128,(16,16,32,32),generator=gen,device='cuda')
    p=fragment(net,z,'res1_conv1.dw');chosen=p['res3_conv1.dw'][:,:3]+p['res3_conv1.dw'][:,3:6]
    possible=torch.stack([z[:,s*4:s*4+3] for s in range(4)],2)
    assert ((possible==chosen[:,:,None]).all(1).any(1)).all()
    rows=m.load_data()['validation'];export(net,rows,dest/'package');m.t.preview(net,rows,dest/'preview')
    s.update(model=net.state_dict(),parent=a.warm,spatial_sector_consensus=True);torch.save(s,dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name);write_json(dest/'review.json',dict(
        parent_sha256=file_hash(a.warm),random_spatial_pixels=16384,always_one_original_candidate=True,
        pointwise_minimum_cost_guaranteed=False,added_operators=0,postfilter=False,minimum_area_guaranteed=False,visual_acceptance=False))
    print('SECTOR_CONSENSUS_COMPLETE',flush=True)


if __name__=='__main__':main()
