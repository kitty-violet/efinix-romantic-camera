"""Check exact initial re-encoding on native frames and actual B1 bytes."""
import argparse,copy
import torch
from .probe_b1_encoder_gauge import load
from .b1_fine_spatial_decoder import install
from .train_b1_trainable_luma_readout import backend_palette
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);_,old=load(a.checkpoint);net=copy.deepcopy(old);palette=backend_palette(old);install(net)
    assert torch.equal(palette,backend_palette(net))
    raw=net.export();assert len(raw)==20480
    rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');checks=[]
    for row in m.load_data(a.data)['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        aa=old(x,all_nodes=True);bb=net(x,all_nodes=True);cc,_=rt(net.round(x*128),True)
        assert torch.equal(aa['rgb_head'],bb['rgb_head'])
        assert all(torch.equal(aa[k],v) for k,v in bb.items() if k not in ('up2_low_pw','up2_nn','up2_high_dw'))
        assert all(torch.equal(bb[k],v) for k,v in cc.items())
        checks.append(dict(id=row['id'],rgb_error=0,unchanged_node_error=0,actual_all_node_error=0))
        print('FINE_DECODER_NATIVE',row['id'],'RGB0/unchanged0/actual0',flush=True)
    write_json(a.out,dict(checks=checks,bytes=len(raw),initial_constant1215_error=0,initial_only=True,
        checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__),visual_acceptance=False))


if __name__=='__main__':main()
