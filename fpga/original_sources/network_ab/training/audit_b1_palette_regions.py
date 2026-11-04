"""Measure small connected palette regions from reloaded integer B1 nodes.

Half-coverage AA states are assigned to their nearest ordinal class. These
nominal palette components are a diagnostic, not exact constant-RGB areas or
a proof of a universal minimum area bound.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from . import train_b1_minimum_area as m
from .minimum_area_blocks import components
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


def palette_indices(coverage,states):
    """Round each pair's coverage to0/1/2 using its actual decoder scale."""
    assert states in (4,8,16) and coverage.shape[0]==8
    assert coverage.min()>=0 and coverage.max()<=states
    count=(coverage*2+states//2)//states
    assert count.min()>=0 and count.max()<=2
    return count[:4].sum(0)*25+count[4:6].sum(0)*5+count[6:].sum(0)


def check_palette_indices():
    # Full and half-covered classes must agree across all supported scales.
    expected=None
    for states in (4,8,16):
        z=np.tile(np.array([0,states//2,states],dtype=np.int32),(8,2,1))
        result=palette_indices(z,states)
        if expected is None:expected=result
        else:assert np.array_equal(result,expected)
    assert expected.tolist()==[[0,112,224],[0,112,224]]


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();configure_workspace();set_reproducible(20261006131);cv2.setNumThreads(2)
    net=m.t.r.initialize();state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:net.package['residual_encoding']=state['residual_encoding']
    check_palette_indices()
    states=round(float(net.weight['up2_high_dw'][0,0].sum()*127))
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()));records=[]
    for row in m.load_data()['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];nodes,_=actual(x*128,True)
        z=nodes['up2_high_dw'][0].cpu().numpy().astype(np.int32)
        code=palette_indices(z,states)
        labels=components(code);area=np.bincount(labels.ravel())
        record=dict(id=row['id'],regions=int(len(area)),min_area=int(area.min()),
            median_area=float(np.median(area)),pixel_fraction_below256=float(area[area<256].sum()/area.sum()),
            region_fraction_below256=float((area<256).mean()),
            pixel_fraction_below64=float(area[area<64].sum()/area.sum()))
        records.append(record);print('PALETTE_REGIONS',record,flush=True)
    write_json(a.output,dict(checkpoint=a.checkpoint,checkpoint_sha256=file_hash(a.checkpoint),coverage_states=states,records=records,
        interpretation='Connected nominal palette indices; AA assigned to nearest class; not exact RGB components',
        minimum_area_guaranteed=False,visual_acceptance=False))


if __name__=='__main__':main()
