"""Anti-alias learned region edges using only existing PW, DW and Q17 rows.

Sharpen the learned decision feature, then retain four coverage sublevels in
the final normalized 3x3 DW. Region interiors remain at endpoint colours.
This is a coefficient calibration of the trained network, not extra inference.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from . import train_b1_stable_regions as stable
from . import train_b1_luma_regions as region
from . import train_b1_learned_colour as colour
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=stable.OUT/'edge-coverage'


def main():
    configure_workspace();set_reproducible(2026100527);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=stable.Trainer(cache,selected,1800)
    parent=stable.OUT/'checkpoint_001800.pt';tr.restore(parent)
    with torch.no_grad():
        tr.net.weight['up2_low_pw'][:6].mul_(4)
        tr.net.bias['up2_low_pw'][:6].mul_(4).sub_(3*63.5)
        tr.net.weight['up2_high_dw'][:6].mul_(4)
        tr.net.weight['rgb_head'][:,:6].div_(4)
        for key in ('up2_low_pw','up2_high_dw'):
            w=tr.net.weight[key]
            m=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/112).ceil().clamp(1,131071)
            getattr(tr.net,key+'_m').copy_(m.double()[None,:,None,None])
        tr.net.rgb_head_m.fill_(65536)
    tr.save(OUT/'checkpoint_calibrated.pt')
    colour.review(tr,data,selected,OUT/'package',integer=True,oracle_comparison=False)
    write_json(OUT/'calibration.json',dict(parent=str(parent),sha256=file_hash(parent),learned_steps=5300,
        class_decision_center=63.5,feature_gain=4,coverage_steps=4,gaussian_cross_radius_pixels=1,
        only_changed=('up2_low_pw','up2_high_dw','rgb_head'),no_additional_ops=True,postfilter=False,
        anti_alias_calibrated_not_extra_training=True,visual_accepted=False))
    shutil.copy2(__file__,OUT/Path(__file__).name);print('EDGE_COVERAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
