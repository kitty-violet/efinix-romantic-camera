"""Structural ABI exercise only; untrained, arbitrary fixed scales."""
import torch
from .coarse_skip_net import CoarseSkipNet
from .region_detail_target import ROOT,RUN
from .runtime import write_json,file_hash


def main():
    torch.set_num_threads(2);torch.manual_seed(2026100441);net=CoarseSkipNet().eval()
    with torch.no_grad():
        y=net(torch.zeros(1,3,720,1280));assert y.shape==(1,32,720,1280)
    net.scales={n['id']:1/128 for n in net.nodes};net.scales['input']=1/128
    out=ROOT/'runs'/RUN/'coarse-skip-abi-diagnostic';out.mkdir(exist_ok=False);p=net.integer_package()
    p.update(untrained_structural_diagnostic=True,calibration=False,style_accepted=False,source_sha256=file_hash(__file__))
    write_json(out/'runtime_parameters.json',p)


if __name__=='__main__':main()
