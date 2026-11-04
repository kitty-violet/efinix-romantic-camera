"""Independent nonlinear initialization control, with identical graph/loss/data."""
import argparse,json,math
from pathlib import Path
import torch
from torch import nn
from .train_teacher_clean_skip import prepare
from .clean_skip_capacity_net import CleanSkipNet
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='teacher-clean-skip-fresh-v1');ap.add_argument('--steps',type=int,default=4000);a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100463);out=ROOT/'runs'/RUN/a.name;prepare(out,a.steps)
    net=CleanSkipNet()
    with torch.no_grad():
        for n in net.nodes:
            if not n['kernel']:continue
            c=net.conv(n['id']);nn.init.kaiming_normal_(c.weight,nonlinearity='relu');c.bias.zero_()
            if n['id'] in ('res1_conv2.pw','context.up.pw'):c.weight.mul_(.1)
        nn.init.normal_(net.conv('rgb').weight,std=.02)
    initial=out/'fresh_initial.pt';torch.save(dict(model=net.state_dict(),seed=2026100463,independent_initialization=True),initial)
    cfg=json.loads((out/'config.json').read_text());cfg.update(initial=str(initial),initial_sha256=file_hash(initial),fp32_lr=.001,
        initialization='Independent Kaiming nonlinear filters; residual endpoint scale0.1; RGB head std0.02. No copied channels or old weights.',source_model_note='Independent initialization control')
    write_json(out/'config.json',cfg);binding=json.loads((out/'binding.json').read_text());binding['code'][str(Path(__file__))]=file_hash(__file__);binding['config_sha256']=file_hash(out/'config.json');write_json(out/'binding.json',binding)
    checks=json.loads((out/'objective_checks.json').read_text());checks.pop('initial_function_error',None);checks.update(initial_function_preserved=False,independent_initialization=True);write_json(out/'objective_checks.json',checks)
    print('FRESH_PREPARED',sum(p.numel() for p in net.parameters()),flush=True)


if __name__=='__main__':main()
