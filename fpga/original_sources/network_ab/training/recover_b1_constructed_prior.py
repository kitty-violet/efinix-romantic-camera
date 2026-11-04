"""Alternative exact-B1 initialization and learning, without an inference filter."""
import json
from pathlib import Path
import shutil
import sys
import torch
from torch import nn
from torch.nn import functional as F
from . import train_b1_cartoon_budget as base
from .seed_b1_cartoon import seed
from .runtime import configure_workspace,write_json,file_hash
from .repair_style_retention import gray

RUN='b1-constructed-recovery-20261004'


def construct():
    # Luminance quantization preserves chroma better than separate RGB staircases.
    net=seed(.28,1.5)
    with torch.no_grad():
        seq=net.operators[net.names['res1_conv1.pw']]
        conv=next(m for m in seq if isinstance(m,nn.Conv2d));bn=next(m for m in seq if isinstance(m,nn.BatchNorm2d))
        conv.weight.zero_();bn.bias.zero_();conv.weight[0,3,0,0]=1
        for j,threshold in enumerate((.25,.75,1.25,1.75)):
            for side,sign in enumerate((-1,1)):
                k=1+2*j+side;conv.weight[k,3,0,0]=1;bn.bias[k]=-(threshold+sign*.14)
        seq=net.operators[net.names['res1_conv2.pw']];conv=next(m for m in seq if isinstance(m,nn.Conv2d));conv.weight.zero_()
        conv.weight[:3,0,0,0]=-1
        for j in range(4):conv.weight[:3,1+2*j,0,0]=.5/.28;conv.weight[:3,2+2*j,0,0]=-.5/.28
        # Populate residual3 feature extractors; its zero final PW keeps the
        # initial mapping exact while allowing the residual to learn immediately.
        g=torch.Generator().manual_seed(2026100409)
        for name in ('res3_conv1.dw','res3_conv1.pw','res3_conv2.dw'):
            seq=net.operators[net.names[name]];conv=next(m for m in seq if isinstance(m,nn.Conv2d));bn=next(m for m in seq if isinstance(m,nn.BatchNorm2d))
            conv.weight.copy_(torch.randn(conv.weight.shape,generator=g)*(.2/conv.weight[0].numel()**.5));bn.bias.fill_(.05)
    return net


class Training(base.Training):
    def __init__(self,root,cfg,manifest):
        super().__init__(root,cfg,manifest)
        self.prior=construct().cuda().eval().requires_grad_(False)

    def spatial(self,p,t,x,native):
        terms,projection,cropped=super().spatial(p,t,x,native)
        with torch.no_grad():target=self.prior(x)
        m=self.cfg['margin'];pp=gray(p[:,:,m:-m,m:-m]);tt=gray(target[:,:,m:-m,m:-m])
        # Training-only realizable style anchor prevents collapse back to weak
        # identity/color transfer. It is deliberately NOT the original teacher.
        terms['prior']=F.l1_loss(F.avg_pool2d(pp,2),F.avg_pool2d(tt,2))
        return terms,projection,cropped


def prepare():
    root=configure_workspace();old=root/'runs'/base.RUN;run=root/'runs'/RUN;run.mkdir(exist_ok=False)
    cfg=json.loads((old/'config.json').read_text());shutil.copy2(old/'data_manifest.json',run/'data_manifest.json')
    cfg.update(max_steps=400,lr=.0002,lr_floor=.00003,critic_warmup=50,stability_ramp=100,native_interval=25)
    cfg['weights'].update(rgb=.25,planes=.7,prior=2.,adversarial=.04,phase=2.,temporal=1.)
    path=run/'constructed_initial_fp32.pt';torch.save(dict(model=construct().state_dict(),trained=False,hand_constructed=True),path)
    cfg.update(initial=str(path),initial_sha256=file_hash(path),
        style_target='Teacher structure/color plus a realizable luminance-surface/edge B1 prior; not exact teacher reproduction',
        inference='Only final B1 weights. The prior model exists only during training and is discarded.')
    write_json(run/'config.json',cfg)
    paths=[Path(p) for p in json.loads((old/'binding.json').read_text())['code']]+[Path(__file__),Path(__file__).with_name('seed_b1_cartoon.py')]
    folder=run/'source';folder.mkdir()
    for p in paths:shutil.copy2(p,folder/p.name)
    write_json(run/'binding.json',dict(code={str(p):file_hash(p) for p in paths},config_sha256=file_hash(run/'config.json'),manifest_sha256=file_hash(run/'data_manifest.json')))
    shutil.copy2(old/'hardware_boundary.json',run/'hardware_boundary.json');shutil.copy2(old/'data_checks.json',run/'data_checks.json')
    print('CONSTRUCTED_RECOVERY_PREPARED',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        base.RUN=RUN;base.Training=Training;base.main()
