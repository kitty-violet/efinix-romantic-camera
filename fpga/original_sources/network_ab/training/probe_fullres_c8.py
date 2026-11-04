"""Bounded architecture diagnostic, isolated from the production B1 contract.

Uses existing operation *types*, not a claim that the existing B1 program can
execute this graph unchanged. No RTL, coefficient image or shared schema edits.
"""
import copy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from .b1_fit_controls import inputs, score, update
from .engine import set_reproducible
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .train_b1_distillation import as_tensor, to_rgb


class FullresC8(nn.Module):
    """Stem + four DW/PW pairs + RGB head, all at input resolution."""
    def __init__(self):
        super().__init__()
        def layer(cin, cout, kernel, groups=1):
            return nn.Sequential(
                nn.ReplicationPad2d(1) if kernel == 3 else nn.Identity(),
                nn.Conv2d(cin, cout, kernel, groups=groups, bias=False),
                nn.BatchNorm2d(cout, eps=1e-5), nn.ReLU())
        self.body = nn.Sequential(layer(3, 8, 3), *[
            module for _ in range(4) for module in (layer(8, 8, 3, 8), layer(8, 8, 1))])
        self.head = nn.Conv2d(8, 3, 1, bias=True)

    def forward(self, x):
        return self.head(self.body(x)).clamp(-1, 1)

    def initialize_identity(self):
        # Positive/negative RGB pairs preserve signed RGB through all ReLUs.
        with torch.no_grad():
            for seq in self.body:
                conv, bn = seq[1], seq[2]
                conv.weight.zero_()
                bn.weight.fill_(math.sqrt(1 + bn.eps)); bn.bias.zero_()
                bn.running_mean.zero_(); bn.running_var.fill_(1)
            for c in range(3):
                self.body[0][1].weight[2*c, c, 1, 1] = 1
                self.body[0][1].weight[2*c+1, c, 1, 1] = -1
            for seq in list(self.body)[1:]:
                conv = seq[1]
                for c in range(8):
                    if conv.groups == 8: conv.weight[c, 0, 1, 1] = 1
                    else: conv.weight[c, c, 0, 0] = 1
            self.head.weight.zero_(); self.head.bias.zero_()
            for c in range(3):
                self.head.weight[c, 2*c, 0, 0] = 1
                self.head.weight[c, 2*c+1, 0, 0] = -1
        return self.eval()


def profile(model, image):
    records = []
    def capture(name):
        def hook(module, args, output):
            records.append(dict(name=name, shape=list(output.shape), kernel=module.kernel_size,
                                groups=module.groups, stride=module.stride,
                                weights=module.weight.numel(),
                                macs=output.numel()*module.weight[0].numel(),
                                output_int8_bytes=output.numel()))
        return hook
    handles = [m.register_forward_hook(capture(n)) for n,m in model.named_modules() if isinstance(m,nn.Conv2d)]
    with torch.inference_mode(): model(image)
    for handle in handles: handle.remove()
    return dict(nodes=records, raw_weights=sum(r['weights'] for r in records),
                macs=sum(r['macs'] for r in records),
                materialized_conv_output_int8_bytes=sum(r['output_int8_bytes'] for r in records),
                parameters=sum(p.numel() for p in model.parameters()),
                receptive_field=11, feature_channels=8,
                traffic_note='Tensor volume only. Not measured DDR traffic or hardware FPS.')


def main():
    root = configure_workspace(); _, plan = contract_guard()
    run = root/'runs/sampling-hardware-audit-20261003/fullres-c8-probe'
    run.mkdir(exist_ok=False)
    old = root/'runs/b1-fit-controls-20261003'
    cfg = json.loads((old/'config.json').read_text(encoding='utf-8'))
    cfg.update(model='fullres-c8-ds4-research', architecture_changed=True,
               production_candidate=False, scope='single-patch capacity diagnostic, NOT style/generalization acceptance')
    cfg.pop('code')
    for key in ('source','target'):
        assert file_hash(cfg['selected'][key]) == cfg['selected'][key+'_sha256']
        shutil.copy2(old/(key+'_720.png'), run/(key+'_720.png'))
    source_dir = run/'source'; source_dir.mkdir()
    cfg['code'] = {}
    for p in (Path(__file__), Path(__file__).with_name('b1_fit_controls.py'),
              Path(__file__).with_name('engine.py'), Path(__file__).with_name('train_b1_distillation.py'),
              Path(__file__).with_name('runtime.py')):
        shutil.copy2(p,source_dir/p.name); cfg['code'][str(p)]=file_hash(p)
    write_json(run/'config.json',cfg)
    set_reproducible(cfg['seed'])
    full, teacher, x, target = inputs(run,cfg)
    model = FullresC8().cuda().initialize_identity()
    costs = profile(model,full)
    costs['baseline'] = dict(raw_weights=plan['raw_weight_count'], macs=plan['macs_per_frame'],
                             parameters=plan['trainable_parameter_count'],
                             all_node_output_int8_bytes=sum(math.prod(r['output_shape_hwc']) for r in plan['nodes']),
                             conv_output_int8_bytes=sum(math.prod(r['output_shape_hwc']) for r in plan['nodes'] if r['kind'] in ('conv3x3','dw3x3','pw1x1')))
    assert costs['raw_weights']==784 and costs['macs']==722534400 and costs['parameters']==931
    write_json(run/'costs.json',costs)
    with torch.inference_mode():
        identity_error=float((model(full)-full).abs().max())
        # Same period-four diagnostic as the B1 controls, with replicated RGB.
        xx=torch.arange(128,device='cuda')[None,:]; yy=torch.arange(128,device='cuda')[:,None]
        p=((xx+yy)%4==0).float()*190/127.5 + (32/127.5-1)
        pattern=p[None,None].expand(1,3,128,128)
        periodic_error=float((model(pattern)-pattern).abs().max())
    assert max(identity_error,periodic_error)<2e-6
    # Full-resolution backward and optimizer-state restore, on a disposable net.
    pre=copy.deepcopy(model);opt=torch.optim.Adam(pre.parameters(),lr=cfg['lr'])
    full_step=update(pre,opt,full,teacher,0,cfg)
    for step in range(1,5): update(pre,opt,x,target,step,cfg)
    state=copy.deepcopy(dict(model=pre.state_dict(),opt=opt.state_dict()))
    update(pre,opt,x,target,5,cfg);expected=copy.deepcopy(pre.state_dict())
    pre.load_state_dict(state['model']);opt.load_state_dict(state['opt'])
    update(pre,opt,x,target,5,cfg)
    restore_error=max(float((v-expected[k]).abs().max()) for k,v in pre.state_dict().items())
    assert restore_error==0
    write_json(run/'preflight.json',dict(status='PASS',identity_max_error=identity_error,
               period4_max_error=periodic_error,full720_backward=full_step,restore_max_error=restore_error,
               fp32=True,amp=False,tf32=False))
    del pre,opt,state,expected;torch.cuda.empty_cache()
    # Same small symmetry-breaking perturbation and objective as previous controls.
    g=torch.Generator(device='cuda').manual_seed(cfg['seed']+100)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.add_(torch.randn(m.weight.shape,generator=g,device='cuda')*(.02/math.sqrt(m.weight[0].numel())))
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);begin=time.perf_counter()
    with (run/'metrics.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for step in range(cfg['steps']):
            row=update(model,opt,x,target,step,cfg);log.write(json.dumps(row)+'\n')
            if (step+1)%100==0:
                print('FULLRES_C8',step+1,round(row['loss'],6),round(time.perf_counter()-begin,2),flush=True)
                write_json(run/'status.json',dict(status='TRAINING',step=step+1))
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    path=run/'weights_fp32.pt'
    torch.save(dict(schema='fullres-c8-research-v1',model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    production_candidate=False,steps=cfg['steps'],config_sha256=file_hash(run/'config.json')),path)
    reload=FullresC8().cuda().eval();reload.load_state_dict(torch.load(path,weights_only=True)['model'])
    m=cfg['margin'];px,py,pw,ph=cfg['xywh'];records=[]
    with torch.inference_mode():
        output=model(full);assert float((reload(full)-output).abs().max())==0
        patch_error=float((output[:,:,py+m:py+ph-m,px+m:px+pw-m]-model(x)[:,:,m:-m,m:-m]).abs().max())
        assert patch_error<5e-5
        fit=to_rgb(output[0])[py+m:py+ph-m,px+m:px+pw-m]
        tgt=to_rgb(teacher[0])[py+m:py+ph-m,px+m:px+pw-m]
        Image.fromarray(fit).save(run/'fit_interior.png')
        for r in cfg['regression']:
            a=np.array(Image.open(r['source']));b=np.array(Image.open(r['target']))
            out=to_rgb(model(as_tensor(a)[None].cuda())[0]);Image.fromarray(out).save(run/(r['id']+'.png'))
            records.append(dict(id=r['id'],against_teacher=score(out,b),against_source=score(out,a),
                                role='reviewed_regression_not_blind'))
    canvas=Image.new('RGB',(1280,348),(30,30,30));draw=ImageDraw.Draw(canvas)
    pics=[(old/'source_fit_interior.png','Input'),(old/'target_fit_interior.png','Teacher'),
          (old/'perturbed_style/final/fit_interior.png','B1 / 1200 steps'),(run/'fit_interior.png','Fullres C8 / 1200 steps')]
    for i,(file,label) in enumerate(pics):
        im=Image.open(file);assert im.size==(320,320);canvas.paste(im,(i*320,28));draw.text((i*320+8,8),label,fill='white')
    canvas.save(run/'comparison_1to1.png')
    result=dict(status='COMPLETE_DIAGNOSTIC_NOT_ACCEPTED',steps=cfg['steps'],seconds=seconds,
                fit_teacher=score(fit,tgt),regression=records,full_crop_error=patch_error,reload_error=0,
                weights=str(path),weight_sha256=file_hash(path),limitations=[
                    'Single training patch; prior stairs validation is now fitting data for this diagnostic.',
                    'Whole graph changed (width, depth, resolution); this is NOT a causal single-variable ablation.',
                    'No production graph compiler, full DMA schedule, INT8 or FPGA throughput proof.',
                    'Identity preservation is not style-transfer success; videos and generalization not accepted.'])
    write_json(run/'result.json',result);write_json(run/'status.json',dict(status='COMPLETE',step=cfg['steps']))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
