"""Dependency-consistent physical channel removal from the accepted teacher.

All equal-width residual paths share channel indices. This is structured
pruning, not a mask on full tensors. No claim of B1 operator compatibility.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from .trainable_animegan import TrainableAnimeGAN
from .train_b1_distillation import as_tensor, to_rgb, read_image
from .smooth_surface_guard import correct_surface
from .runtime import configure_workspace, file_hash, write_json

RUN = 'teacher-pruning-20261003'
TEACHER_SHA = 'c86b9b99e901c0ad6be7ce74bd35388a5028dc8c12228c579489025fa6b2f7a5'


class AttentionFlatten(nn.Module):
    def forward(self, x, shape):
        return x.reshape(x.shape[0], -1, int(shape[-1]))


class AttentionUnflatten(nn.Module):
    def forward(self, x, shape):
        target = shape.detach().cpu().tolist()
        target[0] = x.shape[0]
        return x.reshape(target)


def make_indices(teacher, keep):
    scores = {n: torch.zeros(n) for n in (32, 64, 128)}
    for module in teacher.modules():
        if not isinstance(module, nn.Conv2d): continue
        w = module.weight.detach().cpu().abs()
        for width, value in [(module.out_channels, w.mean((1,2,3))), (module.in_channels, w.mean((0,2,3)))]:
            if width in scores:
                scores[width] += value/value.mean().clamp_min(1e-12)
    indices = {str(c): torch.topk(s, round(c*keep)).indices.sort().values.tolist() for c,s in scores.items()}
    return indices, {str(c): v.tolist() for c,v in scores.items()}


def prune_inplace(teacher, indices):
    indices = {int(c): torch.tensor(v, dtype=torch.long) for c,v in indices.items()}
    records = []
    for name, module in list(teacher.named_modules()):
        if not isinstance(module, nn.Conv2d): continue
        assert module.groups == 1
        ii = indices.get(module.in_channels, torch.arange(module.in_channels))
        oo = indices.get(module.out_channels, torch.arange(module.out_channels))
        new = nn.Conv2d(len(ii), len(oo), module.kernel_size, module.stride, module.padding,
                        module.dilation, bias=module.bias is not None)
        with torch.no_grad():
            new.weight.copy_(module.weight.detach().cpu()[oo][:,ii])
            if module.bias is not None: new.bias.copy_(module.bias.detach().cpu()[oo])
        parent, leaf = name.rsplit('.', 1)
        teacher.get_submodule(parent)._modules[leaf] = new
        records.append(dict(name=name, before=[module.in_channels,module.out_channels], after=[len(ii),len(oo)]))
    # These are explicit channel dimensions in norm/attention shape constants,
    # not spatial dimensions or learned numeric values.
    for name, value in list(teacher.generator.initializers.named_buffers()):
        if value.dtype not in (torch.int32, torch.int64): continue
        updated = value.clone()
        for c, idx in indices.items(): updated[value == c] = len(idx)
        # The pinned author export hard-codes batch=1 in normalization reshapes.
        if updated.numel()==4 and value.tolist()[:3]==[1,1,1] and int(value[-1]) in indices:
            updated[0] = -1
        teacher.generator.initializers._buffers[name] = updated
    teacher.generator._modules['generator_1/main/External_attention/Reshape'] = AttentionFlatten()
    teacher.generator._modules['generator_1/main/External_attention/Reshape_1'] = AttentionUnflatten()
    return teacher.eval(), records


def build_pruned(root, indices, state=None):
    teacher = TrainableAnimeGAN(root)
    if state is not None: teacher.load_state_dict(state)
    return prune_inplace(teacher, indices)[0]


def profile(net, x):
    rows=[]
    def capture(name):
        def hook(module,args,output):
            rows.append(dict(name=name,shape=list(output.shape),kernel=list(module.kernel_size),
                             weights=module.weight.numel(),macs=output.numel()*module.weight[0].numel()))
        return hook
    hooks=[m.register_forward_hook(capture(n)) for n,m in net.named_modules() if isinstance(m,nn.Conv2d)]
    with torch.inference_mode(): net(x)
    for h in hooks:h.remove()
    return dict(parameters=sum(p.numel() for p in net.parameters()), conv_macs=sum(r['macs'] for r in rows), convs=rows,
        scope='Convolution mathematical work only; excludes dynamic LADE normalization, external attention softmax, resize, tanh and surface guard; not FPGA resource/timing proof.')


def main():
    root=configure_workspace(); run=root/'runs'/RUN; run.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    path=root/'runs/texture-recovery-20261003/final/refined_style_fp32.pt'
    assert file_hash(path)==TEACHER_SHA
    state=torch.load(path,map_location='cpu',weights_only=True)['generator']
    original=TrainableAnimeGAN(root).cuda().eval();original.load_state_dict(state)
    manifest=json.loads((root/'runs/fullres-c8-stylepush-color-20261003/data_manifest.json').read_text(encoding='utf-8'))
    write_json(run/'data_manifest.json',manifest)
    image_rows=manifest['validation']
    full_indices,_=make_indices(original,1.)
    control=build_pruned(root,full_indices,state).cuda().eval()
    errors=[]
    with torch.inference_mode():
        for r in image_rows:
            x=as_tensor(read_image(r['source']))[None].cuda()
            errors.append(float((original(x)-control(x)).abs().max()))
        # Real mini-batch equivalence: each image must retain independent statistics.
        a=as_tensor(read_image(image_rows[0]['source'])[0:360,0:640])[None].cuda()
        b=as_tensor(read_image(image_rows[2]['source'])[160:520,384:1024])[None].cuda()
        batched=control(torch.cat([a,b]))
        separate=torch.cat([original(a),original(b)])
        batch_error=float((batched-separate).abs().max())
    assert max(errors)<5e-5 and batch_error<5e-5,(errors,batch_error)
    write_json(run/'equivalence.json',dict(status='PASS',full720_errors=errors,batch2_360_error=batch_error,
        source_teacher_sha256=TEACHER_SHA,explanation='100% retention, no weight alteration; also remove batch=1 reshape limitation.'))
    del control;torch.cuda.empty_cache()
    records=[]
    for keep in (.75,.5,.25):
        indices,ranking=make_indices(original,keep)
        candidate=build_pruned(root,indices,state).cuda().eval()
        name=f'keep{round(keep*100):02d}';out=run/name;out.mkdir()
        p=out/'initial_fp32.pt'
        torch.save(dict(schema='structured-pruned-animation-teacher-fp32-v1',indices=indices,
            generator={k:v.cpu() for k,v in candidate.state_dict().items()},source_teacher_sha256=TEACHER_SHA,
            keep_channel_fraction=keep,recovery_steps=0,production=False),p)
        info=profile(candidate,x)
        with torch.inference_mode():
            for r in image_rows:
                src=read_image(r['source']); inp=as_tensor(src)[None].cuda()
                raw=(candidate(inp)[0].permute(1,2,0).cpu().numpy()*128+128).clip(0,255).astype(np.uint8)
                corrected=correct_surface(src,raw)
                Image.fromarray(raw).save(out/(r['id']+'_raw.png'))
                Image.fromarray(corrected).save(out/(r['id']+'.png'))
                if r['id'] not in ('val_stairs_090','val_plaza_090','val_new_hallway_120'):continue
                xy=(424,320) if 'stairs' in r['id'] else (80,400) if 'plaza' in r['id'] else (560,150)
                xx,yy=xy;panel=Image.new('RGB',(960,348),'#191f29');draw=ImageDraw.Draw(panel)
                for i,(v,label) in enumerate([(src,'Source'),(read_image(r['target']),'Teacher'),(corrected,name+' / no recovery')]):
                    panel.paste(Image.fromarray(v[yy:yy+320,xx:xx+320]),(320*i,28));draw.text((320*i+8,8),label,fill='white')
                panel.save(out/(r['id']+'_comparison.png'))
        records.append(dict(name=name,keep=keep,indices=indices,ranking=ranking,profile=info,weights_sha256=file_hash(p)))
        write_json(run/'pruning.json',dict(status='INITIAL_PRUNING_COMPLETE' if len(records)==3 else 'PRUNING',records=records))
        print('PRUNED',name,info['parameters'],info['conv_macs'],flush=True)
        del candidate;torch.cuda.empty_cache()
    write_json(run/'source_binding.json',dict(code={str(Path(__file__)):file_hash(__file__)},teacher=str(path),teacher_sha256=TEACHER_SHA,
               note='Original teacher and source files unchanged; physical smaller Conv2d tensors, not sparse masks.'))
    print('PRUNING_COMPLETE',flush=True)


if __name__=='__main__':main()
