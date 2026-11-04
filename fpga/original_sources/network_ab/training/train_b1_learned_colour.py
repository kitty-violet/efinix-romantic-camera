"""Learn colour reconstruction inside B1 instead of freezing a local mean.

Oracle contour input is a training/diagnostic intervention, never deployment.
Published actual outputs and coefficient images always use predicted features.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F

from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .engine import set_reproducible
from .qat_b1_deploy_preview import Round
from .b1_trained_image import Parameters, encode, decode, runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .train_b1_structural_gate import CROPS
from .train_b1_deep_contour import OUT as LABELS
from .train_b1_late_contour import OUT as PARENT
from .engineer_b1_blocks import oracle_records
from network_ab.contract_reference import execute_integer

ROOT = LABELS.parent
OUT = ROOT/'b1-learned-colour-20261005-v1'
SEED = 2026100517
SOURCE = PARENT/'step_002200/coefficients-padded.bin'
SELECT = [('070939', 1, (672, 256, 992, 576)),
          ('080118', 1, (640, 336, 960, 656)),
          ('121509', 1, (768, 288, 1088, 608)),
          ('121831', 1, (416, 224, 736, 544))]


class ColourB1(nn.Module):
    """Effective FP32 coefficients, then fixed Q17/INT8 STE in the same graph."""
    def __init__(self, revive=True, source=SOURCE):
        super().__init__()
        self.plan = contract_guard()[1]
        self.package = runtime_from_image(Path(source).read_bytes())
        self.weight, self.bias = nn.ParameterDict(), nn.ParameterDict()
        self.round = Round()
        self.quantized = False
        for r in self.plan['nodes']:
            name, key = r['id'], r['id'].replace('.', '__')
            if name not in self.package['parameters']:
                continue
            p = self.package['parameters'][name]
            w = torch.tensor(p['weights'], dtype=torch.float32)
            if r['kind'] == 'dw3x3':
                w = w.reshape(r['channels'], 1, 3, 3)
            else:
                w = w.reshape(r['channels'], r['kernel'], r['kernel'], r['input_channels']).permute(0, 3, 1, 2).contiguous()
            s = torch.tensor(p['multipliers'], dtype=torch.float32)/131072
            self.weight[key] = nn.Parameter(w*s[:, None, None, None])
            self.bias[key] = nn.Parameter(torch.tensor(p['bias'], dtype=torch.float32)*s)
            self.register_buffer(key+'_m', torch.tensor(p['multipliers'], dtype=torch.float64)[None, :, None, None])
        if revive:
            # Previously zero filters/rows + ReLU cannot learn from zero input.
            # Revive reserved decoder channels without adding graph operations.
            with torch.no_grad():
                w = self.weight['up1_low_dw']
                for c in range(6, 32):
                    w[c].zero_(); w[c, 0, 1, 1] = 1
                w = self.weight['up1_low_pw']
                w[:, 6:24].normal_(0, .002)
                for c in range(9, 16):
                    w[c].normal_(0, .003)
                    w[c, c+1, 0, 0] += .1
                    self.bias['up1_low_pw'][c] = 8
                for c in range(3, 8):
                    self.weight['up2_low_pw'][c].normal_(0, .003)
                    self.weight['up2_low_pw'][c, c % 3, 0, 0] += .1
                    self.bias['up2_low_pw'][c] = 8
                self.weight['rgb_head'][:, 3:].normal_(0, .002)

    def forward(self, x, gate=None, mix=0., all_nodes=False):
        rnd = self.round if self.quantized else lambda a: a
        nodes = {'input': rnd(x*128).clamp(-128, 127)}
        prediction = None
        for r in self.plan['nodes']:
            name, key = r['id'], r['id'].replace('.', '__')
            a = nodes[r['inputs'][0]]
            if key in self.weight:
                w, b = self.weight[key], self.bias[key]
                if r['kernel'] == 3:
                    a = F.pad(a, (1, 1, 1, 1), mode='replicate')
                if self.quantized:
                    m = getattr(self, key+'_m')
                    s = (m/131072).float().flatten()
                    wi = self.round(w/s[:, None, None, None]).clamp(-127, 127)
                    bi = self.round(b/s).double()[None, :, None, None]
                    dot = F.conv2d(a, wi, stride=r['stride'], groups=r['groups'])
                    pre_activation = self.round((dot.double()+bi).clamp(-(2**31), 2**31-1)*m/131072).float()
                    y = pre_activation.clamp(-128, 127)
                else:
                    pre_activation = F.conv2d(a, w, b, stride=r['stride'], groups=r['groups'])
                    y = pre_activation.clamp(-128, 127)
                if r['activation'] == 'relu':
                    y = y.clamp_min(0)
                if torch.is_grad_enabled() and hasattr(self, 'training_activation_proxy'):
                    y = self.training_activation_proxy(name, y, pre_activation, r)
            elif r['kind'] == 'nearest_x2':
                y = a.repeat_interleave(2, -2).repeat_interleave(2, -1)
            else:
                z = nodes[r['inputs'][1]]
                q = self.package['residual_encoding'][name]
                small, other = (a, z) if q['main_scaled'] else (z, a)
                if self.quantized:
                    aligned = rnd(small.double()*q['multiplier']/131072).clamp(-128, 127)
                    y = rnd((aligned+other.double())*q['post_multiplier']/131072).clamp(-128, 127).float()
                else:
                    y = ((small*q['multiplier']/131072).clamp(-128, 127)+other)*q['post_multiplier']/131072
                    y = y.clamp(-128, 127)
            if name == 'res3_add':
                prediction = y[:, 24:25]
                if gate is not None and mix:
                    y = torch.cat((y[:, :24], (1-mix)*prediction+mix*gate, y[:, 25:]), 1)
            if torch.is_grad_enabled() and hasattr(self, 'training_node_proxy'):
                y = self.training_node_proxy(name, y, nodes)
            nodes[name] = y
        return nodes if all_nodes else (nodes['rgb_head']/128, prediction)

    @torch.no_grad()
    def calibrate(self):
        for key, w in self.weight.items():
            m = (w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/112).ceil().clamp(1, 131071)
            getattr(self, key+'_m').copy_(m.double()[None, :, None, None])

    @torch.no_grad()
    def export(self):
        params = {}
        for r in self.plan['nodes']:
            key = r['id'].replace('.', '__')
            if key not in self.weight:
                continue
            m = getattr(self, key+'_m').flatten()
            scale = (m/131072).float()
            w = self.round(self.weight[key]/scale[:, None, None, None]).clamp(-127, 127).long()
            b = self.round(self.bias[key]/scale).long()
            assert b.abs().max() < 2**31
            params[r['id']] = Parameters(tuple(map(tuple, w.permute(0, 2, 3, 1).reshape(len(w), -1).cpu().tolist())),
                                         tuple(b.cpu().tolist()), tuple(m.long().cpu().tolist()), r['activation'] == 'relu')
        raw = encode(params, self.package['residual_encoding'])
        return raw+bytes(20480-len(raw))


def image_tensor(path):
    a = np.array(Image.open(path).convert('RGB'))
    return torch.tensor(a.transpose(2, 0, 1).copy(), device='cuda', dtype=torch.float32)/128-1


def load_data():
    data = json.loads((LABELS/'data.json').read_text(encoding='utf-8'))
    paired = json.loads((ROOT/'b1-deploy-preview-20261005-v1/data.json').read_text(encoding='utf-8'))
    selected = []
    for video, index, box in SELECT:
        pair = [r for r in paired['pairs'] if Path(r['source']).stem.endswith(video)][index]
        row = next(r for r in data if r['source'] == pair['current'])
        assert row['split'] == 'train'
        selected.append(dict(row, box=list(box), video=video))
    cached = []
    for r in data:
        for k in ('source', 'target', 'label'):
            assert file_hash(r[k]) == r[k+'_sha256']
        cached.append(dict(row=r, source=image_tensor(r['source']), target=image_tensor(r['target']),
                           gate=torch.tensor(np.load(r['label']).astype(np.float32), device='cuda')[None]*32-8))
    return data, selected, cached


def rgb_image(t):
    return Image.fromarray((t.detach().cpu().permute(1, 2, 0)*128+128).round().clamp(0, 255).numpy().astype(np.uint8))


def loss_colour(p, t):
    # Targets alone decide which spatial changes to remove or preserve.
    # Strong edges and flat interiors are normalized separately so the many
    # low-gradient pixels cannot win simply by blurring the whole output.
    rgb = F.l1_loss(p, t)
    boundary, interior = [], []
    for scale in (1, 2, 4):
        a, b = (p, t) if scale == 1 else (F.avg_pool2d(p, scale), F.avg_pool2d(t, scale))
        for dim in (-1, -2):
            ga, gb = torch.diff(a, dim=dim), torch.diff(b, dim=dim)
            strength = gb.abs().amax(1, keepdim=True)
            edge = strength > .06
            flat = strength < .012
            boundary.append(((ga-gb).abs()*edge).sum()/(3*edge.sum()).clamp_min(1))
            interior.append((ga.abs()*flat).sum()/(3*flat.sum()).clamp_min(1))
    same = []
    for offset in (4, 12):
        for dim in (-1, -2):
            pa, pb = p.narrow(dim, offset, p.shape[dim]-offset), p.narrow(dim, 0, p.shape[dim]-offset)
            ta, tb = t.narrow(dim, offset, t.shape[dim]-offset), t.narrow(dim, 0, t.shape[dim]-offset)
            mask = (ta-tb).abs().amax(1, keepdim=True) < .025
            same.append(((pa-pb).abs()*mask).sum()/(3*mask.sum()).clamp_min(1))
    values = dict(rgb=rgb, boundary=torch.stack(boundary).mean(),
                  interior=torch.stack(interior).mean(), same_colour=torch.stack(same).mean())
    return rgb+1.2*values['boundary']+.3*values['interior']+.3*values['same_colour'], values


class Trainer:
    def __init__(self, cached, selected, mode, steps):
        self.net = ColourB1().cuda().eval()
        self.cached, self.selected, self.mode, self.steps = cached, selected, mode, steps
        self.rng, self.step = random.Random(SEED), 0
        groups = []
        for kind, parameters in [('w', self.net.weight), ('b', self.net.bias)]:
            for key, p in parameters.items():
                decoder = key.startswith('up') or key == 'rgb_head'
                p.requires_grad_(decoder or mode != 'oracle')
                if not p.requires_grad:
                    continue
                rate = (.001 if mode == 'oracle' else .00025) if kind == 'w' else .02
                if key == 'rgb_head':
                    rate *= .2
                groups.append(dict(params=[p], lr=rate, initial_lr=rate))
        self.opt = torch.optim.Adam(groups)

    def update(self, native=False):
        start = time.perf_counter()
        selected = self.selected[self.step % len(self.selected)]
        item = next(v for v in self.cached if v['row']['id'] == selected['id'])
        x0, y0, x1, y1 = selected['box']
        if native:
            x0, y0, x1, y1 = 0, 0, 1280, 720
        x = item['source'][None, :, y0:y1, x0:x1]
        t = item['target'][None, :, y0:y1, x0:x1]
        gate = item['gate'][None, :, y0:y1:4, x0:x1:4]
        mix = 1. if self.mode in ('oracle', 'oracle_full') else max(0., 1-self.step/200) if self.mode == 'joint' else 0.
        p, predicted = self.net(x, gate, mix)
        m = 40
        loss, values = loss_colour(p[:, :, m:-m, m:-m], t[:, :, m:-m, m:-m])
        contour = F.smooth_l1_loss(predicted[:, :, 10:-10, 10:-10]/24, gate[:, :, 10:-10, 10:-10]/24)
        if self.mode != 'oracle':
            loss = loss+.08*contour
        for group in self.opt.param_groups:
            group['lr'] = group['initial_lr']*(.25+.75*.5*(1+math.cos(math.pi*self.step/max(self.steps, 1))))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5, error_if_nonfinite=True)
        assert torch.isfinite(loss) and grad > 0
        self.opt.step()
        self.step += 1
        torch.cuda.synchronize()
        return dict(step=self.step, loss=float(loss.detach()), values={k:float(v.detach()) for k,v in values.items()},
                    contour=float(contour.detach()), grad=float(grad), sample=selected['id'],
                    crop=[x0, y0, x1, y1], gate_mix=mix, seconds=time.perf_counter()-start)

    def save(self, path):
        torch.save(dict(model=self.net.state_dict(), optimizer=self.opt.state_dict(), step=self.step,
                        mode=self.mode, steps=self.steps, rng=self.rng.getstate(), quantized=self.net.quantized,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), path)

    def restore(self, path):
        s = torch.load(path, map_location='cpu', weights_only=True)
        self.net.load_state_dict(s['model']); self.opt.load_state_dict(s['optimizer']); self.step=s['step']
        self.net.quantized=s['quantized']; self.rng.setstate(s['rng'])
        torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])


@torch.inference_mode()
def review(tr, data, selected, dest, integer=False, oracle_comparison=True):
    dest.mkdir(exist_ok=False)
    previous=tr.net.quantized
    tr.net.quantized=integer
    actual=None
    if integer:
        raw=tr.net.export(); (dest/'coefficients-padded.bin').write_bytes(raw)
        assert len(raw)==20480
        actual=IntegerB1(runtime_from_image((dest/'coefficients-padded.bin').read_bytes()))
    ids={s['id'] for s in selected}|{c[0] for c in CROPS}
    panels, records={},[]
    for item in tr.cached:
        r=item['row']
        if r['id'] not in ids:
            continue
        x=item['source'][None]; gate=item['gate'][None, :, ::4, ::4]
        p,_=tr.net(x)
        oracle=tr.net(x, gate, 1.)[0] if oracle_comparison else p
        record=dict(id=r['id'], training=r['split']=='train', rgb_mae=float((p-item['target']).abs().mean()*128))
        if actual is not None:
            nodes,_=actual(x*128,True); ours=tr.net(x,all_nodes=True)
            error=max(float((v-nodes[k]).abs().max()) for k,v in ours.items()); assert error==0
            record['export_node_error']=error
            p=nodes['rgb_head']/128
        panels[r['id']]=dict(source=rgb_image(x[0]),target=rgb_image(item['target']),
                              output=rgb_image(p[0]),oracle=rgb_image(oracle[0]))
        records.append(record)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for name,cases in [('fixed_training',[(s['id'],s['box'],s['video']) for s in selected]), ('reviewed',CROPS)]:
        width=cases[0][1][2]-cases[0][1][0]; height=cases[0][1][3]-cases[0][1][1]
        columns=[('source','原片'),('target','冻结彩色目标'),('oracle','诊断：给定标签轮廓'),('output','实际：小网自己预测')]
        if not oracle_comparison:columns=[v for v in columns if v[0]!='oracle']
        board=Image.new('RGB',(width*len(columns),(height+32)*len(cases)),'#17202a');draw=ImageDraw.Draw(board)
        for i,(key,box,label) in enumerate(cases):
            for j,(kind,title) in enumerate(columns):
                board.paste(panels[key][kind].crop(box),(width*j,(height+32)*i+32))
                draw.text((width*j+4,(height+32)*i+4),title,font=font,fill='white')
        board.save(dest/(name+'.png'))
    oracle_error=None
    if integer:
        params,residual=decode(raw); op,rp=oracle_records(params); contract,_=contract_guard()
        sample=np.random.default_rng(54).integers(-128,128,(12,12,3),dtype=np.int16)
        reference=execute_integer(contract,sample.tolist(),op,rp)
        nodes,_=actual(torch.tensor(sample.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        oracle_error=max(int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(reference['tensors'][k])).max()) for k,v in nodes.items())
        assert oracle_error==0
    write_json(dest/'review.json',dict(step=tr.step,integer=integer,records=records,independent_oracle_error=oracle_error,
                                      oracle_intervention_diagnostic_only=oracle_comparison,visual_accepted=False,actual_board=False))
    tr.net.quantized=previous
    print('COLOUR_REVIEW',str(dest),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['oracle','oracle_full','joint','qat'])
    parser.add_argument('--steps',type=int,default=600);parser.add_argument('--parent',type=Path)
    args=parser.parse_args();configure_workspace();set_reproducible(SEED);contract_guard()
    data,selected,cached=load_data();dest=OUT/args.mode;dest.mkdir(parents=True,exist_ok=False)
    tr=Trainer(cached,selected,args.mode,args.steps)
    if args.parent:
        tr.net.load_state_dict(torch.load(args.parent,map_location='cpu',weights_only=True)['model'])
    if args.mode=='qat':
        tr.net.calibrate();tr.net.quantized=True
    write_json(dest/'binding.json',dict(source=str(SOURCE),source_sha256=file_hash(SOURCE),parent=str(args.parent),
        selected=selected,steps=args.steps,mode=args.mode,code_sha256=file_hash(__file__),same_graph=True,
        training='RGB + separately normalized teacher boundaries/flat areas/same-colour spatial pairs',
        inference='only unchanged B1, no external contour or colour operators'))
    shutil.copy2(__file__,dest/Path(__file__).name)
    initial=copy.deepcopy(tr.net.state_dict())
    a=tr.update(True);b=tr.update();tr.save(dest/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    fresh=Trainer(cached,selected,args.mode,args.steps);fresh.restore(dest/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    assert error<1e-5 and ref['sample']==got['sample']
    write_json(dest/'preflight.json',dict(native=a,crop=b,disk_resume_error=error,status='PASS'))
    print('COLOUR_PREFLIGHT_PASS',error,flush=True)
    del fresh,tr;torch.cuda.empty_cache();tr=Trainer(cached,selected,args.mode,args.steps)
    tr.net.load_state_dict(initial);tr.net.quantized=args.mode=='qat'
    review(tr,data,selected,dest/'initial',tr.net.quantized)
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<args.steps:
            row=tr.update();log.write(json.dumps(row)+'\n')
            if tr.step%50==0:print('LEARN_COLOUR',args.mode,tr.step,round(row['loss'],5),row['values'],flush=True)
            if tr.step%300==0 or tr.step==args.steps:
                tr.save(dest/f'checkpoint_{tr.step:06d}.pt')
                review(tr,data,selected,dest/f'step_{tr.step:06d}',tr.net.quantized)
    print('LEARN_COLOUR_COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
