"""Retain fractional boundary coverage before the final720 count decision.

Only existing up2_low_dw coefficients change. Native720 still rounds to the
same 200 RGB colours. This is an explicit decoder experiment, not training.
"""
import argparse
import copy
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .audit_b1_native_colour_regions import nearest_codes, spatial_permutation_audit
from .minimum_area_blocks import components
from .runtime import configure_workspace, contract_guard, file_hash, write_json


@torch.no_grad()
def install(net, span, kernel='center'):
    assert span in (1, 3, 5)
    key = 'up2_low_dw'
    shape = net.weight[key].new_zeros(3,3)
    if kernel == 'center': shape[1,1] = 127
    elif kernel == 'halfgauss':
        shape.copy_(shape.new_tensor([[1,2,1],[2,20,2],[1,2,1]])*6)
    elif kernel == 'gauss':
        shape.copy_(shape.new_tensor([[1,2,1],[2,4,2],[1,2,1]])*16)
    else: raise ValueError(kernel)
    mult = min(131071, round(131072*127/(float(shape.sum())*span)))
    gain = float(shape.sum())*mult/131072
    low = (127-span)/2
    net.weight[key][:15].zero_()
    net.weight[key][:15, 0] = shape*mult/131072
    net.bias[key][:15] = -low*gain
    net.up2_low_dw_m[:, :15] = mult


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--kernel', choices=('center','halfgauss','gauss'), default='center')
    args = ap.parse_args()
    configure_workspace(); contract_guard(); torch.set_num_threads(2); cv2.setNumThreads(1)
    dest = m.OUT/args.name; dest.mkdir(exist_ok=False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert state.get('group_native_count_decoder') == 'box'
    net = m.t.r.c.ColourB1(revive=False, source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']); net.package['residual_encoding'] = state['residual_encoding']; net.quantized = True
    initial = copy.deepcopy(net.state_dict()); original = net.export()
    rows = m.load_data()['validation']
    palette = (np.array(state['palette'], np.float32)*128+128).round().clip(0,255)
    patterns = ((torch.arange(32768)[:, None] >> torch.arange(15)[None]) & 1).float()
    z = torch.zeros(32768, 16, 1, 1); z[:, :15, 0, 0] = patterns*127
    expected = fragment(net, z, 'up1_high_dw')['rgb_head'].clone()
    images = {}; records = []
    for span in (1, 3, 5):
        net.load_state_dict(initial); install(net, span, args.kernel)
        blob = net.export()
        if span == 1 and args.kernel == 'center': assert blob == original, 'Control must exactly preserve all original bytes'
        constant_error = float((fragment(net, z, 'up1_high_dw')['rgb_head']-expected).abs().max())
        assert constant_error == 0
        spatial = spatial_permutation_audit(net)
        assert max(v['max_rgb_error'] for v in spatial) == 0
        sub = dest/f'span{span}'; sub.mkdir(); (sub/'coefficients-padded.bin').write_bytes(blob)
        actual = m.t.r.IntegerB1(m.t.r.runtime_from_image(blob), device='cpu'); stats = []
        for row in rows:
            rgb = m.t.r.read(row['path'])
            x = torch.tensor(rgb.transpose(2,0,1)[None].copy(), dtype=torch.float32)-128
            nodes, _ = actual(x, True); ours = net(x/128, all_nodes=True)
            error = max(float((v-ours[k]).abs().max()) for k,v in nodes.items()); assert error == 0
            im = m.t.r.c.rgb_image(nodes['rgb_head'][0]/128)
            im.save(sub/(row['id']+'.png')); images[span,row['id']] = im
            code, membership = nearest_codes(np.array(im), palette)
            assert membership['off_palette_fraction'] == 0, 'Do not add RGB transition levels'
            size = np.bincount(components(code).ravel())
            stats.append(dict(id=row['id'], all_node_error=error, small_area_fraction=float(size[size<256].sum()/size.sum()),
                              regions=len(size), **membership))
        saved = {k:v for k,v in state.items() if k not in ('optimizer','bounded_training_model','rng','torch_rng','cuda_rng')}
        saved.update(model=copy.deepcopy(net.state_dict()), manual_boundary_phase_span=span,
                     manual_decoder_change=span!=1 or args.kernel!='center', boundary_field_kernel=args.kernel,
                     parent_checkpoint=args.checkpoint,
                     parent_sha256=file_hash(args.checkpoint), added_training_steps=0,
                     visual_acceptance=False, minimum_area_inference_guarantee=False)
        torch.save(saved, sub/'candidate.pt')
        records.append(dict(span=span, constant_32768_error=constant_error, spatial=spatial,
                            bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), stats=stats))
        print('NATIVE_BOUNDARY_PHASE', span, stats, flush=True)
    board = Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a'); draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14); by_id = {r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics = [Image.open(by_id[key]['target'])]+[images[span,key] for span in (1,3,5)]
        first = '原硬边界' if args.kernel == 'center' else '邻域字段后硬判定'
        for j,(im,title) in enumerate(zip(pics,('离线目标',first,'保留3级边界位置','保留5级边界位置'))):
            board.paste(im.crop(box),(420*j,314*i+34)); draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png'); shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=args.checkpoint,parent_sha256=file_hash(args.checkpoint),records=records,
        classifier_changed=False,manual_decoder_coefficient_change=True,added_training_steps=0,kernel=args.kernel,
        added_inference_operations=0,external_filter=False,actual_board=False,visual_acceptance=False,
        note='Fractional360 boundary coverage; unchanged integer720 palette; no hard area guarantee'))


if __name__ == '__main__':
    main()
