"""Compare matched TRAIN-only count-objective probes after actual byte reload."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .train_b1_count_regions import count_targets
from .fit_b1_sixteen_regions import palette_key, region_class_map
from .minimum_area_blocks import components, boundary_mask
from .runtime import write_json, file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=('unbalanced','pure'),default='unbalanced')
    args=ap.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    balanced = m.OUT/'native-region-objective-matched-probe'
    plain = m.OUT/('count-unbalanced-matched-probe' if args.variant=='unbalanced' else 'count-pure-footprint-matched-probe')
    ar = json.loads((balanced/'result.json').read_text(encoding='utf-8'))
    br = json.loads((plain/'result.json').read_text(encoding='utf-8'))
    assert ar['warm_sha256'] == br['warm_sha256'] and ar['train_ids'] == br['train_ids']
    sources = [('balanced', balanced/'control/candidate.pt'), (args.variant, plain/'candidate.pt')]
    ref = torch.load(sources[0][1], map_location='cpu', weights_only=True)
    palette = np.array(ref['palette'])*128+128; pk = palette_key(torch.tensor(ref['palette']))
    rows = {r['id']:r for r in m.load_data()['training']}
    samples = []
    for key,(xx,yy) in zip(ar['train_ids'],((768,288),(416,224))):
        row = rows[key]
        rgb = m.t.r.read(row['path'])[yy:yy+320,xx:xx+320]
        code = region_class_map(row['target'],row['region'],pk)[yy:yy+320,xx:xx+320]
        samples.append((key,rgb,code))
    pictures = {}; records = []
    for name,path in sources:
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert state['step'] == 150
        net = m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
        net.load_state_dict(state['model']); net.package['residual_encoding'] = state['residual_encoding']; net.quantized = True
        actual = m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        for key,rgb,code in samples:
            x = torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            nodes,_ = actual(x,True)
            pred = nodes['up2_high_dw'][0,:3].numpy()
            predicted = (pred[0]*25+pred[1]*5+pred[2]).astype(np.int32)
            lab = components(predicted); size = np.bincount(lab.ravel())
            interior = cv2.dilate(boundary_mask(code),np.ones((7,7),np.uint8)) == 0
            roi = np.zeros(code.shape,bool); roi[40:-40,40:-40] = True
            valid = interior & roi
            truth = count_targets(torch.tensor(code.astype('int64'))[None])
            pure = torch.stack([v.amax(1)>.999 for v in truth]).all(0)[:,20:-20,20:-20]
            wanted = truth[0].argmax(1)*25+truth[1].argmax(1)*5+truth[2].argmax(1)
            bit = nodes['up2_low_dw'][:,:15] > 63.5
            got = bit[:,:7].sum(1)*25+bit[:,7:11].sum(1)*5+bit[:,11:15].sum(1)
            exact360 = (got[:,20:-20,20:-20][pure] == wanted[:,20:-20,20:-20][pure]).float().mean()
            pictures[name,key] = m.t.r.c.rgb_image(nodes['rgb_head'][0]/128)
            records.append(dict(name=name,id=key,checkpoint_sha256=file_hash(path),
                exact360_pure=float(exact360),exact720_roi=float((predicted[roi]==code[roi]).mean()),
                interior_wrong_fraction=float((predicted[valid]!=code[valid]).mean()),
                small_area_roi_fraction=float((size[lab][roi]<256).mean()),
                area_components_measured_on_full_crop=True))
    board = Image.new('RGB',(1280,356*2),'#17202a'); draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,rgb,code) in enumerate(samples):
        pics = [Image.fromarray(rgb), Image.fromarray(palette[code].round().clip(0,255).astype(np.uint8)),
                pictures['balanced',key], pictures[args.variant,key]]
        title='取消频率加权·整数' if args.variant=='unbalanced' else '仅纯色采样格监督·整数'
        for j,(im,title) in enumerate(zip(pics,('TRAIN原图','同一训练目标','原计数监督·整数',title))):
            board.paste(im,(j*320,i*356+36)); draw.text((j*320+4,i*356+6),title,font=font,fill='white')
    board.save(plain/'matched-comparison.png')
    write_json(plain/'matched-review.json',dict(records=records,train_only=True,validation_fitted=False,
        same_warm_sha256=ar['warm_sha256'],outer_steps=150,optimizer='bounded FP32 L-BFGS max_iter3 max_eval5 history40 strong_wolfe',
        balanced_closures=ar['results'][0]['calls'],probe_closures=br['closure_calls'],variant=args.variant,
        actual_reloaded_integer=True,extra_inference_operations=0,visual_acceptance=False))
    print('COUNT_OBJECTIVE_MATCHED',records,flush=True)


if __name__ == '__main__': main()
