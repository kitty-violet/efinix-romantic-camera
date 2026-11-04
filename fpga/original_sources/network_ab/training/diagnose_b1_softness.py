"""Separate codec, EMA and initializer observations without claiming capacity limits."""
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace, write_json
from .train_b1_distillation import as_tensor, to_rgb
from .verify_b1_delivery import edge_mean


def main():
    root = configure_workspace()
    run = root / 'runs/b1-distillation-20261003'
    out = run / 'softness-diagnosis'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    data = json.loads((run / 'data/manifest.json').read_text(encoding='utf-8'))
    state = torch.load(run / 'train/checkpoint_002000.pt', map_location='cpu', weights_only=True)
    raw = build_fp32_model().eval()
    raw.load_state_dict(state['model'])
    rows = []
    with torch.inference_mode():
        for row in data['validation']:
            source = np.array(Image.open(row['source']).convert('RGB'))
            target = np.array(Image.open(row['target']).convert('RGB'))
            init = np.array(Image.open(run / 'train/validation_000000' / (row['id'] + '.png')))
            ema = np.array(Image.open(run / 'train/validation_002000' / (row['id'] + '.png')))
            actual = to_rgb(raw(as_tensor(source)[None])[0])
            sampled = source[::4, ::4].repeat(4, axis=0).repeat(4, axis=1)
            records = {'id': row['id'], 'initial_exact_stride4_repeat': bool(np.array_equal(init, sampled)),
                       'raw_ema_mae_rgb255': float(np.abs(actual.astype(np.float32)-ema).mean()),
                       'edges': {name: edge_mean(a) for name, a in
                                 [('source', source), ('teacher', target), ('initial', init),
                                  ('raw_2000', actual), ('ema_2000', ema)]}}
            rows.append(records)
            Image.fromarray(actual).save(out / (row['id'] + '_raw720.png'))
            # Same 1:1 central 512x320 crop: no resizing, sharpening or compositing output.
            x, y, w, h = 384, 200, 512, 320
            panel = Image.new('RGB', (w*4, h+28), '#151515')
            draw = ImageDraw.Draw(panel)
            for i, (label, a) in enumerate([('SOURCE', source), ('TEACHER TARGET', target),
                                            ('B1 RAW 2000', actual), ('B1 EMA 2000', ema)]):
                panel.paste(Image.fromarray(a[y:y+h, x:x+w]), (i*w, 28))
                draw.text((i*w+8, 8), label, fill='white')
            panel.save(out / (row['id'] + '_native_crop.png'))
    write_json(out / 'report.json', {
        'status': 'OBSERVATIONS_ONLY_NOT_CAPACITY_PROOF', 'records': rows,
        'user_assessment': 'User reviewed video: quality is blurred, animation style absent. Visual acceptance FAILED.',
        'interpretation': 'Initialization is stride-4 RGB sampling, not a lossless identity. Raw and EMA comparison separates averaging from softness. These checks do not isolate loss/data/architecture capacity causes.',
        'next_control': 'Keep contract fixed; compare identity reconstruction and paired style fitting on the same sharp native-scale patch, then source-isolated video. Do not use RGB MAE alone as acceptance.'})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
