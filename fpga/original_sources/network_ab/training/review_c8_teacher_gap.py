"""Read-only weight audit and native-pixel teacher/student visual comparison."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn

from .runtime import configure_workspace, file_hash, write_json
from .probe_fullres_c8 import FullresC8
from .train_b1_distillation import as_tensor, to_rgb
from .trainable_animegan import TrainableAnimeGAN
from .smooth_surface_guard import correct_surface


def main():
    begin = time.perf_counter()
    root = configure_workspace()
    out = root/'runs/c8-teacher-gap-review-20261003'
    out.mkdir(exist_ok=False)
    color = root/'runs/fullres-c8-stylepush-color-20261003'
    high = root/'runs/fullres-c8-stylepush-20261003'
    manifest = json.loads((color/'data_manifest.json').read_text(encoding='utf-8'))
    rows = {r['id']: r for r in manifest['validation']}
    bound = {str(Path(__file__)): file_hash(__file__)}
    images = {}
    for key, row in rows.items():
        images[key] = {}
        for variant, path in [('source', Path(row['source'])), ('teacher', Path(row['target'])),
                              ('high250', high/'train/validation_000250'/f'{key}.png'),
                              ('color2000', color/'train/validation_002000'/f'{key}.png')]:
            digest = file_hash(path)
            if variant in ('source', 'teacher'):
                assert digest == row[('source' if variant == 'source' else 'target')+'_sha256']
            bound[str(path)] = digest
            images[key][variant] = np.array(Image.open(path).convert('RGB'))
            assert images[key][variant].shape == (720, 1280, 3)

    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    regions = [('石柱：表面是否成为连贯色块', 'val_stairs_090', (424, 320, 744, 576)),
               ('植物：是否把细碎叶片归并成形状', 'val_stairs_090', (180, 430, 500, 686)),
               ('人物：衣服、面部与背景轮廓', 'val_plaza_090', (80, 448, 400, 704))]
    crop_records = []
    for name, columns in [('teacher_students', [('teacher', '教师'), ('high250', '高强度试验 / 250步'), ('color2000', '保色加强 / 2000步')]),
                          ('source_teacher_student', [('source', '原片'), ('teacher', '教师'), ('color2000', '保色加强 / 2000步')])]:
        panel = Image.new('RGB', (960, 946), '#171b24')
        draw = ImageDraw.Draw(panel)
        for j, (_, label) in enumerate(columns):
            draw.text((j*320+8, 8), label, font=font, fill='white')
        for i, (label, key, box) in enumerate(regions):
            y = 42+i*288
            draw.text((8, y), label, font=small, fill='#bcd0e8')
            for j, (variant, _) in enumerate(columns):
                crop = Image.fromarray(images[key][variant]).crop(box)
                panel.paste(crop, (j*320, y+28))
                assert np.array_equal(np.array(panel.crop((j*320, y+28, j*320+320, y+284))), np.array(crop))
                crop_records.append(dict(panel=name, id=key, variant=variant, box=list(box)))
        draw.text((8, 920), '同一帧、同一坐标，720p原生像素直接裁切；未缩放、未锐化、未调色。', font=small, fill='#bcd0e8')
        panel.save(out/f'{name}.png')
    write_json(out/'image_binding.json', dict(inputs=bound, crops=crop_records, crop_pixels_exact=True))
    print('PANELS_COMPLETE', flush=True)

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audits = {}
    models = {}
    for tag, run in [('high250', high), ('color2000', color)]:
        p = run/'final/fullres_c8_fp32.pt'
        net = FullresC8().cuda().eval()
        net.load_state_dict(torch.load(p, map_location='cpu', weights_only=True)['model'])
        models[tag] = net
        layers = {}
        def capture(name):
            def hook(module, args, result):
                maximum = result.detach().abs().amax((0, 2, 3)).cpu().numpy()
                layers[name] = np.maximum(layers.get(name, np.zeros(8)), maximum)
            return hook
        hooks = [m.register_forward_hook(capture(f'body.{i}')) for i, m in enumerate(net.body)]
        matches = []
        with torch.inference_mode():
            for key in rows:
                pred = net(as_tensor(images[key]['source'])[None].cuda())
                error = int(np.abs(to_rgb(pred[0]).astype(int)-images[key][tag].astype(int)).max())
                assert error == 0, (tag, key, error)
                matches.append(dict(id=key, exported_weight_vs_saved_png_max=error))
        for h in hooks:
            h.remove()
        gradients = []
        for _, key, box in regions:
            x0, y0, x1, y1 = box
            x = as_tensor(images[key]['source'][y0:y1, x0:x1])[None].cuda()
            target = as_tensor(images[key]['teacher'][y0:y1, x0:x1])[None].cuda()
            net.zero_grad(set_to_none=True)
            torch.nn.functional.l1_loss(net(x), target).backward()
            gradients.append(dict(id=key, box=list(box),
                inactive_branch_max=max(float(p.grad[6:8].abs().max()) for n, p in net.named_parameters() if n.startswith('body.')),
                head_inactive_input_grad_max=float(net.head.weight.grad[:, 6:8].abs().max()),
                all_parameter_grad_norm=float(torch.stack([p.grad.norm() for p in net.parameters()]).norm())))
        audits[tag] = dict(weights=str(p), sha256=file_hash(p), parameters=sum(p.numel() for p in net.parameters()),
            nominal_channels=8, receptive_field=11, source_images=6,
            zero_output_channels={k: np.flatnonzero(v == 0).tolist() for k, v in layers.items()},
            per_layer_channel_max={k: v.tolist() for k, v in layers.items()},
            head_inactive_input_weight_max=float(net.head.weight[:, 6:8].abs().max()),
            gradient_probes=gradients, reload_image_matches=matches)
        print('STUDENT_AUDIT', tag, audits[tag]['zero_output_channels'], flush=True)
    write_json(out/'student_audit.json', audits)

    teacher_path = Path(manifest['teacher'])
    assert file_hash(teacher_path) == manifest['teacher_sha256']
    teacher = TrainableAnimeGAN(root).cuda().eval()
    teacher.load_state_dict(torch.load(teacher_path, map_location='cpu', weights_only=True)['generator'])
    convs = [dict(name=n, in_channels=m.in_channels, out_channels=m.out_channels,
                  kernel=list(m.kernel_size), stride=list(m.stride), groups=m.groups)
             for n, m in teacher.named_modules() if isinstance(m, nn.Conv2d)]
    teacher_info = dict(weights=str(teacher_path), sha256=file_hash(teacher_path),
        parameters=sum(p.numel() for p in teacher.parameters()), convolutions=convs,
        spatial_global_pool_modules=[n for n, m in teacher.named_modules() if 'GlobalAveragePool' in type(m).__name__],
        surface_guard='Fixed R48 correction is in stored targets, outside the learned teacher network.')
    context = []
    originals = {}
    # Diagnostic intervention only; these synthetic inputs are not training/evaluation images.
    with torch.inference_mode():
        for label, key, xy in [('stone', 'val_stairs_090', (548, 501)),
                               ('plants', 'val_stairs_090', (356, 578)),
                               ('person', 'val_plaza_090', (245, 598))]:
            rgb = images[key]['source']
            x, y = xy
            changed = cv2.GaussianBlur(rgb, (0, 0), 8)
            changed[y-16:y+17, x-16:x+17] = rgb[y-16:y+17, x-16:x+17]
            original = as_tensor(rgb)[None].cuda()
            modified = as_tensor(changed)[None].cuda()
            if key not in originals:
                originals[key] = teacher(original)
                raw = (originals[key][0].permute(1,2,0).cpu().numpy()*128+128).clip(0,255).astype(np.uint8)
                corrected = correct_surface(rgb, raw)
                delta = np.abs(corrected.astype(int)-images[key]['teacher'].astype(int))
                teacher_info.setdefault('target_reproduction', []).append(dict(id=key, max_rgb_error=int(delta.max()), mean_rgb_error=float(delta.mean())))
                assert delta.max() <= 1, (key, delta.max())
            before = originals[key]
            after = teacher(modified)
            region = lambda z: z[:, :, y-5:y+6, x-5:x+6]
            d = (region(before)-region(after)).abs()*128
            differences = {}
            for tag, net in models.items():
                error = float((region(net(original))-region(net(modified))).abs().max()*128)
                assert error == 0, (tag, error)
                differences[tag] = error
            context.append(dict(region=label, id=key, center_xy=list(xy), unchanged_input_square=33,
                output_square=11, teacher_raw_mean_rgb_change=float(d.mean()), teacher_raw_max_rgb_change=float(d.max()),
                student_max_rgb_change=differences))
            print('CONTEXT', context[-1], flush=True)
    write_json(out/'teacher_audit.json', teacher_info)
    write_json(out/'context_probe.json', dict(scope='Synthetic context sensitivity, not style quality, generalization or sole-cause proof',
        intervention='Keep central 33x33 original; Gaussian blur sigma8 elsewhere; measure central11x11', records=context))
    write_json(out/'completion.json', dict(status='REVIEW_AND_DIAGNOSTIC_COMPLETE_NO_TRAINING',
        seconds=time.perf_counter()-begin, cuda_device=torch.cuda.get_device_name(), optimizer_updates=0,
        source_sha256=file_hash(__file__), panel_pixel_verification=True,
        shared_contract_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json')))
    print('COMPLETE', time.perf_counter()-begin, flush=True)


if __name__ == '__main__':
    main()
