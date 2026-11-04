"""Bounded same-method target comparison followed by author landscape model."""
import argparse
import importlib.util
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from .runtime import configure_workspace, write_json, file_hash
from .animegan_teacher import AnimeTeacher, REVISION, MODEL_SHA
from .wavelet_game_teacher import WaveletTeacher
from .probe_wavelet_game import tensor, picture


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['wct', 'anime'], required=True); args = ap.parse_args()
    root = configure_workspace(); run = root / 'runs/clear-style-control-20261003'
    cache = root / 'cache/animegan-probe' / REVISION
    out = run / args.stage; out.mkdir(exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cases = ['quiet_stairs', 'plaza_stone', 'park_people']
    inputs = root / 'runs/wavelet-game-probe-20261003'
    records = []
    if args.stage == 'wct':
        refpath = cache / 'results/AnimeGANv3_Hayao/4.jpg'
        combined = Image.open(refpath).convert('RGB')
        # Visually inspected author before/after: exclude the separator and left photo.
        box = (1088, 0, combined.width, combined.height)
        anime = ImageOps.fit(combined.crop(box), (1280, 720), Image.Resampling.LANCZOS)
        anime.save(out / 'anime_reference720.png')
        game = Image.open(inputs / 'references/garden_720.png').convert('RGB')
        game.save(out / 'game_reference720.png')
        net = WaveletTeacher(root)
        net.prepare_style('anime', tensor(anime)); net.prepare_style('game', tensor(game))
        variants = [('reconstruction', None, 0), ('game_07', 'game', .7), ('anime_07', 'anime', .7), ('game_10', 'game', 1.), ('anime_10', 'anime', 1.)]
        provenance = {'anime_reference': str(refpath), 'sha256': file_hash(refpath), 'crop_xyxy': box,
                      'role': 'author published stylized output, not original animation training ground truth',
                      'source_crop_size': [box[2]-box[0], box[3]], 'resized_to': [1280, 720]}
    else:
        net = AnimeTeacher(root, profile=out / 'gpu_profile')
        # Compare our exact preprocessing/output with the reviewed author utility on a supplied image.
        spec = importlib.util.spec_from_file_location('animegan_author_io', cache / 'deploy/test_by_onnx.py')
        author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
        sample = cache / 'inputs/imgs/v3_3.jpg'
        a, shape = author.load_test_data(str(sample), str(net.path))
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(sample)).astype(np.float32), (a.shape[2], a.shape[1])), cv2.COLOR_BGR2RGB)
        expected = net.session.run(None, {net.input: a})[0]
        expected = np.clip((expected.squeeze()+1)/2*255, 0, 255).astype(np.uint8)
        actual = net(rgb)
        delta = np.abs(actual.astype(float)-expected.astype(float))
        assert delta.max() <= 1
        Image.fromarray(rgb.clip(0,255).astype(np.uint8)).save(out / 'author_sample_source.png'); Image.fromarray(actual).save(out / 'author_sample_reproduced.png')
        provenance = {'model_sha256': MODEL_SHA, 'revision': REVISION, 'author_code_max_pixel_error': float(delta.max()),
                      'author_sample': str(sample), 'note': 'author code preprocessing/output agreement; not claim of pixel identity with JPEG illustrations'}
        variants = [('hayao_author', None, None)]
    for case in cases:
        d = out / case; d.mkdir(exist_ok=True)
        source = Image.open(inputs / case / 'source.png').convert('RGB'); source.save(d/'source.png')
        rows = [source]
        for name, style, alpha in variants:
            start = time.perf_counter()
            if args.stage == 'wct':
                y = net(tensor(source), style, ('encoder',) if style else (), alpha)
                im = picture(y)
            else:
                arr = net(np.array(source)); assert np.array_equal(arr, net(np.array(source))); im = Image.fromarray(arr)
            seconds = time.perf_counter()-start; im.save(d/(name+'.png')); rows.append(im)
            rec = {'case': case, 'variant': name, 'seconds': seconds, 'path': str(d/(name+'.png')), 'sha256': file_hash(d/(name+'.png'))}
            records.append(rec); print(rec, flush=True)
        board = Image.new('RGB', (640*len(rows), 390), '#161616'); draw = ImageDraw.Draw(board)
        for i, im in enumerate(rows):
            board.paste(im.resize((640, 360), Image.Resampling.LANCZOS), (640*i, 30))
            draw.text((i*640+8, 8), 'source' if i==0 else variants[i-1][0], fill='white')
        board.save(out/(case+'_overview.jpg'), quality=95)
    if args.stage == 'anime':
        provenance['execution_profile'] = net.finish_profile()
    write_json(out/'report.json', {'status': 'REQUIRES_VISUAL_REVIEW', 'records': records, 'provenance': provenance,
        'code_sha256': {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name('animegan_teacher.py')]}})


if __name__ == '__main__': main()
