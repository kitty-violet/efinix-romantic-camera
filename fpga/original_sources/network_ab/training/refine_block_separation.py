"""One controlled increase from the gentle v2, evaluated on identical inputs."""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .block_video_runtime import STILLS
from .fpga_video_resize import bilinear_2to3
from .gentle_block_boundaries import process
from .runtime import file_hash, write_json

PARAMETERS = dict(light_step=6., light_sharp=6., light_mix=.70,
                  chroma_step=4., chroma_sharp=5., chroma_mix=.44,
                  boundary_mix=1., max_lab_change=[2.8, 1.5, 1.5])


def main():
    cv2.setNumThreads(2)
    previous = STILLS.parent / 'gentle-global-boundaries-20261005-v2'
    out = STILLS.parent / 'gentle-global-boundaries-20261005-v3'
    out.mkdir(exist_ok=False)
    previous_experiment = json.loads((previous / 'experiment.json').read_text(encoding='utf-8'))
    cases = [('plaza_090', (0, 240, 420, 660)),
             ('plaza_330', (630, 280, 1050, 700)),
             ('stairs_090', (120, 300, 540, 720)),
             ('hallway_120', (0, 0, 420, 420))]
    board = Image.new('RGB', (840, 4 * 454), '#17202a')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    records = []
    for i, (name, box) in enumerate(cases):
        binding = next(r for r in previous_experiment['records'] if r['id'] == name)
        assert file_hash(binding['input']) == binding['input_sha256']
        base = np.array(Image.open(binding['input']))
        before = np.array(Image.open(previous / (name + '_after720.png')))
        assert np.array_equal(process(base, previous_experiment['parameters']), before)
        # Recompute from the same original; do not repeatedly quantize the v2 PNG.
        after = process(base, PARAMETERS)
        assert base.shape == before.shape == after.shape == (720, 1280, 3)
        Image.fromarray(after).save(out / (name + '_after720.png'))
        Image.fromarray(bilinear_2to3(after)).save(out / (name + '_1080p.png'))
        assert np.array_equal(np.array(Image.open(out / (name + '_after720.png'))), after)
        delta = np.abs(after.astype(np.int16) - before.astype(np.int16))
        baseline_delta = np.abs(after.astype(np.int16) - base.astype(np.int16))
        records.append(dict(id=name, input=binding['input'], input_sha256=binding['input_sha256'],
                            previous_output=str(previous / (name + '_after720.png')),
                            previous_output_sha256=file_hash(previous / (name + '_after720.png')),
                            previous_exact_reproduction=True,
                            mae_vs_v2=float(delta.mean()), p95_vs_v2=float(np.percentile(delta, 95)),
                            max_vs_v2=int(delta.max()), mae_vs_base=float(baseline_delta.mean()),
                            output_sha256=file_hash(out / (name + '_1080p.png'))))
        for col, (value, title) in enumerate([(before, '刚才的小幅版'), (after, '本次：色块分界再加强一档')]):
            draw.text((col * 420 + 5, i * 454 + 4), title, font=font, fill='white')
            board.paste(Image.fromarray(value).crop(box), (col * 420, i * 454 + 34))
    board.save(out / 'comparison.png')
    board.crop((0, 0, 840, 908)).save(out / 'trees_people_comparison.png')
    source = out / 'source_code'
    source.mkdir()
    for filename in ['refine_block_separation.py', 'gentle_block_boundaries.py',
                     'painting_parameters.py', 'block_separation.py', 'fpga_video_resize.py']:
        shutil.copy2(Path(__file__).with_name(filename), source / filename)
    html = (previous / 'preview.html').read_text(encoding='utf-8')
    html = html.replace('回到之前，只小幅收紧过渡', '色块分界再加强一档')
    html = html.replace('轻微加强已有色块的分界', '进一步加强已有色块的分界')
    html = html.replace('之前的画面', '刚才的小幅版').replace('本次小幅调整', '本次加强一档')
    html = html.replace("picture.src=scene.value+(mode==='before'?'_before1080p.png':'_1080p.png');",
                        "picture.src=(mode==='before'?'../gentle-global-boundaries-20261005-v2/':'')+scene.value+'_1080p.png';")
    html = html.replace("for(const suffix of ['_before1080p.png','_1080p.png']){const i=new Image();i.src=s+suffix}",
                        "for(const prefix of ['../gentle-global-boundaries-20261005-v2/','']){const i=new Image();i.src=prefix+s+'_1080p.png'}")
    (out / 'preview.html').write_text(html, encoding='utf-8')
    write_json(out / 'experiment.json', dict(parameters=PARAMETERS,
               previous_parameters=previous_experiment['parameters'], records=records,
               same_original_input=True, cumulative_processing=False,
               new_smoothing=False, new_outline=False, hard_palette=False,
               new_training=False, video_rendered=False, temporal_validated=False,
               source_sha256=file_hash(__file__)))
    (out / 'REPORT.md').write_text('''# 色块分界再加强一档

用户要求在刚才的小幅版上再突出、更分明。保持色阶间距、色调、平滑方式与空间处理范围，增加连续过渡收紧和现有边界收紧的强度。

相对v2：L混合0.40→0.70，ab混合0.24→0.44，边界单步混合0.50→1.00；Lab变化限幅1.8/1/1→2.8/1.5/1.5。仍为单次弱梯度边界更新，不追加描边或TV平滑，也不采用被否定的大色阶硬量化。各物体固定相同参数。

从同一份原生720p输入重新计算，避免在v2结果上重复分级。对照左侧是刚才实际交付的v2，不是更早版。四个场景均实际运行，旧输出按冻结参数逐像素复现，输入与输出SHA见experiment.json。修改于720p进行，显示使用项目整数双线性720→1080。四场景均为重复观察的回归样例，并非盲测。

本次为静态软件外观对照，没有新训练/新权重，没有整段视频或动态稳定性验证，也没有融合FPGA。图像差值仅描述变化量，不用于声称风格质量。
''', encoding='utf-8')
    manifest = [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=file_hash(p))
                for p in sorted(out.rglob('*')) if p.is_file()]
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(dict(output=str(out), files=len(manifest),
                          bytes=sum(r['bytes'] for r in manifest), records=records), ensure_ascii=False))


if __name__ == '__main__':
    main()
