"""Freeze the requested black-and-white style with standalone inference."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('E:/fpga/model-training-workspace/runs')
SOURCE = ROOT/'region-detail-deploy-20261004/native-teacher-strokes-v1'
DEST = ROOT/'style-library/black-white-line-v1-20261004'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    for name in ('contour_fp32.pt', 'checkpoint_001600.pt', 'config.json', 'binding.json', 'data.json'):
        shutil.copy2(SOURCE/name, DEST/name)
    shutil.copytree(SOURCE/'source', DEST/'training-source')
    for name in ('contours_12s_1080p.mp4', 'comparison.png', 'weight_verification.json', 'video.json'):
        shutil.copy2(SOURCE/'delivery'/name, DEST/name)
    code = Path(__file__).parent
    shutil.copy2(code/'native_contour_net.py', DEST/'native_contour_net.py')
    shutil.copy2(code/'fpga_video_resize.py', DEST/'fpga_video_resize.py')
    source = (code/'train_context_contours.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'ContextContourNet')
    (DEST/'model.py').write_text('from torch.nn import functional as F\nfrom native_contour_net import NativeContourNet, Block\n\n'+ast.get_source_segment(source, cls)+'\n', encoding='utf-8')
    (DEST/'infer.py').write_text('''"""Standalone archived style; requires the existing torch/numpy/Pillow environment."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import ContextContourNet
from fpga_video_resize import area_3to2, bilinear_2to3

def load(device='cuda'):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    net=ContextContourNet(difference=False).to(device).eval()
    state=torch.load(Path(__file__).parent/'contour_fp32.pt',map_location='cpu',weights_only=True)
    assert state['folded']
    net.load_state_dict(state['model'])
    return net

@torch.no_grad()
def predict(net, rgb):
    is1080=rgb.shape==(1080,1920,3)
    if is1080:rgb=area_3to2(rgb)
    if rgb.shape!=(720,1280,3):raise ValueError('Input must be RGB 1280x720 or 1920x1080')
    x=torch.from_numpy((rgb.astype(np.float32)-128)/128).permute(2,0,1)[None].to(next(net.parameters()).device)
    z=net(x).sigmoid()[0,0].cpu().numpy()
    mono=np.uint8(np.clip(1-z,0,1)*255)
    out=np.repeat(mono[:,:,None],3,axis=2)
    return bilinear_2to3(out) if is1080 else out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--device',default='cuda');a=p.parse_args()
    Image.fromarray(predict(load(a.device),np.array(Image.open(a.input).convert('RGB')))).save(a.output)
''', encoding='utf-8')
    data=json.loads((SOURCE/'data.json').read_text(encoding='utf-8'))
    tests=DEST/'tests';tests.mkdir()
    rows=[]
    for row in data['validation']:
        name=row['id'];shutil.copy2(row['source'], tests/(name+'_input.png'))
        shutil.copy2(SOURCE/'eval-001600'/(name+'_probability.png'), tests/(name+'_expected.png'))
        rows.append(name)
    (DEST/'verify.py').write_text('''from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from infer import load,predict
torch.set_num_threads(4)
root=Path(__file__).parent
net=load();records=[]
for path in sorted((root/'tests').glob('*_input.png')):
    actual=predict(net,np.array(Image.open(path).convert('RGB')))
    expected=np.array(Image.open(path.with_name(path.name.replace('_input','_expected'))).convert('RGB'))
    error=int(np.abs(actual.astype(int)-expected.astype(int)).max())
    assert error==0,(path,error)
    records.append(dict(id=path.stem,max_uint8_error=error))
report=dict(status='PASS',device=torch.cuda.get_device_name(),standalone_imports=True,records=records)
(root/'archive_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
''', encoding='utf-8')
    (DEST/'README.md').write_text('''# 黑白线条风 v1（备用）

用户于2026-10-04要求单独保存。本包冻结实际训练输出；后续彩色实验使用其他目录。

- `contour_fp32.pt`：折叠普通卷积的真实 FP32 推理权重，保持原文件不变。
- `infer.py` / `model.py` / `native_contour_net.py`：独立推理代码，不需要导入项目训练模块。
- `checkpoint_001600.pt` / `training-source`：原训练状态与源码快照，供研究续训溯源；原数据仍按 data.json 引用，不重复打包190张训练图。
- `contours_12s_1080p.mp4`：实际12秒、360帧、30fps、1920×1080示例。
- `tests` / `verify.py`：6张完整720p样例及冻结输出，可验证归档独立重现。
- `manifest.json`：包内文件SHA-256及来源；`archive_verification.json`：实际CUDA重载结果。

PowerShell（使用项目已经安装的隔离环境，无需下载）：

```powershell
Set-Location 'E:\\fpga\\model-training-workspace\\runs\\style-library\\black-white-line-v1-20261004'
& 'E:\\fpga\\model-training-workspace\\env-probes\\animegan\\Scripts\\python.exe' -B infer.py tests/val_stairs_090_input.png example.png
```

输入仅原片RGB；1080输入按整数area缩至720，网络输出轮廓logit，经sigmoid映射白底黑线，再按整数bilinear回1080。没有推理Canny、形态学、细化或视频混帧。

网络9529个训练参数、9032个卷积权重。此包是备用风格资产，未量化或通过当前FPGA的缓冲、命令、RTL、资源和时序验收，不是可直接刷板包，也不改变单风格基线。草丛线条仍较碎，稳定性未完成验收。
''', encoding='utf-8')
    (DEST/'preview.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>黑白线条风 · 已封存</title><style>body{background:#171d27;color:#edf1f6;font:17px system-ui;max-width:1280px;margin:24px auto}video,img{width:100%}a{color:#9edfea}</style><h1>黑白线条风 v1 · 备用归档</h1><p>2026-10-04封存。实际12秒连续视频，1080→720网络→1080。权重和独立推理代码已保存；未通过上板验收。</p><video controls preload="metadata" src="contours_12s_1080p.mp4"></video><p><a href="contour_fp32.pt">真实FP32权重</a> · <a href="README.md">复现说明</a> · <a href="archive_verification.json">独立重载验证</a> · <a href="manifest.json">文件清单</a></p><img src="comparison.png" alt="封存前的教师和轮廓对照图"></html>''', encoding='utf-8')
    # Isolated mode ignores PYTHONPATH; -c inserts only this standalone package.
    finalize()


def finalize():
    command="import sys,runpy;sys.path.insert(0,"+repr(str(DEST))+");runpy.run_path("+repr(str(DEST/'verify.py'))+",run_name='__main__')"
    subprocess.run([sys.executable,'-I','-B','-c',command],cwd=DEST,check=True)
    files={p.relative_to(DEST).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in DEST.rglob('*') if p.is_file() and p.name!='manifest.json'}
    manifest=dict(style_id='black-white-line-v1',display_name='黑白线条风',purpose='备用独立风格归档',source=str(SOURCE),
        source_weight_sha256=sha(SOURCE/'contour_fp32.pt'),inference_weight_identical=sha(SOURCE/'contour_fp32.pt')==sha(DEST/'contour_fp32.pt'),
        files=files,board_validated=False,total_bytes=sum(v['bytes'] for v in files.values()))
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ARCHIVE_COMPLETE',DEST,manifest['total_bytes'],flush=True)


if __name__=='__main__':
    if '--finalize' in sys.argv:finalize()
    else:main()
