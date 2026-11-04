"""Export actual learned FP32 state and verify a standalone inference process."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from .runtime import configure_workspace, file_hash, write_json


def main():
    root=configure_workspace();run=root/'runs/clear-style-control-20261003';out=run/'final-2000';out.mkdir(exist_ok=True)
    checkpoint=run/'student_train/checkpoint_002000.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['step']==2000 and all(v.dtype==torch.float32 for v in state['ema'].values())
    package={'schema':'surface-appearance-fp32-weights-v1','config':state['config']['model'],'generator':state['ema'],
             'training_steps':state['step'],'checkpoint_sha256':file_hash(checkpoint),'binding':state['binding'],
             'role':'clear animation diagnostic student, not Dying Light or FPGA-approved weights'}
    weights=out/'shared_style_fp32.pt';torch.save(package,weights)
    for n in ['coherent_shared_model.py','coherent_surface_model.py']:shutil.copy2(Path(__file__).with_name(n),out/n)
    old=root/'runs/coherent-surface-20261002/final-4850/infer_image.py'
    code=old.read_text(encoding='utf-8').replace('result=cv2.resize(rgb,(1920,1080),interpolation=cv2.INTER_LANCZOS4)',
                                               'result=np.array(Image.fromarray(rgb).resize((1920,1080),Image.Resampling.LANCZOS))')
    (out/'infer_image.py').write_text(code,encoding='utf-8');compile(code,str(out/'infer_image.py'),'exec')
    source=run/'video_sources/new_hallway_source_1080p.mp4';cap=cv2.VideoCapture(str(source));ok,bgr=cap.read();cap.release();assert ok
    image=out/'export_check_input.png';Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(image)
    prediction=out/'export_check_output.png'
    subprocess.run([sys.executable,'-B',str(out/'infer_image.py'),'--input',str(image),'--output',str(prediction)],check=True)
    expected=np.array(Image.open(run/'student_videos/new_hallway_000_preencode1080.png')).astype(np.int16)
    actual=np.array(Image.open(prediction)).astype(np.int16);delta=np.abs(actual-expected)
    assert delta.max()<=1 and (delta!=0).mean()<.0001
    write_json(out/'verification.json',{'status':'PASS','independent_process':True,'max_pixel_error':int(delta.max()),
                'changed_value_fraction':float((delta!=0).mean()),'mean_error':float(delta.mean()),'weights_sha256':file_hash(weights),
                'checkpoint_sha256':file_hash(checkpoint),'dtype':'FP32','training_steps':2000,
                'files':{p.name:file_hash(p) for p in out.glob('*.py')}})
    (out/'README.md').write_text('''# 明显动画风格对照：2000步FP32共享CNN

这是本项目实际训练的共享CNN权重，目标是经过平滑区域假边缘抑制的AnimeGANv3场景教师。它用于诊断明确风格能否学入共享CNN；不是消光写实目标达标，也不是已验证的FPGA权重。

教师来源：[TachibanaYoshino/AnimeGANv3](https://github.com/TachibanaYoshino/AnimeGANv3)，官方v1.1.0的Hayao_36模型；源码固定版本73ada3910dafa87b4cc972bcd80ebfebce9f3abe。作者模型与本目录的2000步学生权重是不同产物；本项目没有从零训练作者网络。

推理只需本目录权重和Python模型源码，已有PyTorch/Pillow/Numpy/OpenCV环境即可；不调用ONNX、教师模型、材质库、光流或平滑区域修正规则。网络为SurfaceAppearanceNet，32/64/128通道、6个残差块，共2,116,710参数。输入输出归一化和固定缩放见infer_image.py。

输入为已转换的1920×1080 SDR RGB；PIL bicubic降至1280×720，FP32 CNN推理，PIL Lanczos升至1920×1080。手机原始HLG视频须按项目既有固定转色流程处理，不能把任意HDR像素当SDR。

独立推理示例：

```powershell
& 'E:\\fpga\\model-training-workspace\\env\\Scripts\\python.exe' -B 'E:\\fpga\\model-training-workspace\\runs\\clear-style-control-20261003\\final-2000\\infer_image.py' --input 'E:\\fpga\\model-training-workspace\\runs\\clear-style-control-20261003\\final-2000\\export_check_input.png' --output 'E:\\fpga\\model-training-workspace\\runs\\clear-style-control-20261003\\final-2000\\reproduced.png'
```

verification.json记录独立进程重载后与实际视频编码前PNG的像素对照。完整训练恢复使用上级student_train/checkpoint_002000.pt；该断点含优化器、EMA与随机状态。三个场景视频均未用于梯度训练，但已参与本轮教师/效果评审，不声称独立盲测或普适泛化。
''',encoding='utf-8')
    print('EXPORTED',weights,file_hash(weights),flush=True)


if __name__=='__main__':main()
