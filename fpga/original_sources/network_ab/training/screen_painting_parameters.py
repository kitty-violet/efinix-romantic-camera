"""Research-stage stills only; integration awaits user's visual approval."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .painting_parameters import transform,soft_levels,PRESETS,TONE

ROOT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004')
OUT=ROOT/'painting-parameters-20261004-v2'
OLD=ROOT/'seven-structured-fill-cool-v2/delivery'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return np.array(Image.open(p).convert('RGB'))


def contacts(out):
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),
           ('val_plaza_090',(80,400,400,680)),('val_new_hallway_120',(0,0,320,280))]
    b=Image.new('RGB',(1600,len(cases)*316),'#17202a');d=ImageDraw.Draw(b)
    for j,(key,box) in enumerate(cases):
        for i,(name,title) in enumerate([('teacher','教师参考'),('student','实际模型输出'),('tone','A 仅色调明暗'),('poster','B 哑光色块'),('poster_coarse','C 较粗色块')]):
            d.text((320*i+5,316*j+5),title,font=f,fill='white')
            b.paste(Image.open(out/(key+'_'+name+'.png')).crop(box),(320*i,316*j+34))
    b.save(out/'comparison.png')
    b=Image.new('RGB',(1280,428),'#17202a');d=ImageDraw.Draw(b)
    for i,(name,title) in enumerate([('teacher','教师参考'),('student','模型输出'),('poster','B 哑光色块'),('poster_coarse','C 较粗色块')]):
        d.text((320*i+5,5),title,font=f,fill='white')
        b.paste(Image.open(out/('val_stairs_090_'+name+'.png')).crop((150,340,470,720)),(320*i,34))
    b.save(out/'grass_focus.png')


def page(out):
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图片参数化绘画质感实验</title><style>body{background:#17202b;color:#eef3f8;font:17px system-ui;margin:24px auto;max-width:1600px;padding:0 20px}p{line-height:1.7}button,select{font:inherit;padding:9px;margin:5px;background:#30455c;color:white;border:1px solid #b2c8d9}a{color:#9edcf7}img{max-width:100%;display:block}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.note{color:#ffe0a6}</style>
<h1>从模型图片出发：固定参数绘画质感实验</h1><p>目标观感：哑光、不透明水彩／广告色式色块，主要结构清晰，块内细碎色差减少。先看图片，再决定是否将步骤融入模型。</p><p class="note">本轮没有重新训练。所有方案均处理同一份实际学生RGB输出；教师只用于右侧比较，不进入转换算法。</p>
<select id="scene" onchange="show()"><option value="val_stairs_090">台阶与草丛</option><option value="val_stairs_210">台阶后段</option><option value="val_plaza_090">人物与树木</option><option value="val_plaza_210">广场后段</option><option value="val_new_hallway_120">强光走廊</option><option value="val_new_hallway_210">走廊后段</option></select>
<select id="variant" onchange="show()"><option value="poster">B 哑光色块</option><option value="poster_coarse">C 较粗色块</option><option value="tone">A 仅色彩与明暗</option><option value="watercolor">D OpenCV绘画化对照</option><option value="student">实际模型输出</option><option value="previous">上一版处理结果</option><option value="source">原片</option></select>
<div class="row"><div><p id="name">B 哑光色块</p><img id="result" alt="固定参数处理结果"></div><div><p>教师参考</p><img id="teacher" alt="教师参考"></div></div>
<p><button onclick="pick('student')">模型原输出</button><button onclick="pick('tone')">A 仅调色</button><button onclick="pick('poster')">B 哑光色块</button><button onclick="pick('poster_coarse')">C 较粗色块</button></p>
<h2>同位置局部比较</h2><img src="comparison.png" alt="教师、实际输出与ABC对照">
<p>A用作对照，B/C加入保边表面简化、连续明暗色阶归并与主要暗边强化。D是现成绘画化方法的对照，不当作目标答案。所有PNG为真实1280×720处理结果。</p>
<p><a href="RESEARCH_AND_RESULTS.md">画质判断、来源、参数与限制</a> · <a href="experiment.json">冻结参数和输入哈希</a> · <a href="validation.json">验证记录</a></p>
<script>function show(){let s=scene.value,k=variant.value;result.src=s+'_'+k+'.png';teacher.src=s+'_teacher.png';document.getElementById('name').textContent=variant.options[variant.selectedIndex].text}function pick(k){variant.value=k;show()}show()</script></html>'''
    (out/'preview.html').write_text(page,encoding='utf-8')


def main():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=False)
    for n in ['painting_parameters.py',Path(__file__).name]:ast.parse(Path(__file__).with_name(n).read_text(encoding='utf-8'))
    # The quantizer must be monotone and continuous, rather than a hard palette jump.
    x=np.linspace(-100,100,100001,dtype=np.float32)
    checks={}
    for name,cfg in PRESETS.items():
        if cfg['kind']=='poster':
            y=soft_levels(x,cfg['l_step'],cfg['q_sharp'],cfg['q_mix'])
            assert np.diff(y).min()>=-1e-4 and np.isfinite(y).all()
            checks[name]=dict(max_adjacent_step=float(np.diff(y).max()),monotone=True)
    data=json.loads((OLD/'data.json').read_text());adapter=np.load(OLD/'color_adapter.npz')['lab_chroma_affine']
    assets=[];timings=[]
    for row in data['validation']:
        key=row['id'];pred=load(OLD/(key+'_before.png'))
        for name,src in [('student',OLD/(key+'_before.png')),('source',Path(row['source'])),
                         ('teacher',Path(row['target'])),('previous',OLD/(key+'_g4_color.png'))]:
            shutil.copy2(src,OUT/(key+'_'+name+'.png'))
        assets.append(dict(id=key,actual_student_prediction=str(OLD/(key+'_before.png')),
                           input_sha256=sha(OLD/(key+'_before.png')),target_comparison_only_sha256=sha(row['target'])))
        for preset in PRESETS:
            start=time.perf_counter();rgb,stages=transform(pred,adapter,preset,True)
            assert rgb.shape==(720,1280,3) and rgb.dtype==np.uint8
            Image.fromarray(rgb).save(OUT/(key+'_'+preset+'.png'))
            if key=='val_stairs_090' and preset=='poster':
                for k,a in stages.items():Image.fromarray(a).save(OUT/('stage_'+k+'.png'))
            timings.append(dict(id=key,preset=preset,seconds=time.perf_counter()-start,
                                mean_change_255=float(np.abs(rgb.astype(float)-pred).mean())))
        print('PAINTING_PARAMETERS',key,flush=True)
    contacts(OUT);page(OUT);shutil.copy2(OLD/'color_adapter.npz',OUT/'color_adapter.npz')
    (OUT/'source_code').mkdir()
    for name in ['painting_parameters.py',Path(__file__).name]:shutil.copy2(Path(__file__).with_name(name),OUT/'source_code'/name)
    config=dict(presets=PRESETS,tone=TONE,inputs=assets,adapter_sha256=sha(OLD/'color_adapter.npz'),
        neural_weight_source=str(OLD/'student_RGB_fp32.pt'),neural_weight_sha256=sha(OLD/'student_RGB_fp32.pt'),
        new_training=False,inference_uses_teacher=False,model_integration=False,video_tested=False,board_validated=False,
        opencv=cv2.__version__,code={name:sha(Path(__file__).with_name(name)) for name in ['painting_parameters.py',Path(__file__).name]})
    (OUT/'experiment.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(dict(operator_checks=checks,records=timings),ensure_ascii=False,indent=2),encoding='utf-8')
    print('PAINTING_PARAMETERS_DONE',OUT,flush=True)


if __name__=='__main__':main()
