"""Local review page and reproducible final report for the bounded trial."""
import json
from pathlib import Path
import shutil
from .audit_style_branches import RUN,REPAIR_SHA
from .runtime import configure_workspace,write_json,file_hash


def main():
    root=configure_workspace();run=root/'runs'/RUN
    evaluation=json.loads((run/'final_evaluation/measurements.json').read_text());video=json.loads((run/'video_diagnostics.json').read_text());assert video['status']=='COMPLETE'
    export=json.loads((run/'recovery/export.json').read_text());p=evaluation['profile'];base=evaluation['baseline']['train'];new=evaluation['values']['train']
    status='通过本轮诊断筛查，仍需用户视觉评价' if evaluation['comparison']['screen_pass'] else '仍有诊断项目未通过，保留为待评估候选'
    notes=f'''<p>按用户要求先查功能，再删减。教师与修正版各做17条路径的两档干预；另做40次真实解码通道删除。这40个通道均未同时通过逐场景筛查，因此没有强删。</p>
<p>本次候选实际移除注意力残差的4个卷积与softmax，保留多尺度跳接和LADE统计路径；再做240步FP32修正，训练中压制教师平整区域里的额外碎细节。推理没有加入模糊、锐化、贴脸、混原片或时间混帧。</p>
<p><b>{status}</b>。参数123,394 → {p['parameters']:,}（减少7.13%）；720卷积计算13.9217 → {p['conv_macs']/1e9:.4f}G（减少0.91%）。删掉了一个复杂算子分支，但尚不是大幅压缩。仍不能直接替换B1系数，也未证明FPGA资源与时序达标。</p>
<p>视频：走廊12秒、台阶12秒、广场15秒，均30fps逐帧网络推理。分辨率链路1080→720→1080。可在同一时刻切换原片、教师、上轮修正版和本轮候选。</p>'''
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>按功能删减：连续视频对比</title><style>body{max-width:1440px;margin:24px auto;padding:0 18px;background:#151b24;color:#edf3fa;font:17px/1.7 system-ui}a{color:#aed7ff}select,button{font:inherit;padding:7px;margin:5px;background:#273648;color:white;border:1px solid #748ba4;border-radius:5px}video{width:100%;max-height:78vh;background:#000}img{max-width:100%}.controls{position:sticky;top:0;background:#151b24;padding:8px;z-index:2}table{border-collapse:collapse}th,td{border:1px solid #52657d;padding:8px}</style><h1>先保住风格、结构和稳定，再删除</h1>NOTES
<div class="controls"><label>场景<select id="scene"><option value="stairs">台阶 / 12秒</option><option value="new_hallway">走廊 / 12秒</option><option value="plaza">广场人物 / 15秒</option></select></label><label>版本<select id="version"><option value="new">本轮按功能删减＋修正</option><option value="previous">上轮修正目标续训版</option><option value="teacher">教师目标</option><option value="source">原片</option></select></label><button id="play">播放 / 暂停</button><button id="seek">定位3秒</button><span id="loading"></span></div>
<video controls playsinline preload="metadata"></video><p id="error"></p>
<p>图片四列：原片／教师／上轮修正版／本轮。720原生像素裁切，点击看原尺寸。</p><a href="final_comparison.png"><img src="final_comparison.png" alt="原片教师修正版和功能剪枝版对比"></a>
<p><a href="hallway_exception.png">单独查看未过细节筛查的走廊7秒帧：教师／上轮／本轮</a>。该帧已保留，没有从结果中排除。</p>
<p>广场11.97秒近处人物：原片／教师／上轮／本轮。统一固定区域，没有局部修脸。</p><a href="near_faces.png"><img src="near_faces.png" alt="近处人物同帧对照"></a>
<p><a href="index.html">分支功能实验与未采用的破坏性对照</a> · <a href="RESULTS.md">结果和限制</a> · <a href="recovery/functional_pruned_fp32.pt">真实FP32候选权重</a> · <a href="INFERENCE.md">加载方法</a></p>
<h2>连续帧诊断</h2><p>逐个检查全部1167对相邻帧。残差越低不等于风格越好；教师变化投影也不是风格百分比。光流可能排除遮挡和细小植物，最终以完整视频观感为准。</p>TABLE
<script>
const v=document.querySelector('video'),scene=document.querySelector('#scene'),version=document.querySelector('#version'),seek=document.querySelector('#seek'),loading=document.querySelector('#loading');let pendingTime=0,busy=false,wantPlay=false,serial=0;
function load(keep){const n=++serial;if(!busy){pendingTime=keep?v.currentTime:0;wantPlay=!v.paused}else if(!keep)pendingTime=0;v.pause();busy=true;seek.disabled=true;loading.textContent='读取视频…';const id=scene.value,k=version.value;const url=k==='new'?'videos/'+id+'_1080p.mp4':k==='previous'?'reference/videos/'+id+'_1080p.mp4':k==='teacher'?'../b1-distillation-20261003/videos-2000/'+id+'_teacher_1080p.mp4':'../clear-style-control-20261003/video_sources/'+id+'_source_1080p.mp4';v.onloadedmetadata=()=>{if(n!==serial)return;v.currentTime=Math.min(pendingTime,v.duration-.05);busy=false;seek.disabled=false;loading.textContent='';if(wantPlay)v.play().catch(()=>{})};v.onerror=()=>{busy=false;loading.textContent='';document.querySelector('#error').textContent='视频读取失败，请使用本地文件链接。'};document.querySelector('#error').textContent='';v.src=url;v.load()}
scene.onchange=()=>load(false);version.onchange=()=>load(true);document.querySelector('#play').onclick=()=>{if(!busy){if(v.paused){wantPlay=true;v.play().catch(()=>{})}else{wantPlay=false;v.pause()}}};seek.onclick=()=>{if(!busy){v.pause();wantPlay=false;v.currentTime=3}};load(false);
</script></html>'''
    table='<table><tr><th>场景</th><th>版本</th><th>效果时间残差</th><th>教师纹理变化投影</th><th>新增白点比例</th></tr>'
    for r in video['records']:
        for k,label in [('previous','上轮修正版'),('new','本轮'),('teacher','教师')]:
            v=r['variants'][k];table+=f"<tr><td>{r['id']}</td><td>{label}</td><td>{v['temporal_mean']:.4f}</td><td>{v['material_projection']:.4f}</td><td>{v['newly_white_fraction']:.6f}</td></tr>"
    table+='</table>';(run/'review.html').write_text(html.replace('NOTES',notes).replace('TABLE',table),encoding='utf-8')
    lines=['# 按功能剪枝：本轮证据与结果','',f'当前状态：{status}。用户最新意见是修正版已接近教师，主要问题为不必要的细节；不沿用此前整版失败的概括。','',
      '## 功能实验','',
      '- 2个模型×(17条路径×2档干预+1基准)×56帧=3920次诊断前向。包含7训练来源14张图、6张已见回归图、3个12帧连续窗口33个相邻帧对。所有门在1时四图像素误差0。',
      '- 180行跳接对多尺度风格变化和结构均有贡献；720行跳接不仅带细节，也承担颜色和轮廓，不能把它当成多余原片细节整条删除。首末统计路径旁路后细节和结构同时恶化，并放大教师参照时间误差。',
      '- 注意力残差在本批场景的教师和修正版里均低敏感；并不证明对任意场景都冗余。初始物理删除仍在一张广场图超过细节筛查线，证据保留，未放宽原阈值。',
      '- 三个解码专属依赖组的40个通道逐个实际删除，全部未过逐场景三轴筛查，没有强删；这不能证明经过重新训练永远不可删除。',
      '- 稳定性并非一条独立分支提供。推理无循环或光流状态，功能耦合；低时间残差必须结合风格和结构判断。',
      '- 旧静态目标为截断，本次教师诊断为四舍五入；已严格FP32重现旧20目标像素误差0。所有干预相对自身同量化基准；量化差异内不作效果解释。连续窗量化一致。新训练沿用原始目标PNG。','',
      '## 真实候选与修正','',
      f'- 物理删除4个注意力卷积及softmax等计算；27卷积、{p["parameters"]}参数、{p["conv_macs"]} MAC/720。六图与关分支诊断像素差0。',
      '- 240步严格FP32；120步进程退出后从磁盘恢复。完整720双帧更新、小批次2×360、磁盘重放通过；预检重放最大误差1.49e-8，CUDA grid_sample反向不保证逐bit确定。最终六图独立重载像素差0。',
      '- 保留原教师结构/纹理变化/时间变化目标，并在教师平整区域惩罚额外细节；此为训练项，推理没有增加模糊、材质分类或后处理。',
      f'- 同一筛查标准复查：{json.dumps(evaluation["comparison"],ensure_ascii=False)}',
      f'- 真实候选权重：recovery/functional_pruned_fp32.pt，SHA {export["sha256"]}。上轮权重SHA {REPAIR_SHA}，原文件保留。','',
      '## 视频与限制','',
      '- 本轮和上轮修正版各完整1170帧（12/12/15秒30fps），1080整数面积→720网络→1080整数双线性；CRF16编码，并保存编码前PNG。没有重复补帧、时间混合、贴脸或锐化。',
      '- video_diagnostics.json覆盖全部1167相邻帧对，见review.html。未做独立盲测；原回归场景在决策中使用过。全部统计只是诊断，不代替用户对风格、人物、植物的观看。',
      '- 仍非B1算子图，未做INT8、FPGA fit或物理时序；只删减一个分支不代表已经得到上板权重。没有修改硬件RTL/合同/其他工作流。','']
    (run/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (run/'INFERENCE.md').write_text('''# 加载本轮候选

在项目隔离环境中使用，PYTHONPATH=E:\\fpga：

```python
from pathlib import Path
import torch
from network_ab.training.functional_pruned_model import build
root = Path("E:/fpga/model-training-workspace")
state = torch.load(root / "runs/branch-function-audit-20261004/recovery/functional_pruned_fp32.pt", map_location="cpu", weights_only=True)
net = build(root, state).cuda().eval()
# RGB uint8 source720 -> float32 NCHW /128 - 1
# Use train_b1_distillation.as_tensor and to_rgb; no output guard.
```

必须用本轮构图函数读取 removed_attention 元数据，不能用旧31卷积图硬加载。其余图构造依赖项目现有固定ONNX缓存和半通道父模型，代码绑定见recovery/binding.json。权重不是B1系数ABI。FPGA部署仍需另做算子映射、量化、资源和时序验证。
''',encoding='utf-8')
    source=run/'frozen_source';source.mkdir(exist_ok=True)
    files=list(json.loads((run/'recovery/binding.json').read_text())['code'])+[str(Path(__file__))]+[str(Path(__file__).with_name(n)) for n in ['publish_functional_pruning.py','summarize_branch_audit.py','diagnose_functional_clip.py','probe_branch_quantization.py','make_functional_face_review.py','publish_fullres_c8_stylepush.py','fpga_video_resize.py','render_temporal_native_detail.py','render_wavelet_game.py']]
    for path in files:shutil.copy2(path,source/Path(path).name)
    names=['review.html','RESULTS.md','INFERENCE.md','final_comparison.png','hallway_exception.png','near_faces.png','video_diagnostics.json','recovery/functional_pruned_fp32.pt','recovery/export.json','final_evaluation/measurements.json','recovery/training_verification.json']
    write_json(run/'delivery_manifest.json',dict(status=status,files={n:file_hash(run/n) for n in names},source={str(p):file_hash(p) for p in source.iterdir()},hardware_modified=False,production=False))
    print('REVIEW_READY',status,flush=True)


if __name__=='__main__':main()
