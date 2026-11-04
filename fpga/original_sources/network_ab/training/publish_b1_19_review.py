"""Compare the newest nineteen-level package with the preceding thirteen-level one."""
import json
import os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .audit_b1_sampling_v2 import OUT
from .runtime import file_hash,write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    previous=OUT/'temporal-margin';current=OUT/'nineteen-levels';package=current/'qat/package';videos=current/'videos'
    old=read(previous/'videos/results.json');new=read(videos/'results.json')
    assert {r['id'] for r in new['results']}=={'stairs','plaza'}
    boards=[Image.open(previous/'package/v1_v2_teacher.png'),Image.open(package/'v1_v2_teacher.png')]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for name,indices in [('levels13_vs19.png',(0,1,2,3)),('trees_people_13_vs19.png',(2,3))]:
        img=Image.new('RGB',(840,312*len(indices)),'#17202a');draw=ImageDraw.Draw(img)
        for row,index in enumerate(indices):
            for col,board in enumerate(boards):
                img.paste(board.crop((840,index*312+32,1260,index*312+312)),(col*420,row*312+32))
                draw.text((col*420+4,row*312+4),('上一轮：13层','本轮：19层')[col],font=font,fill='white')
        img.save(package/name)
    rel=lambda p:Path(os.path.relpath(p,videos)).as_posix();sections=[];stats=[]
    for r in new['results']:
        key=r['id'];p=next(v for v in old['results'] if v['id']==key)
        assert p['source_sha256']==r['source_sha256'] and p['video']['frames']==r['video']['frames']
        values={}
        for label,record in [('13_levels',p),('19_levels',r)]:
            values[label]={k:sum(v['output'][k] for v in record['motion'])/len(record['motion']) for k in ('mean','p95','over16')}
        stats.append(dict(scene=key,motion=values))
        a=rel(previous/'videos'/f'{key}_integer1080.mp4');b=f'{key}_integer1080.mp4'
        sections.append(f'<article><h2>{"草丛与台阶" if key=="stairs" else "树冠与人物"} · {r["video"]["seconds"]:g} 秒</h2><button onclick="sync(this)">同步播放 / 暂停</button><div class="pair"><section><h3>13 层</h3><video muted controls preload="metadata" src="{a}"></video></section><section><h3>19 层</h3><video muted controls preload="metadata" src="{b}"></video></section></div></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>13层与19层实际小网络对比</title><style>body{margin:24px;background:#111b25;color:#edf3f7;font:17px system-ui;line-height:1.6}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%}section{min-width:0}h1{font-size:26px}h2{font-size:22px}button{background:#234459;color:white;border:1px solid #5b8199;border-radius:5px;padding:8px 18px}article{margin:28px 0}img{max-width:100%}.note{max-width:1080px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>13 层 → 19 层：真实小网络输出</h1><p class="note">两列使用相同源片、相同1080→720→1080链路。增加的是模型的明暗层级，色度仍独立预测，并非全图只有19种RGB颜色。两列都来自真实20 KiB权重包的软件整数运行；没有输出后滤镜。人物和细小轮廓仍需改进，当前不宣称最终画质通过或已上板。</p>'''+''.join(sections)+f'<h2>原尺寸局部</h2><img src="{rel(package/"levels13_vs19.png")}">'+'''<script>function sync(button){const vs=[...button.closest('article').querySelectorAll('video')];if(vs.some(v=>!v.paused)){vs.forEach(v=>v.pause());return}const t=vs[1].currentTime;vs.forEach(v=>{v.currentTime=t;v.play().catch(()=>{});});}</script></html>'''
    (videos/'compare_13_19.html').write_text(page,encoding='utf-8');write_json(videos/'comparison_13_19.json',stats)
    verify=read(package/'verification.json');proof=read(package/'colour_head_proof.json')
    assert verify['learned_steps']==13700 and proof['maximum_output_error']<=1
    write_json(current/'delivery_manifest.json',dict(status='RESEARCH_CANDIDATE_CONTOURS_NOT_FULLY_RESOLVED',
        baseline_fallback=str(OUT.parent/'b1-style-v1-fallback-20261005'),previous_13_level=str(previous),current_19_level=str(package),
        graph_changed=False,transport_bytes=20480,postfilter=False,actual_board=False,visual_accepted=False,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in [package/'coefficients-padded.bin',package/'load_words.hex',
        package/'verification.json',package/'colour_head_proof.json',current/'qat/checkpoint_001000.pt',
        videos/'stairs_integer1080.mp4',videos/'plaza_integer1080.mp4',package/'levels13_vs19.png']]))
    lines=[f"- {r['scene']}：平均运动对齐误差 {r['motion']['13_levels']['mean']:.3f} → {r['motion']['19_levels']['mean']:.3f}；超过16级变化 {100*r['motion']['13_levels']['over16']:.3f}% → {100*r['motion']['19_levels']['over16']:.3f}%。" for r in stats]
    report=f'''# 19层实际B1候选

用户最新要求色层继续增加，因此从13层改为19个明暗代表值；色度仍独立学习。每个亮度特征从三态变为四态，仍然是6个亮度特征加2个色度特征，没有增加节点、算子或包大小。

本阶段1000 FP32 + 1000 QAT完成，加父模型共13700步（7600 FP32 + 6100 QAT）。两阶段均实际通过720p GPU前向/反向、小批次及新实例磁盘恢复；导出全节点复演和独立整数oracle误差0，坏包拒绝/CRC/hex回读通过。

RGB头需要三分之一系数，使用Q17乘数17476（近似2/15）。80275组合法有序亮度状态和色度组合的最大输出误差为{proof['maximum_output_error']}级；全部状态的连续算术误差上界{proof['all_state_continuous_error_bound']:.6f}级。这与13层的m65536不同，已单独验证，不能混用校准。

权重SHA256：`{verify['sha256']}`。有效17792B，传输20480B，小端64位装载2560字。软件推理链仍为1080→整数area720→B1 INT8/Q17→整数bilinear1080；没有外部后处理，未修改硬件或上板。

## 可查看材料

- `videos/compare_13_19.html`：同源13层与19层并排视频，台阶12秒、人物15秒，均1920×1080/30fps。
- `qat/package/levels13_vs19.png`：四处原尺寸局部对比。
- `qat/package/coefficients-padded.bin`、`load_words.hex`：实际推理权重。

## 时序诊断

'''+ '\n'.join(lines)+'''

这些数值不是主观“无抽动”验收。V1仍保留为用户指定兜底，不能仅凭新增色层或单一误差下降替换它。人物和细小轮廓仍然偏软，本轮未标记完整画质通过。

## 已有采样诊断的适用范围

同一末端360→720解码器拿到正确分区时比实际模型清楚，说明实际分区预测误差是问题的一部分；该旁路不证明完整180p瓶颈无信息损失。细线测试显示2–4像素结构会随采样相位丢对比度，因此不能把问题简单归到视频码率。

前一格点版解码中段饱和从约43%降至14%–19%，但最终灰阶阶跃宽度并未明显变窄；这些是前阶段测量，不冒称为19层最新测量。原始证据保留在上级目录的sampling_audit.json、reconstruction_audit.json、lattice-recovery和temporal-margin/sampling-check。

外部资料用于理解采样：[作者项目](https://richzhang.github.io/antialiased-cnns/)、[官方代码](https://github.com/adobe/antialiased-cnns)、[PyTorch像素格点约定](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html)。实际图与数值仍以本项目合同和整数复演为准。
'''
    (current/'engineering_review.md').write_text(report,encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
