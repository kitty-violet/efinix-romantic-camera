"""Publish completed media and exact weight bindings for the clarity revision."""
import argparse
import json
from pathlib import Path
from . import train_b1_19_levels as n
from .runtime import file_hash,write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--videos',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);a=ap.parse_args()
    package,videos=a.package,a.videos;check=read(package/'verification.json');res=read(videos/'results.json')
    assert {r['id'] for r in res['results']}=={'stairs','plaza'}
    assert res['sha256']==check['sha256']==file_hash(package/'coefficients-padded.bin')
    before=read(n.OUT.parent/'temporal-margin/videos/results.json');sections=[];metrics=[]
    for r in res['results']:
        key=r['id'];prev=next(v for v in before['results'] if v['id']==key)
        assert prev['source_sha256']==r['source_sha256'] and prev['video']['frames']==r['video']['frames']
        values={}
        for name,row in [('earlier_13_level',prev),('new_19_level',r)]:
            values[name]={k:sum(m['output'][k] for m in row['motion'])/len(row['motion']) for k in ('mean','p95','over16')}
        metrics.append(dict(scene=key,motion=values))
        sections.append(f'<h2>{"台阶与草丛" if key=="stairs" else "树冠与人物"} · {r["video"]["seconds"]:g}秒</h2><video controls preload="metadata" src="{key}_integer1080.mp4"></video>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19层清晰度与偏色修正版</title><style>body{max-width:1320px;margin:24px auto;padding:0 16px;background:#15202b;color:#eef3f8;font:17px system-ui;line-height:1.65}video{width:100%;max-height:78vh}img{width:100%}h1{font-size:27px}h2{margin-top:32px}a{color:#9ddaff}</style><h1>保持19层：清晰度与偏色修正</h1><p>视频直接来自本版权重包，按1080→720→1080进行软件整数复演。没有外部锐化或色彩滤镜。仍需检查细小人物和植被，未标记最终画质或真实上板通过。</p>'''+''.join(sections)+'''</html>'''
    (videos/'watch.html').write_text(page,encoding='utf-8');write_json(videos/'comparison_motion.json',metrics)
    files=[package/'coefficients-padded.bin',package/'load_words.hex',package/'verification.json',a.checkpoint,
           package/'sharpness-review/before_after_teacher.png',videos/'stairs_integer1080.mp4',videos/'plaza_integer1080.mp4']
    write_json(videos.parent/'delivery_manifest.json',dict(status='NUMERICALLY_VERIFIED_VISUAL_REVIEW_REQUIRED',
        luminance_levels=19,graph_changed=False,postfilter=False,actual_board=False,visual_accepted=False,
        learned_steps=check['learned_steps'],fp32_steps=check['fp32_steps'],qat_steps=check['qat_steps'],
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in files],motion=metrics))
    comparison=read(package/'sharpness-review/comparison.json')
    colour_lines=[f"{r['id']}: {r['before']['green_excess_relative_to_teacher']:.3f} -> {r['after']['green_excess_relative_to_teacher']:.3f}" for r in comparison['records']]
    report='''19层清晰度与绿色偏色修正版

保持19层设定，新增800步FP32和1200步QAT。先通过真实720p GPU前向/反向、小批次、新实例磁盘断点恢复，再正式训练。累计15700步（8400 FP32+7300 QAT）。

改动均在训练与权重内：以减弱解码低通作为初值；监督真实RGB强边缘，并惩罚教师附近不存在的杂边；加强低饱和及低频色度误差。继续使用相邻帧光流、采样相位和阈值稳定约束。运行时不使用光流、教师、人物分割、额外锐化或色彩滤镜。

结果：去绿明显，清晰度仅小幅改善，细小人物和植被仍软。更激进的解码权重探针产生明显格点/碎点，未采用。不能声称所有轮廓已解决。

六张反复审阅的回归图（不是盲测），明亮低饱和区域相对教师的绿色偏差，单位8bit RGB级：
'''+ '\n'.join(colour_lines)+'''

自然图强边缘附近的对比不足有所下降，但该指标允许2像素位置容差，不证明轮廓形状正确或全面变窄。独立灰阶阶跃测试的最终RGB 10%-90%宽度平均2.827->3.151像素，反而变宽，因此不能把局部对比改善夸大成整体采样分辨率提升。

两段视频均完整解码检查：台阶12秒/360帧，人物15秒/450帧，1920×1080、30fps、CRF16。处理为1080整数area720→B1 INT8/Q17→整数bilinear1080。编码前PNG也保存，可分清模型模糊与编码损失。运动误差是诊断量，不能单独等同“无抽动”。

有效权重17792字节，传输20480字节；24卷积29节点，同一B1合同。导出全节点复演和独立整数oracle误差0，CRC坏包拒绝、截断/非零填充拒绝、小端64位hex回读通过。RGB固定解码头80275种有序亮度/色度组合误差0。

未修改硬件，未上板，没有RTL/P&R/时序新放行。用户指定V1兜底独立封存，没有覆盖。
'''
    (videos.parent/'engineering_review.txt').write_text(report,encoding='utf-8')
    print('CRISP_PUBLICATION_COMPLETE',json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
