"""Readable evidence for measured channel damage and compression tradeoffs."""
import argparse
from html import escape
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .importance_pruned_teacher import RUN,WIDTHS
from .runtime import configure_workspace,write_json,file_hash

LABELS=dict(b0='720p入口＋高分辨率跳连',b1='360p第一编码块',b2='360p编码＋跳连',b3='180p第一编码块',
            b4='180p编码＋跳连',b5='90p第一编码块',b6='90p编码＋注意力残差',
            attn_projection='注意力投影',attn_memory='注意力记忆通道',d1='180p解码细化',d3='360p解码细化',d5='720p输出细化')


def importance(run):
    data=json.loads((run/'importance.json').read_text(encoding='utf-8'));assert data['status']=='COMPLETE'
    assert len(data['records'])==sum(WIDTHS.values())
    ordered=[]
    for g,c in WIDTHS.items():
        a=sorted([r for r in data['records'] if r['group']==g],key=lambda r:r['score']);assert len(a)==c
        ordered.append(dict(group=g,label=LABELS[g],channels=c,minimum=a[0]['score'],maximum=a[-1]['score'],
                            ratio=a[-1]['score']/max(a[0]['score'],1e-12),lowest=[r['channel'] for r in a[:4]],highest=[r['channel'] for r in a[-4:]],mac_saved=a[0]['mac_saved']))
    write_json(run/'importance_summary.json',dict(groups=ordered,measurement_seconds=data['seconds'],
        description='Relative damage from deleting one physical dependency channel in the parent; no semantic label is inferred for any channel.'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    panel=Image.new('RGB',(1200,100+len(ordered)*58),'#151b24');draw=ImageDraw.Draw(panel)
    draw.text((18,12),'实际删除一个通道后，对画面造成的影响',font=font,fill='white')
    draw.text((18,45),'蓝＝本组最小损伤；橙＝最大损伤。7来源14张720p临近采样帧。数值不是视觉质量百分比。',font=small,fill='#c3d5e9')
    maximum=max(r['maximum'] for r in ordered)
    for i,r in enumerate(ordered):
        y=90+i*58;draw.text((18,y),r['label'],font=font,fill='white')
        x=345;scale=610/maximum
        draw.rectangle((x,y+1,x+r['maximum']*scale,y+16),fill='#e2a160')
        draw.rectangle((x,y+22,x+r['minimum']*scale,y+37),fill='#65b7da')
        draw.text((975,y),f"{r['minimum']:.4f}–{r['maximum']:.4f}",font=small,fill='white')
        draw.text((975,y+24),f"相差{r['ratio']:.1f}倍",font=small,fill='#c3d5e9')
    panel.save(run/'importance.png')
    body=''.join(f"<tr><td>{r['group']}</td><td>{r['label']}</td><td>{r['channels']}</td><td>{r['minimum']:.5f}</td><td>{r['maximum']:.5f}</td><td>{r['ratio']:.1f}</td><td>{r['mac_saved']/1e6:.2f}M</td><td>{r['lowest']}</td></tr>" for r in ordered)
    details=''.join(f"<details><summary>{LABELS[g]}：查看所有通道</summary><table><tr><th>通道编号</th><th>损伤分</th><th>最差来源损伤</th></tr>"+''.join(f"<tr><td>{r['channel']}</td><td>{r['score']:.6f}</td><td>{r['worst']:.6f}</td></tr>" for r in sorted([r for r in data['records'] if r['group']==g],key=lambda r:r['score']))+'</table></details>' for g in WIDTHS)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实际测得的通道重要性</title><style>body{max-width:1200px;margin:25px auto;padding:0 18px;background:#151b24;color:#edf3fa;font:17px/1.8 system-ui}a{color:#acd4ff}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:8px;border:1px solid #42536a;text-align:left}details{margin:16px 0}summary{cursor:pointer}.scroll{overflow:auto}</style><h1>先实际删除，再测重要性</h1><p>12组连接关系，共528个可删通道。每一项都实际建成少一个通道的网络，在7个训练来源的14张完整720p相邻帧上运行。只在残差相加和LADE配对位置绑定通道；不同层不再仅因宽度相同而一起剪。</p><p>损伤分综合与父模型的RGB、边缘、低频、局部最差块和时间变化。数值越低只表示这组测量下影响较小，不表示这个通道永远没用。多通道一起删除仍需重新评估、恢复训练和看完整视频。</p><p><a href="index.html">返回最终视频</a> · <a href="RESULTS.md">结果与限制</a> · <a href="importance.json">逐来源完整数据</a></p><a href="importance.png"><img src="importance.png" alt="各连接组最小与最大删除损伤"></a><div class="scroll"><table><tr><th>组</th><th>位置</th><th>通道数</th><th>最小损伤</th><th>最大损伤</th><th>相差倍数</th><th>删一通道省MAC</th><th>本组较低损伤编号</th></tr>BODY</table></div>DETAILS<p>卷积MAC是计算工作量，不是FPGA资源、DDR流量或实测帧率。未给通道贴“人脸/草地”等未经验证的语义标签。</p></html>'''
    html=html.replace('12组连接关系，共528个可删通道。','12组连接关系，共528个可删通道。编号为本轮半通道父模型的组内索引，不是原始教师的通道编号。').replace('相邻帧','临近采样帧')
    (run/'importance_summary.html').write_text(html.replace('BODY',body).replace('DETAILS',details),encoding='utf-8')
    print('IMPORTANCE_REPORT_READY',flush=True)


def result(root,run,candidate):
    importance(run)
    v=json.loads((run/(candidate+'_recovery')/'verification.json').read_text())
    p=json.loads((run/(candidate+'_recovery')/'preflight/result.json').read_text())
    videos=json.loads((run/('videos_'+candidate)/'report.json').read_text());assert videos['status']=='COMPLETE'
    diag=json.loads((run/('evaluation_'+candidate)/'video_diagnostics.json').read_text());assert diag['status']=='COMPLETE'
    chosen=json.loads((run/candidate/'plan.json').read_text());summary=json.loads((run/'importance_summary.json').read_text())
    lines=['# 按实际重要性继续压缩教师网络','',
           '本轮采用真实连接依赖分组、逐通道物理删除测量、按损伤与节省计算量选择，以及训练特征统计补偿。原父模型和原教师保留。动画属于压缩/效果诊断，不改变本项目游戏写实、左手交互、恶灵叠加的基线范围。','',
           '## 具体做了什么','',
           '上一轮把所有相同宽度层绑定到同一套通道索引；本轮拆成12组，只保留残差与LADE必要的对齐。对528个通道逐个实际删除，检查7个训练来源的14张完整720p相邻帧。先保住已有风格，再比较省下的实际卷积计算量。',
           '', '参考了[Molchanov CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)和[作者代码](https://github.com/NVlabs/Taylor_pruning)对损失敏感性的讨论；本轮运行的是直接物理删除测量，并未安装或宣称运行该仓库的Taylor剪枝算法。','',
           '[可展开每个通道的重要性表](importance_summary.html)','', '![删除损伤实测](importance.png)','',
           '分数为RGB + 2×边缘 + 0.5×低频 + 0.5×局部最差10%块 + 0.2×光流时间变化，再取七来源均值+0.5×最差来源。只衡量对当前父模型的影响，不是风格百分比，也不把通道解释成人脸或草地专用单元。','',
           f"全通道测量用时{summary['measurement_seconds']:.1f}秒；测量与统计校准没有使用留出视频。组合删除存在相互影响，后续仍做实际恢复，不能把单通道损伤简单相加当质量保证。",'',
           '## 本轮候选与实际训练','',
           f"选择`{candidate}`：{v['profile']['parameters']:,}参数，{v['profile']['conv_macs']/1e9:.4f}G数学卷积MAC/720p帧。父模型263,264参数、28.1641G；本轮相对父模型减少{(1-v['profile']['conv_macs']/28164096000)*100:.2f}%卷积工作量。",'',
           '|连接组|父模型通道|本轮保留|','|---|---:|---:|']
    for g,c in WIDTHS.items():lines.append(f'|{LABELS[g]}|{c}|{len(chosen["plan"][g])}|')
    lines += ['', '这轮按预算生成三档初始权重：17.4864G / 168194参数、13.9217G / 123394参数、9.3916G / 84760参数。9.4G初始在石柱、台阶与人物处明显变软，13.9G初始保留轮廓但色偏，选13.9G进行恢复；其余两档未训练或制作视频。选择参考了已见回归静态图，因此也不能视为独立盲测。', '',
              f"相对半通道父模型，参数减少{(1-v['profile']['parameters']/263264)*100:.2f}%；相对原始教师（1043648参数、103.9859712G），参数减少{(1-v['profile']['parameters']/1043648)*100:.2f}%，卷积MAC减少{(1-v['profile']['conv_macs']/103985971200)*100:.2f}%。",'',
              '在7来源95帧对上进行600步完整720p FP32恢复，每步相邻两帧。先通过GPU完整720前后向、batch2的360p小批次优化和磁盘恢复重放，再正式训练；300步结束进程，另进程恢复至600。', '',
              f"优化循环用时{v['optimizer_seconds']:.2f}秒，峰值{v['peak_mib']:.2f}MiB；预检恢复误差{p['restore_error']:.3g}，最终独立重载六图像素差{v['reload_image_error']}。CUDA grid_sample反向非严格确定性警告保留。",'',
              '监督目标仍为原教师加既有表面修正；本轮网络推理不添加该修正、锐化、原片混合、贴脸、EMA或时间混帧。这里600步不同于父模型300步，不把效果差异全归因于某一项剪枝规则。','',
              '## 图片、视频与真实权重','', '[完整视频同帧切换](index.html)','', '![原教师、上一轮、本轮](comparison.png)','',
              f'[真实FP32权重]({candidate}_recovery/pruned_teacher_fp32.pt)，SHA-256 `{v["weights_sha256"]}`。','']
    for r in videos['records']:lines.append(f"- [{r['id']} / {r['validation']['frames']/30:g}秒](videos_{candidate}/{r['id']}_pruned_1080p.mp4)：{r['model_calls']}次真实网络调用，相邻完全重复输出{r['identical_adjacent_outputs']}。")
    lines += ['', '三段保持1080整数面积下采样→720完整网络→1080整数双线性，30fps，H.264 CRF16。共1170帧逐帧实际处理和完整解码。回归素材来源隔离，但以前已看过，不能当新盲测。','',
              '|场景|本轮时间残差|上一轮半通道|原教师|本轮新增白像素比例|','|---|---:|---:|---:|---:|']
    for r in diag['records']:
        a=r['variants'];lines.append(f"|{r['id']}|{a['new']['temporal_mean']:.4f}|{a['previous']['temporal_mean']:.4f}|{a['teacher']['temporal_mean']:.4f}|{a['new']['newly_white_fraction']:.6%}|")
    lines += ['', '|场景|本轮在教师纹理变化方向上的投影|上一轮投影|本轮相关性|上一轮相关性|','|---|---:|---:|---:|---:|']
    for r in diag['records']:
        a=r['variants'];lines.append(f"|{r['id']}|{a['new']['material_projection']:.4f}|{a['previous']['material_projection']:.4f}|{a['new']['material_cosine']:.4f}|{a['previous']['material_cosine']:.4f}|")
    lines += ['', '纹理变化以灰度高频分量相对原片的变化计算，投影是对教师变化方向的线性投影系数，相关性是余弦相似度；都不是材质质量或风格保留百分比。它们与时间残差同时列出，避免把较弱的风格误算成稳定性进步。','']
    lines += ['', '全部1167相邻帧对使用相同源图光流掩膜测量。弱风格也可降低时间残差；细植物与遮挡部分被掩膜排除。新增白像素阈值不等于通用曝光检测。以上均不是视觉验收或不抽动的保证。','',
              '## 仍需跨过的板端差距','',
              f"本轮卷积量仍是B1（0.9667584G）的{v['profile']['conv_macs']/966758400:.2f}倍；保留动态LADE全局归一化、sqrt/div、注意力softmax、7×7和tanh。未改RTL、未做INT8定标/整数对拍、资源fit或P&R/STA。共享阵列资源不会简单按参数量同比缩小。",'',
              '新图、权重和算子定义必须配套；不能直接作为现有B1系数文件。下一步是否继续削块或替换算子，要由实际损伤和资源/时序约束共同决定。','',
              '复现使用已有env-probes/animegan环境，设置PYTHONPATH=E:\\fpga。构图入口`importance_pruned_teacher.build_group_pruned(root, state["indices"])`，加载`state["generator"]`。恢复代码与数据/源码SHA、检查点、采样器/优化器状态均随recovery目录保存。','']
    note='训练与排名使用约0.1秒间隔的临近采样帧对，不是每对都取30fps紧挨的下一帧；具体PTS和索引间隔见training_assets_audit.json。交付视频则处理全部连续帧，并对所有相邻帧对测量。'
    lines += [note,'','[权重、匹配网络及现有环境中的加载方法](INFERENCE.md)。实际31卷积算子图和接口清单已随recovery目录保存。','']
    if (run/'near_faces.png').exists():lines += ['![四版近处人物对照](near_faces.png)','']
    if (run/'visual_review.json').exists():
        review=json.loads((run/'visual_review.json').read_text(encoding='utf-8'))
        lines += ['## 实际画面复核','']+review['observations']+['']
    if (run/candidate/'cost_breakdown.json').exists():
        breakdown=json.loads((run/candidate/'cost_breakdown.json').read_text())
        assert breakdown['conv_macs']==v['profile']['conv_macs'] and breakdown['parameters']==v['profile']['parameters']
        lines += ['## 剩余计算量集中在哪里','',f'[全部31个卷积的计算明细]({candidate}/cost_breakdown.json)。首尾两个7×7卷积合计约29.2%计算量；下一轮不宜仅按参数数目判断是否值得改。','']
    stages=dict(parent=root/'runs/teacher-pruning-calibrated-20261003/keep50_recovery/validation_000300/scores.json',
                initial=run/(candidate+'_recovery')/'validation_000000/scores.json',
                final=run/(candidate+'_recovery')/'validation_000600/scores.json')
    quality={}
    for stage,path in stages.items():
        records=json.loads(path.read_text())['records']
        quality[stage]={k:sum(r['teacher'][k] for r in records)/len(records) for k in ['mae_rgb255','gradient_mae']}
    write_json(run/'validation_summary.json',dict(stages=quality,n=6,scope='Already seen regression images; pixel/gradient error versus original teacher target, not a style acceptance percentage.'))
    lines += ['## 六张已见回归图误差','', '|阶段|相对原教师RGB平均绝对误差 /255|梯度误差|','|---|---:|---:|']
    for stage,label in [('parent','上一轮半通道'),('initial','本轮刚剪完'),('final','本轮恢复600步')]:
        lines.append(f"|{label}|{quality[stage]['mae_rgb255']:.4f}|{quality[stage]['gradient_mae']:.4f}|")
    lines += ['', '误差仅作为数值回归，不能代替材质、人脸与视频稳定性的视觉判断。','']
    (run/'RESULTS.md').write_text('\n'.join(lines).replace('14张完整720p相邻帧','14张完整720p临近采样帧').replace('每步相邻两帧','每步临近采样两帧'),encoding='utf-8')
    names=['RESULTS.md','INFERENCE.md','index.html','comparison.png','importance_summary.html','importance_summary.json','importance.png','importance.json','media_checks.json','browser_checks.json','visual_review.json','training_assets_audit.json','validation_summary.json','near_faces.png',
           candidate+'/cost_breakdown.json',candidate+'_recovery/pruned_teacher_fp32.pt',candidate+'_recovery/verification.json',candidate+'_recovery/inference_graph.txt',candidate+'_recovery/operator_inventory.json','evaluation_'+candidate+'/video_diagnostics.json','videos_'+candidate+'/report.json']
    write_json(run/'delivery_manifest.json',dict(candidate=candidate,weight_sha256=v['weights_sha256'],
        files={p:file_hash(run/p) for p in names},videos={r['id']:r['validation']['sha256'] for r in videos['records']},
        publication_code={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in ['report_pruning_importance.py','publish_importance_teacher.py','check_importance_media.py','pruning_cost_breakdown.py','audit_importance_assets.py']},production=False))
    print('RESULT_REPORT_READY',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['importance','result']);ap.add_argument('--candidate',default='cost14g');args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN
    if args.mode=='importance':importance(run)
    else:result(root,run,args.candidate)


if __name__=='__main__':main()
