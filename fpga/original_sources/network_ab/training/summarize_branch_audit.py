"""Readable evidence for causal branch tests; never labels a metric as acceptance."""
import json
from pathlib import Path
from html import escape
from PIL import Image,ImageDraw,ImageFont
from .audit_style_branches import RUN,paths
from .runtime import configure_workspace,write_json,file_hash


LABELS={**{f'base_lade{i}':f'编码{i}统计路径' for i in range(7)},
        **{f'main_lade{i}':f'解码{i}统计路径' for i in range(6)},
        'skip180':'180行跳接','skip360':'360行跳接','skip720':'720行跳接','attention':'注意力残差'}


def panel(run,model,variants,name):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    assets=json.loads((run/'assets.json').read_text())['rows'];lookup={r['id']:r for r in assets}
    specs=[('val_stairs_090',(424,320),'石柱与台阶'),('val_stairs_090',(180,430),'植物'),('val_plaza_090',(80,448),'人物')]
    columns=[('teacher','教师目标'),('baseline','当前修正版' if model=='repair' else '原教师')]+variants
    canvas=Image.new('RGB',(320*len(columns),922),'#151b24');draw=ImageDraw.Draw(canvas)
    for c,(_,label) in enumerate(columns):draw.text((c*320+6,8),label,fill='white',font=font)
    for j,(key,(x,y),label) in enumerate(specs):
        top=38+j*290;draw.text((6,top),label,fill='#b9d5ef',font=font)
        for c,(variant,_) in enumerate(columns):
            p=lookup[key]['target'] if variant=='teacher' else run/model/variant/(key+'.png')
            canvas.paste(Image.open(p).crop((x,y,x+320,y+256)),(320*c,top+27))
    draw.text((6,902),'同帧原生720像素裁切；路径干预仅诊断，未训练、未做输出混合。',fill='#b9d5ef',font=font)
    canvas.save(run/name)


def main():
    root=configure_workspace();run=root/'runs'/RUN;allmodels={};tables=[]
    for model in ('repair','teacher'):
        obj=json.loads((run/model/'summary.json').read_text());assert obj['status']=='COMPLETE'
        rows=obj['records'];baseline=rows[0];out=[]
        for r in rows[1:]:
            a,b=r['means']['train'],baseline['means']['train']
            d={k:a[k]-b[k] for k in a};dt={seq:{k:r['temporal'][seq][k]-baseline['temporal'][seq][k] for k in ('effect_residual','teacher_relative_residual')} for seq in r['temporal']}
            out.append(dict(name=r['name'],path=r['path'],alpha=r['alpha'],train_delta=d,temporal_delta=dt,
                regression_delta={k:r['means']['seen_regression'][k]-baseline['means']['seen_regression'][k] for k in a}))
        allmodels[model]=out
        body=''
        for r in out:
            d=r['train_delta'];t=r['temporal_delta'];avg=lambda k:sum(v[k] for v in t.values())/len(t)
            body+=f"<tr><td>{LABELS[r['path']]}</td><td>{r['alpha']}</td><td>{d['fine_error']:+.3f}</td><td>{d['excess_detail']:+.3f}</td><td>{d['structural_edge_error']:+.3f}</td><td>{d['change_projection']:+.3f}</td><td>{avg('effect_residual'):+.3f}</td><td>{avg('teacher_relative_residual'):+.3f}</td></tr>"
        tables.append(f"<details><summary>{'修正版' if model=='repair' else '原教师'}：展开完整逐路径变化数据</summary><table><tr><th>干预路径</th><th>保留系数</th><th>纹理误差</th><th>多余细节</th><th>主边缘误差</th><th>教师变化投影</th><th>效果时间残差</th><th>教师参照时间误差</th></tr>{body}</table></details>")
    write_json(run/'functional_deltas.json',allmodels)
    # Fixed anatomical examples, not selected to maximize apparent improvement.
    panel(run,'repair',[('skip720_0.75','720跳接保留75%'),('skip720_0','720跳接断开')],'skip720_comparison.png')
    panel(run,'repair',[('attention_0.75','注意力保留75%'),('attention_0','注意力残差断开')],'attention_comparison.png')
    panel(run,'repair',[('main_lade5_0.75','末层统计保留75%'),('main_lade5_0','末层统计旁路')],'last_statistics_comparison.png')
    notes='''<p>用户最新反馈：修正版已比较接近教师，剩下的多余细节影响动画感。本轮先调查功能，未把旧报告的整版失败判断沿用到新任务。</p>
<p>三条跳接和注意力残差测试对应加法分支的强度；13个LADE统计路径测试学习的统计与特征自身统计之间的过渡。0表示旁路该统计变换，不是把整层输出抹零。所有17个门在系数1时已验证与原模型像素完全一致。</p>
<p>14张训练图来自7来源；6张回归图已经见过；时间检查来自三个各12帧的连续30fps窗口，共33对相邻帧。它们只用于筛查功能，不能证明完整视频不闪动，也不是独立盲测。教师保留既有R48表面修正，学生无输出修正。</p>
<p>纹理误差、多余细节、主边缘误差越低越好；教师变化投影只描述方向相关性，不是风格百分比。时间残差变低可能只是风格弱了。每条轴单列，不能把糊画面或弱风格当成稳定性提升。光流遮挡区域可能被排除，人物视觉必须另外检查。</p>
<p>重要性是这批数据和这些权重上的局部干预证据；串联层、跳接、归一化彼此补偿，不能宣称某条通道只有一种功能。此处只是诊断门，没有声称已经实现物理剪枝或FPGA资源下降。</p>
<p>量化核查：历史静态教师目标使用截断，本轮教师诊断使用四舍五入，故静态基准约有0.5/255差异。逐条干预均相对本模型同量化基准；连续窗目标全程采用一致的四舍五入。已用严格FP32重现历史20张目标，像素误差0。小于量化差异的变化不作风格结论；新训练仍使用原始目标PNG。</p>'''
    conclusions='''<h2>本轮确认的功能边界</h2><table><tr><th>要保的效果</th><th>实际观察</th><th>删减决定</th></tr>
<tr><td>风格与材质变化</td><td>断开180行跳接，修正版教师变化投影0.641→0.374；教师0.999→0.492，且结构也受损。</td><td>保护多尺度融合，不能按小分辨率就先砍。</td></tr>
<tr><td>整洁、颜色与轮廓</td><td>720跳接减弱能少碎细节，但颜色、轮廓也被改坏；首末统计路径旁路后碎细节和结构误差一起上升。</td><td>这些功能耦合，不能把跳接直接当噪声源删除。</td></tr>
<tr><td>连续帧稳定</td><td>图里没有独立防抖模块。破坏首末统计或720跳接，会放大多场景的教师参照时间误差；弱化风格也会降低部分原始时间残差。</td><td>时间、风格和结构并行把关，不能以低残差单独放行。</td></tr>
<tr><td>低贡献候选</td><td>注意力残差在教师与修正版中都低敏感；但初始删除仍有一张广场图细节超线。</td><td>只作物理删除和恢复试验，最终结果另见视频页；40个解码通道未过筛查，均未强删。</td></tr></table><p>这些是当前权重和样本上的证据，不是全网所有通道的独占语义划分。</p>'''
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分支功能诊断</title><style>body{margin:24px auto;padding:0 20px;max-width:1440px;background:#151b24;color:#edf3fa;font:17px/1.65 system-ui}a{color:#abd4ff}img{max-width:100%}table{border-collapse:collapse;font-size:15px}td,th{border:1px solid #52657d;padding:7px}h2{margin-top:30px}summary{cursor:pointer;padding:16px}</style><h1>哪些路径在维持风格、结构和稳定？</h1><p><a href="review.html">本轮删减、修正与完整视频</a></p>'+conclusions+notes
    for p in ['skip720_comparison.png','attention_comparison.png','last_statistics_comparison.png']:html+=f'<a href="{p}"><img src="{p}"></a>'
    html+=''.join(tables)+'<p><a href="functional_deltas.json">独立三轴变化数据</a> · <a href="protocol.json">实验协议</a></p></html>'
    (run/'index.html').write_text(html,encoding='utf-8')
    write_json(run/'summary_manifest.json',dict(files={p:file_hash(run/p) for p in ['index.html','functional_deltas.json','skip720_comparison.png','attention_comparison.png','last_statistics_comparison.png']},diagnostic_only=True))
    print('SUMMARY_READY',flush=True)


if __name__=='__main__':main()
