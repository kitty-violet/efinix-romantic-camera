"""Verify and show the fixed-size style-recovery trial honestly."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .repair_style_retention import RUN,PREVIOUS,change_loss
from .importance_pruned_teacher import build_group_pruned
from .prune_animation_teacher import profile
from .train_b1_distillation import as_tensor,to_rgb,read_image
from .fpga_video_resize import area_3to2,bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe,video_check
from .runtime import configure_workspace,write_json,file_hash,contract_guard


def load(root,run,mode):
    folder=run/mode;p=folder/'pruned_teacher_fp32.pt';export=json.loads((folder/'export.json').read_text())
    assert file_hash(p)==export['sha256'];state=torch.load(p,map_location='cpu',weights_only=True)
    net=build_group_pruned(root,state['indices']).cuda().eval();net.load_state_dict(state['generator']);return net


def inspect(root,run):
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));records=[]
    stages=dict(initial=root/'runs'/PREVIOUS/'cost14g_recovery/validation_000600',control=run/'control/validation_000240',repair=run/'repair/validation_000240')
    with torch.no_grad():
        for row in manifest['validation']:
            s=as_tensor(read_image(row['source']))[None];t=as_tensor(read_image(row['target']))[None]
            result={}
            for stage,folder in stages.items():
                p=as_tensor(read_image(str(folder/(row['id']+'.png'))))[None]
                loss,projection=change_loss(p,t,s)
                result[stage]=dict(change_loss=float(loss),band_projection=float(projection),rgb_mae255=float((p-t).abs().mean()*128))
            records.append(dict(id=row['id'],variants=result))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    panel=Image.new('RGB',(1280,946),'#151b24');draw=ImageDraw.Draw(panel)
    for i,label in enumerate(['原教师目标','上轮13.9G / 被指出偏弱','旧目标再训240步','修正目标再训240步']):draw.text((320*i+6,8),label,font=font,fill='white')
    for j,(key,xy,label) in enumerate([('val_stairs_090',(424,320),'石柱'),('val_stairs_090',(180,430),'植物'),('val_plaza_090',(80,448),'人物')]):
        row=next(x for x in manifest['validation'] if x['id']==key);paths=[Path(row['target'])]+[stages[k]/(key+'.png') for k in ['initial','control','repair']]
        y=42+288*j;draw.text((8,y),label,font=font,fill='#bad3ed');x0,y0=xy
        for i,path in enumerate(paths):panel.paste(Image.open(path).crop((x0,y0,x0+320,y0+256)),(320*i,y+28))
    draw.text((8,918),'同帧720原生像素裁切；三小网络同样123394参数；没有输出混合、锐化或贴脸。',font=font,fill='#bad3ed');panel.save(run/'comparison.png')
    means={stage:{key:sum(x['variants'][stage][key] for x in records)/len(records) for key in ['change_loss','band_projection','rgb_mae255']} for stage in stages}
    write_json(run/'static_comparison.json',dict(records=records,means=means,scope='Six previously seen regression frames. Teacher-change metric is not a style acceptance percentage.'))
    print('STATIC_COMPARISON',json.dumps(means),flush=True)


def verify(root,run):
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));binding=json.loads((run/'binding.json').read_text())
    assert all(file_hash(p)==h for p,h in binding['code'].items());samplers=[];results={}
    for mode in ['control','repair']:
        out=run/mode;rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rows]==list(range(1,241)) and all(np.isfinite(x['loss']) and x['grad_norm']>0 and x['native_frames']==2 for x in rows)
        assert json.loads((out/'launch_000120.json').read_text())['step']==120
        samplers.append([(x['source'],x['indices']) for x in rows]);net=load(root,run,mode);errors=[]
        assert all(p.dtype==torch.float32 for p in net.parameters())
        with torch.inference_mode():
            for row in manifest['validation']:
                x=as_tensor(read_image(row['source']))[None].cuda();fresh=to_rgb(net(x)[0]);saved=read_image(str(out/'validation_000240'/(row['id']+'.png')))
                errors.append(int(np.abs(fresh.astype(int)-saved.astype(int)).max()))
        assert max(errors)==0;prof=profile(net,x);assert prof['parameters']==123394 and prof['conv_macs']==13921689600
        result=dict(status='FP32_PASS_NOT_STYLE_ACCEPTANCE',steps=240,reload_pixel_error=0,profile=prof,source_counts=dict(Counter(x['source'] for x in rows)),
            optimizer_seconds=sum(x['seconds'] for x in rows),peak_mib=max(x['peak_mib'] for x in rows),weights_sha256=file_hash(out/'pruned_teacher_fp32.pt'))
        write_json(out/'verification.json',result);results[mode]=result;del net;torch.cuda.empty_cache()
    assert samplers[0]==samplers[1]
    write_json(run/'paired_verification.json',dict(status='PASS',steps_each=240,same_start_sha256=json.loads((run/'control/config.json').read_text())['initial_sha256'],same_order_all_240=True,
        graph_unchanged=True,weights_sha256={k:v['weights_sha256'] for k,v in results.items()}));print('PAIRED_VERIFY_PASS',flush=True)


def render(root,run):
    net=load(root,run,'repair');out=run/'videos';out.mkdir(exist_ok=False)
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'];records=[]
    for clip in clips:
        assert file_hash(clip['clip'])==clip['clip_sha256'];p=out/(clip['id']+'_repair_1080p.mp4')
        cap=cv2.VideoCapture(clip['clip']);pipe=encode_pipe(p,1920,1080,30);count=0;previous=None;duplicates=0;start=time.perf_counter()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            assert bgr.shape==(1080,1920,3);rgb=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            with torch.inference_mode():result=to_rgb(net(as_tensor(rgb)[None].cuda())[0])
            full=bilinear_2to3(result);data=full.tobytes();duplicates+=int(previous==data);previous=data;pipe.stdin.write(data)
            if count in (0,90,120,210,359,449):
                Image.fromarray(result).save(out/f'{clip["id"]}_{count:03d}_native720.png');Image.fromarray(full).save(out/f'{clip["id"]}_{count:03d}_preencode1080.png')
            count+=1
            if count%90==0:print('REPAIR_VIDEO',clip['id'],count,round(time.perf_counter()-start,1),flush=True)
        cap.release();close_pipe(pipe);assert count==clip['validation']['frames']
        records.append(dict(id=clip['id'],path=str(p),source=clip['clip'],source_sha256=clip['clip_sha256'],validation=video_check(p,count),model_calls=count,identical_adjacent_outputs=duplicates,seconds=time.perf_counter()-start))
        write_json(out/'report.json',dict(status='COMPLETE' if len(records)==len(clips) else 'RENDERING',records=records,
            weights_sha256=file_hash(run/'repair/pruned_teacher_fp32.pt'),pipeline='1080 integer area -> native720 FP32 network ->1080 integer bilinear',source_blend=False,sharpen=False,temporal_blend=False,crf=16))
    print('REPAIR_VIDEOS_COMPLETE',flush=True)


def diagnostics(root,run):
    from . import publish_fullres_c8_stylepush as old
    reference=run/'reference/videos';reference.mkdir(parents=True,exist_ok=False)
    write_json(reference/'report.json',json.loads((root/'runs'/PREVIOUS/'videos_cost14g/report.json').read_text()))
    old.PREVIOUS=RUN+'/reference';old.diagnostics(root,run)


def page(root,run):
    html=(root/'runs'/PREVIOUS/'index.html').read_text(encoding='utf-8')
    start=html.index('<h1>');end=html.index('<label>场景',start)
    html=html[:start]+'''<h1>风格保留修正：同一个13.9G网络</h1><p>上一轮风格变弱，用户验收未通过。本轮先停止继续剪枝：同一权重起点、同一网络、同样本顺序，旧目标与修正目标各继续240步。</p><p>修正版加强教师实际材质变化的学习，并改为匹配教师的时间变化；没有增加推理算子，也没有对输出锐化、混原片或贴脸。三段12／12／15秒，1080→720→1080。此页是实验结果，仍需判断是否足够接近教师。</p>'''+html[end:]
    html=html.replace('本轮按重要性剪枝版','本轮风格修正版').replace('上一轮半通道版','上轮13.9G弱风格版')
    html=html.replace('videos_cost14g/', 'videos/').replace("'videos/'+id+'_pruned_1080p.mp4'", "'videos/'+id+'_repair_1080p.mp4'")
    html=html.replace('../teacher-pruning-calibrated-20261003/videos_keep50/', '../'+PREVIOUS+'/videos_cost14g/')
    html=html.replace('cost14g_recovery/','repair/').replace('evaluation_cost14g/video_diagnostics.json','video_diagnostics.json')
    html=html.replace('原教师 / 上轮13.9G弱风格版 / 本轮风格修正版，点击看原尺寸。','图片四列：原教师 / 上轮13.9G / 旧目标继续240步 / 修正目标继续240步。点击看原尺寸。')
    html=html.replace('<p><a href="importance_summary.html">查看哪些连接组和通道更重要</a></p>','<p><a href="static_comparison.json">同起点静态对照</a> · <a href="paired_verification.json">训练顺序与重载检查</a></p>')
    html=html.replace('<a href="INFERENCE.md">权重与网络使用说明</a>', '<a href="../'+PREVIOUS+'/INFERENCE.md">图构造参考（加载本轮新权重）</a>')
    start=html.find('<p>广场11.97秒的近处人物：')
    if start>=0:html=html[:start]+html[html.index('<script>',start):]
    if (run/'near_faces.png').exists():html=html.replace('<script>','<p>广场11.97秒同帧近处人物：原片／教师／上轮13.9G／本轮修正。</p><a href="near_faces.png"><img src="near_faces.png" alt="本轮近处人物比较"></a><script>')
    (run/'index.html').write_text(html,encoding='utf-8');print('REPAIR_PAGE_READY',flush=True)


def static_page(root,run):
    checked=json.loads((run/'paired_verification.json').read_text());assert checked['status']=='PASS'
    values=json.loads((run/'static_comparison.json').read_text())['means']
    rows=''.join(f"<tr><td>{label}</td><td>{values[key]['change_loss']:.4f}</td><td>{values[key]['band_projection']:.4f}</td><td>{values[key]['rgb_mae255']:.3f}</td></tr>" for key,label in [('initial','上轮13.9G'),('control','旧目标续训240步'),('repair','修正目标续训240步')])
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>风格偏弱：同起点短对照</title><style>body{max-width:1280px;margin:24px auto;padding:0 18px;background:#151b24;color:#edf3fa;font:17px/1.8 system-ui}a{color:#afd5ff}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #587089;padding:8px}</style><h1>风格偏弱：修正仍不够，暂不继续压缩</h1><p>上轮13.9G版的风格强度未通过用户验收。本轮两组采用相同网络、相同起始权重和全部240步相同采样顺序。修正组加强教师局部纹理变化，并让时间约束参照教师。</p><p>数值有所改善，静态画面仍未明显追回教师的材质块面；本轮保留为诊断实验，未将其发布为新的成功版本，也未生成这两组的新视频或证明其视频稳定性。</p><p>四列：原教师 / 上轮13.9G / 旧目标再训240步 / 修正目标再训240步。点击原尺寸查看。</p><a href="comparison.png"><img src="comparison.png" alt="同起点教师、上轮、对照、修正四列图"></a><p>以下仅是六张已见回归图的数值，投影不是风格保留百分比，不能替代视觉判断。</p><table><tr><th>阶段</th><th>局部变化误差↓</th><th>多尺度变化投影</th><th>RGB误差/255</th></tr>ROWS</table><p><a href="RESULTS.md">判断与下一步</a> · <a href="paired_verification.json">同起点、同顺序与GPU重载证据</a> · <a href="temporal_term_audit.json">旧时间项的定向检查</a> · <a href="repair/pruned_teacher_fp32.pt">诊断用修正权重（非验收版）</a></p></html>'''
    (run/'index_static.html').write_text(html.replace('ROWS',rows),encoding='utf-8')
    lines=['# 风格保留不通过后的短对照','',
        '用户指出上一轮画风弱了很多。13.9G版不再作为风格保留成功版本；此前“只是略弱”的描述过于宽松。原教师、半通道父模型、13.9G历史权重与视频都保留。','',
        '## 已确认与仍不能断言的部分','',
        '同帧石柱、植物对照里，教师的大片材质统一和轮廓提炼未充分保留。此前剪枝参考半通道父模型的单通道平均损伤，并按损伤/MAC择取组合，不足以保证原教师的风格功能。中间解码d1从64到16、d3从32到8，压缩幅度很大，但仅凭宽度不能证明是唯一原因。','',
        '旧时间项是输出相对原片的变化经光流对齐后的残差。若把教师变化乘强度α，该项也乘α，所以原样输出可令该项为零。这是时间项单独的倾向，不代表空间损失加总后必然偏爱原样输出；不能据此认定它是全部根因。七个训练来源的实际检查见temporal_term_audit.json。合成混合仅用作损失诊断，没有作为推理产物交付。','',
        '## 实际实施的对照','',
        '两组均使用123394参数、13.9216896G的同一真实网络，从上一轮最终权重开始，各240步完整720p FP32。两组每一步训练来源和帧索引完全一致，均在120步结束进程再从磁盘恢复至240；GPU前后向、batch2小批次、断点恢复及六图独立重载通过。','',
        'control保留旧损失。repair增加按教师实际高频变化强度归一的局部多尺度损失，并将时间损失改成相对教师变化的误差。两项同时修改，因此该对照只说明组合效果，未拆出单项因果。推理图、参数和算量均未增加，无后加锐化、调色、原片/人脸混合或时间混帧。','',
        '|阶段|局部变化误差|多尺度变化投影|RGB误差/255|','|---|---:|---:|---:|']
    for key,label in [('initial','上轮13.9G'),('control','旧目标再训240步'),('repair','修正目标再训240步')]:
        v=values[key];lines.append(f"|{label}|{v['change_loss']:.4f}|{v['band_projection']:.4f}|{v['rgb_mae255']:.3f}|")
    lines += ['', '六张已见回归图；投影系数不是风格保留百分比，与上一轮全视频灰度单频投影也不是同一个测量定义。','',
        '[四列原尺寸对照](comparison.png) · [可查看页面](index_static.html)','',
        '## 本轮判断','',
        '修正组数字有改善，但石柱/植物仍没有出现足够明显的教师风格恢复；不能以这些变化覆盖用户的视觉反馈。当前不继续削减这版通道，也不把短修正实验升格为有效上板候选。240步试验不足以证明该网络永远无法恢复，但已经不足以支持直接延续当前压缩路线。','',
        '本轮未生成新视频，未验证修正版的长视频稳定性。不能宣称教师参照时间损失已经解决闪动；它也可能继承教师的时间变化。','',
        '下一步应回补被大幅缩减的中间解码能力，比较其他计算节省方法（如首尾7×7的空间分解），以原教师的材质变化和连续视频作为效果参照。只要风格明显退化，MAC减少和低时间残差都不能放行。该新结构尚未实施或验证。','',
        '## 实验权重和证据','']
    for mode in ['control','repair']:
        e=json.loads((run/mode/'export.json').read_text());v=json.loads((run/mode/'verification.json').read_text())
        lines += [f"- {mode}: [{mode}/pruned_teacher_fp32.pt]({mode}/pruned_teacher_fp32.pt)，SHA `{e['sha256']}`；240步，GPU优化{v['optimizer_seconds']:.1f}s，峰值{v['peak_mib']:.1f}MiB，六图重载像素差0。"]
    lines += ['', '仅诊断用FP32权重，不符合B1系数ABI，未做INT8、资源fit或物理时序验收。旧冻结源码不改；新训练代码为network_ab/training/repair_style_retention.py，源码/配置/数据SHA与检查点随本轮保存。','']
    (run/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    names=['RESULTS.md','index_static.html','comparison.png','static_comparison.json','paired_verification.json','temporal_term_audit.json','repair/pruned_teacher_fp32.pt','control/pruned_teacher_fp32.pt']
    write_json(run/'delivery_manifest.json',dict(status='DIAGNOSTIC_ONLY_INSUFFICIENT_VISUAL_RECOVERY',files={n:file_hash(run/n) for n in names},videos_generated=False,production=False))
    print('STATIC_REVIEW_READY',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inspect','verify','render','diagnostics','page','static_page']);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN;contract_guard();torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    globals()[args.mode](root,run)


if __name__=='__main__':main()
