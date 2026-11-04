"""Freeze evidence of the corrected scene objective and failed methods."""
from core import *
import ast

def main():
    d=OUT/'scene-teacher-review';d.mkdir(exist_ok=True)
    ref=Path(r'C:\Users\Lenovo\AppData\Local\Temp\codex-clipboard-56bcacc4-1683-4d98-b672-877f6f330eab.png')
    base=Path(r'C:\Users\Lenovo\.codex\generated_images\01a17b77-a1ea-7d51-8ec6-e071fcbdcf72')
    files={'user_reference.png':ref,
        'scene_concept_composite.png':base/'exec-e8ec5015-1ab0-4be6-8653-6b5f241d62dc.png',
        'street_concept.png':base/'exec-4c1dcb33-d444-42ec-ae68-a8f0ff798e25.png',
        'training_park_concept.png':base/'exec-c81e8a3e-2a78-45b9-9347-e3d42230dc60.png'}
    for name,p in files.items():shutil.copy2(p,d/name)
    rejected={
      'r29-region-teacher':['Still photographic region shapes with small extrusions','Fails whole scene geometry'],
      'r30-joint-colour-teacher':['Fixture tests are not a whole-scene style test','User explicitly rejected teacher','Does not jointly reconstruct background'],
      'r32-scene-geometry-teacher':['Manually reconstructed foreground occlusion is wrong','Replaces bikes/foreground with planes','Indoor simplification excessive','Not valid full-scene training teacher'],
    }
    for name,reasons in rejected.items():save_json(OUT/name/'REJECTED_FOR_TRAINING.json',dict(eligible_for_training=False,reasons=reasons,
        produced_new_student=False,reason_is_visual_style_and_geometry=True))
    reports={k:json.loads((OUT/v/'report.json').read_text('utf-8')) for k,v in
        [('rgb_init','r34-full-scene-fit'),('hard_geometry_init','r35-hard-scene-fit')]}
    protected={
        str(WORK/'accepted/B1-A-20261008/coefficients-padded.bin'):'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523',
        str(WORK/'accepted/B1-水墨风-20261027/coefficients-padded.bin'):'d81910c309e14e4ecd965e39554cd13209f061c0e7e0b19c71e28c470d81b445',
        str(DEPLOY):'b62919cf6973530bf3d419e4c509c47321e8b42207384c728d1b9c2b613ada0f'}
    for p,h in protected.items():assert sha(p)==h,(p,sha(p),h)
    sources=['scene_geometry_teacher.py','fit_scene_geometry_diagnostic.py','review_pixel_r28.py','finalize_scene_teacher_audit.py']
    for name in sources:
        p=OUT/'source'/name;ast.parse(p.read_text('utf-8'));shutil.copy2(p,d/name)
    pixel=json.loads((OUT/'pixel-r28-video-review/summary.json').read_text('utf-8'))
    summary=dict(user_objective='scene-level cuboid silhouettes and coherent complete planes, shared perspective and occlusion; people are not required',
        wrong_previous_objective='posterization/luma relief plus independent local cubes',
        full_scene_style_proven=False,AI_images_are_not_B1_outputs=True,
        no_production_training_on_rejected_teachers=True,
        empirical_trials={k:dict(steps=v['steps'],selected_step=v['selected_step'],parameters=v['trained_parameters'],
            metrics=v['metrics'],seconds=v['seconds'],coefficient_sha256=v['coefficient_sha256'],
            general_MC_style_pass=False) for k,v in reports.items()},
        inference_extra_operators=False,review_clips_used_for_training=False,
        teacher_pair_limitations='AI geometry changes and manually estimated registration; finite optimization. Failure is evidence about these methods, not a proof all B1 coefficients are incapable.',
        failure_judgement='visual review required; neither numeric loss nor toy cube success is accepted as scene success',
        rejected_teachers=rejected,
        pixel_trial={r['scene']:dict(frames=r['frames'],original_warp=r['temporal']['original_A']['warp_mae'],
            minimal_cleanup_warp=r['temporal']['candidate_r28']['warp_mae']) for r in pixel['reports']},
        protected_sha256=protected,syntax_checked=sources,disk=disk_guard())
    save_json(d/'audit.json',summary)
    for k in ['r34-full-scene-fit','r35-hard-scene-fit']:
        r=json.loads((OUT/k/'report.json').read_text('utf-8'))
        save_json(OUT/k/'STYLE_NOT_ACCEPTED.json',dict(style_pass=False,
            reason='Did not produce whole-scene complete cuboids; numerical optimization is not style acceptance',
            seen_training_scene_only=True,coefficient_sha256=r['coefficient_sha256']))
    rows=[]
    for name in ['r34-full-scene-fit','r35-hard-scene-fit']:
        rows.append(f'<h2>{name}：训练场景、目标、实际整数输出</h2><img src="../{name}/training_scene_comparison.jpg"><h3>独立来源街景</h3><img src="../{name}/unseen_outdoor_comparison.jpg">')
    html='''<!doctype html><meta charset="utf-8"><title>MC场景教师纠正与实际B1试验</title>
<style>body{font:18px system-ui;background:#161b22;color:#e6edf3;max-width:1600px;margin:24px auto;padding:0 20px}img{width:100%;display:block;margin:12px 0}h1{font-size:28px}h2{font-size:23px}p{max-width:1000px;line-height:1.65}a{color:#80c9ff}.note{background:#283240;padding:14px}</style>
<h1>目标是整个场景由方块组成</h1><p>整片树冠、方形树干、连续墙柱梁面和共同透视是验收重点。人物不列入本轮要求。颜色平台、局部凸边、单立方体测试均不能代替场景效果。</p>
<p class="note">下方生成图是外观目标，不是B1输出。两轮学习均为单训练场景诊断；室内走廊与室外街景没有用于训练。场景MC尚未通过。</p>
<h2>用户原始目标</h2><img src="user_reference.png"><h2>重新生成的整片场景参考</h2><img src="street_concept.png">'''+''.join(rows)+'''
<h2>独立的纯像素路线</h2><p>原曲线A仍为选定版本。R28仅清理孤立杂色，未改Y块面。以下视频保留原始时序，各91帧约3.03秒。</p>
<p><a href="../pixel-r28-video-review/corridor_comparison.mp4">室内对比</a>　<a href="../pixel-r28-video-review/outdoor_comparison.mp4">室外对比</a></p>'''
    (d/'review.html').write_text(html,encoding='utf-8')
    (d/'README.md').write_text('''# MC场景教师纠正与试验记录

用户目标：整片树冠由完整方块群组成，树干方柱化，建筑/墙柱/梁面连续且共用透视，保留遮挡。人物暂不作为验收项。

旧教师把亮度分层或局部色块挤出当成几何，方向不符合目标。R29/R30停用；R32手工场景教师也因遮挡错误与过度简化被否决，未拿这些目标训练新学生。

改用训练集公园场景生成完整块面目标，做路径配准并在训练损失中排除天空与远处人物。R34从RGB初始化学习全部10819参数；R35从硬边三面表示初始化，仍学习全部参数并加重长边差异。两者均只使用既有训练来源121509；走廊与街景保持源隔离。

这是有限的同场景容量诊断，不是通用风格训练成功，也不能据此证明全部B1系数都不可能。生成目标存在几何重建/局部配准误差，需与模型容量、损失和优化因素分开评价。实际图像均由普通B1整数系数生成；训练配准、遮罩和损失不进入推理。

实际目标与学生差距见review.html及两轮training_scene_comparison.jpg。没有部署新MC包，没有修改硬件、Flash、A、水墨或V14基点。

纯像素R28视频为另一条路线的最小杂色修正，原A仍选定。两段均91帧，原始时间戳，无重复帧或后处理。它不证明MC风格成功。
''',encoding='utf-8')
    manifest={p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save_json(d/'manifest.json',manifest)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
