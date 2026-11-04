"""Bind review evidence, protected hashes and candidate limits without adoption."""
from core import *
import ast

PROTECTED={
 WORK/'accepted/B1-A-20261008/coefficients-padded.bin':'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523',
 WORK/'accepted/B1-水墨风-20261027/coefficients-padded.bin':'d81910c309e14e4ecd965e39554cd13209f061c0e7e0b19c71e28c470d81b445',
 DEPLOY:'b62919cf6973530bf3d419e4c509c47321e8b42207384c728d1b9c2b613ada0f'}

def main():
    dest=OUT/'pixel-a-shapes-and-mc-review';summary=json.loads((dest/'summary.json').read_text('utf-8'))
    assert summary['all_six_complete'];files=list((OUT/'source').glob('*.py'))
    for p in files:ast.parse(p.read_text('utf-8-sig'),filename=str(p))
    protected={str(p):dict(expected=v,actual=sha(p)) for p,v in PROTECTED.items()}
    assert all(v['expected']==v['actual'] for v in protected.values())
    controls=json.loads((dest/'colour_controls.json').read_text('utf-8'));ref=np.array(controls['old_A']['gray_curve_mean'])
    for key in ['pixel_r21','pixel_r23']:assert np.array_equal(ref,np.array(controls[key]['gray_curve_mean']))
    clips=[]
    for c in summary['clips']:
        assert c['no_drop_or_repeated_frames'] and all(v['no_frame_drop_or_duplication'] for v in c['video'].values())
        assert all(v==c['frames'] for v in c['raw_unique_frames'].values())
        row=dict(id=c['id'],frames=c['frames'],seconds=c['seconds'],temporal={},content={})
        for key in ['old_A','pixel_r21','pixel_r23','mc_r20']:
            row['temporal'][key]=c['temporal'][key]
            fields=c['content'];row['content'][key]={k:float(np.mean([r[key][k] for r in fields])) for k in fields[0][key]}
        row['cell_geometry']={key:{'mean_straight_edge_run_pixels':float(np.mean([r[key]['geometry']['mean_straight_edge_run_pixels'] for r in c['cell_geometry']])),
            'nonconstant_cell_fraction':max(r[key]['nonconstant_cell_fraction'] for r in c['cell_geometry'])} for key in ['old_A','pixel_r21','pixel_r23']}
        clips.append(row)
    calibration=json.loads((OUT/'r20-multifeature-structure/calibrated/calibration.json').read_text('utf-8'))
    record=dict(user_selected_curve='A',saturation_relative=1.3,curve_A_uniform_256_exact=True,
        prior_approved_export=str(OUT/'exports/flat-r3-user-selected'),curve_A_export=str(OUT/'exports/pixel-user-curve-A'),
        new_spatial_candidates=['pixel-A-square-r21','pixel-A-square-r23'],new_spatial_revision_user_approved=False,
        pixel_changes='larger content-anchored rectangular Y contours, full nine-level chroma palette and symmetric H/V colour voting; R23 protects near-black/white edges',
        pixel_training='constructed ordinary coefficients, no new training run; independent integer-node verification completed',
        spatial_limit='native fixed four-pixel phase, content-dependent larger regions; not a fixed global6/8/16 grid and not a claim that all painterly contours are resolved',
        mc_research=dict(training_steps=700,warmup_steps=200,existing_low_channels_for_support=9,
            calibration_validation=calibration['validation_after_exact_quantization'],unseen_synthetic_test=calibration['unseen_test'],
            selected_lighting_negatives_pass=calibration['selected_lighting_negatives_pass'],positive_geometry_controls_pass=calibration['positive_geometry_controls_pass'],
            status='not accepted: some flat light/shadow controls still extrude; no global cuboid reconstruction'),
        clips=clips,protected_hashes=protected,source_AST_pass=len(files),
        native_integer_outputs=True,inference_filters=False,inference_flow=False,hardware_changed=False,Flash_written=False,
        style_finalized=False,global_temporal_superiority=False,disk=disk_guard())
    save_json(dest/'audit.json',record);save_json(OUT/'SELECTED_PIXEL_CURVE.json',dict(user_selected_curve='A',curve_export=record['curve_A_export'],
        prior_approved_export=record['prior_approved_export'],spatial_candidates=record['new_spatial_candidates'],spatial_candidate_approved=False,deployment_changed=False))
    source_names=['core.py','train_pixel.py','try_user_pixel_curves.py','r3_palette_blocks.py','r4_box_projector.py',
        'r16_pixel_layer_geometry.py','r21_colour_preserving_layers.py','r23_midtone_square_layers.py','verify_export.py']
    for name,label in [('pixel-A-square-r21','色层方块化 R21'),('pixel-A-square-r23','中间色层加强 R23'),('mc-structure-support-r20','MC结构支持研究候选 R20')]:
        d=OUT/'exports'/name;v=json.loads((d/'verification.json').read_text('utf-8'));snapshot=d/'source';snapshot.mkdir(exist_ok=True)
        names=source_names if name.startswith('pixel') else [p.name for p in files]
        for basename in names:shutil.copy2(OUT/'source'/basename,snapshot/basename)
        specification=('构造系数，未新增训练。A曲线256级均匀灰逐点保留；大色块依赖内容，基础空间相位仍为4像素，不能称为全局固定8/16像素网格。' if name.startswith('pixel') else
            'MC结构支持训练700步，前200步浮点热身，后500步普通B1 QAT；部分光斑和阴影仍误触发，不是完成的MC风格，也不使用像素路线的A曲线。')
        body=f'''# {label}\n\n状态：离线整数核验通过，未接受为风格成品，未上板。\n\n- 固定 B1：10819 参数（10256 权重、563 偏置），24 卷积、29 节点，1280×720。\n- 系数有效17792字节，传输补齐20480字节；CRC32 {v['crc32_padded']}。\n- 独立标准库参考68800次标量比较零误差；720p全部节点QAT与解码BIN一致。\n- 无额外推理滤镜、分割、光流或帧历史。\n- {specification}\n\n视频、逐帧PTS、内容与时序指标：{dest}\n\n原A、水墨、V14、硬件工程、Flash和部署记录未修改。\n'''
        (d/'README.md').write_text(body,encoding='utf-8')
        save_json(d/'review-binding.json',dict(review=str(dest),audit_sha256=sha(dest/'audit.json'),user_selected_tone_curve='A' if name.startswith('pixel') else None,
            visual_accepted=False,board_accepted=False,task_complete=False))
        save_json(d/'manifest.json',dict(files={str(p.relative_to(d)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in d.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    lines=['# 曲线A：像素色层形状与MC结构研究对照','',
        '用户已选曲线A。本目录是实际B1整数输出的候选对照，不是成品验收。',
        '', '室内、室外各3秒，其余四个不同场景各1秒；保持原生连续帧和时间戳。所有流的原始帧哈希均不重复；视频再次解码核验帧数与PTS。',
        '', '六宫格：上排原图、旧V14、原曲线A；下排R21色层候选、R23中间色层加强、MC未达标研究候选。',
        'station_ceiling_crop.mp4为用户圈选天花板附近的同步局部放大，只用于观察，未用于推理或指标。',
        '', '像素：保留A的全部256级均匀灰响应和130%色度系数，扩大局部横竖块面约束；R23保留更小的黑白轮廓变动。基础像素相位仍为4，色层形状尚需视觉确认。',
        'MC：原单支持通道会塌缩，改为用已有9通道训练局部结构支持。新独立构造样本命中率约70.0%、假阳性约22.7%；选定灰色/彩色物体有侧面，但光斑/阴影反例未全部通过，禁止称MC成功。',
        '', '不能只凭warp误差低判稳定：audit.json同时列原图关联、边缘能量、对比和高频误差/信号比；未宣称时序全面优于旧版。',
        '', '|片段|帧数|时长/秒|A原版warp|R21 warp|R23 warp|', '|---|---:|---:|---:|---:|---:|']
    for c in clips:lines.append(f"|{c['id']}|{c['frames']}|{c['seconds']:.4f}|{c['temporal']['old_A']['warp_mae']:.3f}|{c['temporal']['pixel_r21']['warp_mae']:.3f}|{c['temporal']['pixel_r23']['warp_mae']:.3f}|")
    lines+=['','缓存逐文件核对后已清理；原A/水墨/V14三份系数SHA核验一致。详细证据见audit.json、binding.json和各片段report.json。']
    (dest/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(dest/'manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('A_SHAPE_REVIEW_AUDIT_COMPLETE',json.dumps(dict(clips=len(clips),disk=record['disk'],MC_accepted=False,style_accepted=False)),flush=True)

if __name__=='__main__':main()
