"""Summarize style significance, fragmentation and temporal diagnostics."""
from pathlib import Path
import json,statistics
from library import D,read_inventory,save_json

def avg(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None

def main():
    rows=[]
    for e in read_inventory():
        if e['id']=='BW_LINE':continue
        key=e['id'];reports=[json.loads((D/'review/native'/key/(s+'.json')).read_text('utf-8')) for s in ['indoor','outdoor','crowd','water']]
        static=json.loads((D/'review/static'/key/'verification.json').read_text('utf-8'))
        pairs=[p for r in reports for p in r['temporal_records']]
        temporal={tag:{metric:avg([p[tag][metric] for p in pairs]) for metric in ['warp_mae','hp_warp_mae','hp_error_over_signal','unaligned_static_mae','mean_luma_jump']} for tag in ['old','new']}
        stats={tag:avg([r['metrics'][tag+'_tiny_islands']['tiny_cells_fraction'] for r in static['records']]) for tag in ['old','new']}
        row=dict(id=key,name=e.get('current_display_name',e['name']),old_new_mean_mae=avg([r['mean_metrics']['mae'] for r in reports]),old_vs_input_mae=avg([r['mean_metrics']['old_vs_input_mae'] for r in reports]),new_vs_input_mae=avg([r['mean_metrics']['new_vs_input_mae'] for r in reports]),mae_raw_reference_frames_per_scene={r['scene']:r['exact_raw_metric_frames'] for r in reports},mean_tiny_cell_fraction_4_midframes=stats,temporal_20_sampled_pairs=temporal,native_frames=sum(r['frames'] for r in reports),all_native_frames_unique=all(r['native_video']['frames']==r['native_video']['unique_encoded_frames'] for r in reports),source_frames_and_pts_preserved=True,visual_accepted=False)
        rows.append(row)
    save_json(D/'style-and-temporal-summary.json',dict(status='DIAGNOSTICS_COMPLETE_NOT_AUTOMATIC_VISUAL_PASS',style_count=16,records=rows,notes=['No inference optical flow. Evaluation flow uses the identical original frames and masks for old/new.', 'Warp error depends on content/occlusion/flow and style detail; lower alone cannot prove visual stability.', 'Fragmentation is measured on four midpoint frames and is not a universal video/noise guarantee.', 'Four-scene sources are disjoint from this fit but previously seen in project reviews.']))
    lines=['# 十六款新结构：风格、碎片与时序诊断','','所有视频完整原PTS、152帧/场景，原生720p；每款608帧。光流仅评价，不参与输出。指标不是自动视觉验收。','',
           '|编号|旧→新MAE|原风格→原图MAE|新风格→原图MAE|旧→新补偿帧差|旧→新碎片格占比|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        t=r['temporal_20_sampled_pairs'];f=r['mean_tiny_cell_fraction_4_midframes']
        lines.append(f"|{r['id']}|{r['old_new_mean_mae']:.2f}|{r['old_vs_input_mae']:.2f}|{r['new_vs_input_mae']:.2f}|{t['old']['warp_mae']:.2f} → {t['new']['warp_mae']:.2f}|{100*f['old']:.2f}% → {100*f['new']:.2f}%|")
    lines+=['','旧→新MAE包含用户要求的有意弱边归并，不是单纯量化误差；前两列风格→原图距离仅表示改变幅度，不证明艺术风格质量。前4款MAE按每景全部152帧、后12款按每景13个原始整数核对帧计算，不用于跨款排名；各款新旧使用相同采样点。',
       '补偿帧差为每景5对、共20对相邻帧的平均值，含光流/遮挡误差，不能只靠抹细节降低它来判成功。实际风格显著性和运动观感须结合完整视频。',
       '碎片采用RGB每通道步长32量化、4×4取格、8连通、面积≤2格的小区域；四个中帧平均，定义相同，不能替代主观“干净”。','']
    (D/'风格与时序诊断.md').write_text('\n'.join(lines),encoding='utf-8')
    print('STYLE_AND_TEMPORAL_DIAGNOSTICS_COMPLETE',len(rows))
if __name__=='__main__':main()
