"""Archive the single final attempt without promoting it over selected R39."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import zlib

OUT=Path(r'E:\fpga\output\b1-mc-pixel-20261027')
D=OUT/'pixel-r40-video-review'
EXP=OUT/'exports/pixel-A-last-coarsening-r40'
BASE=OUT/'exports/pixel-A-sharp-medium-r39'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    summary=read(D/'summary.json');assert summary['complete'] and not summary['strict8_succeeded']
    static=read(OUT/'pixel-r40-double-blocks/summary.json')
    verification=read(EXP/'verification.json')
    assert verification['crc32_padded']=='f74af46f'
    assert f'{zlib.crc32((BASE/"coefficients-padded.bin").read_bytes()):08x}'=='2d9aad69'
    assert sha(EXP/'coefficients-padded.bin')==verification['sha256_padded']
    assert static['controls']['base']['gray_curve_mean']==static['controls']['candidate']['gray_curve_mean']
    assert all(r['candidate_grid4']['nonuniform_cell_fraction']==0 for r in static['records'])
    sections=[];table=[]
    for report in summary['reports']:
        scene=report['scene'];title={'corridor':'室内 · 不过曝走廊','outdoor':'室外 · 街景'}[scene]
        sections.append(f'''<section><h2>{title} · 3 秒</h2>
<p>左：你选定的中强版　｜　右：最后一次块面合并尝试。下半部分是同位置原生像素局部。</p>
<video controls loop playsinline preload="metadata" poster="{scene}_comparison_0000.png" src="{scene}_comparison.mp4"></video>
<p class="links"><a href="{scene}_selected.mp4">已选中强版 · 1280×720 视频</a><a href="{scene}_candidate.mp4">本次尝试 · 1280×720 视频</a>
<a href="{scene}_candidate_0030.png">本次尝试无损原图</a></p></section>''')
        b=report['temporal']['selected_R39'];c=report['temporal']['candidate_R40']
        st=next(x for x in static['records'] if x['scene']==scene)
        table.append(f'<tr><td>{title}</td><td>{st["base_regions"]["mean_run_pixels"]:.2f} → {st["candidate_regions"]["mean_run_pixels"]:.2f} px</td><td>{b["warp_mae"]:.3f} → {c["warp_mae"]:.3f}</td><td>{b["hp_error_over_signal"]:.3f} → {c["hp_error_over_signal"]:.3f}</td></tr>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>已选中强 · 最后一次像素块放大尝试</title><style>
*{box-sizing:border-box}body{margin:0;background:#11151c;color:#eef2f7;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1328px;margin:auto;padding:28px 24px 60px}h1{font-size:29px;margin:0 0 12px}h2{font-size:23px;margin:0}p{margin:10px 0}section{margin-top:28px;padding:20px;background:#1b222e;border-radius:12px}video{display:block;width:100%;height:auto;background:#000}a{color:#9dccff}.links{display:flex;gap:20px;flex-wrap:wrap}.note{border-left:4px solid #d4ae61;padding:10px 18px;background:#26251e}.muted{color:#b5bfce}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #444c58;padding:9px}details{margin-top:26px}summary{cursor:pointer;font-weight:bold}.scroll{overflow:auto}
</style><main><h1>保留中强版；像素块放大最后一试</h1>
<p class="note"><strong>原生单元确认是 4×4。这次没有达到严格的 8×8。</strong><br>现有网络内做了邻近色块合并，部分块面变大，但基本网格仍为 4×4。你选定的 R39 中强版已保留，本次 R40 不替换它。</p>
<p class="muted">两侧均为实际导出 B1 系数的整数输出；室内、室外各 91 帧，约 3.03 秒，保留原时间戳。没有追加推理滤镜或重复帧。</p>
'''+''.join(sections)+'''<details><summary>实测结果与限制</summary>
<p>“同色段长度”统计相邻同色单元合并后的连续长度，不代表原生像素块边长。基础网格仍为 4×4，不能把色块合并量当作 8×8 成功。</p>
<div class="scroll"><table><tr><th>场景</th><th>平均同色段：中强 → 本次</th><th>运动补偿误差 ↓</th><th>高频误差/信号 ↓</th></tr>'''+''.join(table)+'''</table></div>
<p>时间指标由离线光流评估；推理不使用光流。块面合并也减少了部分细节，因此即使时间误差降低，也不能单独证明画质更好。中强版相对 R36 的时序退步记录仍保留。</p>
<p>固定网络只有两次步长 2 的降采样，内部对 4 像素平移保持等变，缺少区分相邻两组 4 像素单元的固定 8 像素相位。本次不更改拓扑、不加外部缩放，未实现严格 8 网格。</p>
<p>本次是实际网络系数构造，没有新增梯度训练。68800 个独立标量及 720p 全层整数一致性验证通过；不等于实板验收。连续运动的主观取舍以你观看视频为准。</p>
<p class="links"><a href="summary.json">视频测量</a><a href="DECISION.json">选择与失败记录</a><a href="../exports/pixel-A-last-coarsening-r40/verification.json">导出验证</a><a href="../pixel-r39-video-review/review.html">此前两档锐度对照</a></p></details>
</main></html>'''
    (D/'review.html').write_text(html,encoding='utf-8')
    decision=dict(status='FINAL_ATTEMPT_COMPLETED_STRICT8_NOT_ACHIEVED',user_selected=str(BASE),selected_crc32='2d9aad69',
        experiment=str(EXP),experiment_crc32='f74af46f',original_cell=4,requested_cell=8,actual_cell=4,
        strict8_succeeded=False,experiment_promoted=False,more_automatic_experiments=False,
        trained=False,additional_inference_filter=False,deployment_changed=False,
        reason='Colour masses coarsen but the physical 4-pixel lattice remains; strict 8-pixel requirement failed.',
        previous_selected_temporal_regression_preserved=True,MC_route_closed=True)
    save(D/'DECISION.json',decision);save(EXP/'DECISION.json',decision)
    selected=read(OUT/'SELECTED_PIXEL_CURVE.json')
    assert selected['user_selected_pixel_export']==str(BASE)
    selected.update(last_attempt_review=str(D/'review.html'),last_attempt_export=str(EXP),last_attempt_achieved=False,
        user_selected_review=str(OUT/'pixel-r39-video-review/review.html'),automatic_experiment_budget_exhausted=True)
    save(OUT/'SELECTED_PIXEL_CURVE.json',selected)
    (EXP/'README.txt').write_text('''R40 最后一次像素块放大实验 — 未达到严格 8×8

用户已选择 R39 中强（CRC32 2d9aad69）。本包是最后一次实验，不替换该选择。
本包 CRC32 f74af46f，固定 B1 10819 参数、24 卷积、29 节点、1280×720。
保留曲线 A 与中强限幅锐化，修改两个现有 DW 核，合并邻近色块。
实际仍为 4×4 基础网格，不能命名为 8×8 像素版。
纯整数 B1 生成；无新增梯度训练，无额外推理滤镜、缩放或光流。
独立整数 oracle 与 720p 全层数值验证通过，不是实板验收。
视频及详细结果：../../pixel-r40-video-review/review.html
主项目、Flash、已接受 A 包、水墨包和 V14 部署均未修改。
''',encoding='utf-8')
    snapshot=EXP/'source_snapshot';snapshot.mkdir(exist_ok=True)
    names=['try_pixel_double_blocks.py','evaluate_pixel_double_blocks.py','finalize_pixel_double_review.py','core.py','verify_export.py','evaluation_utils.py']
    for name in names:
        p=OUT/'source'/name;ast.parse(p.read_text(encoding='utf-8-sig'));shutil.copy2(p,snapshot/name)
    protected=[]
    for previous in read(OUT/'pixel-r39-video-review/review_validation.json')['protected']:
        p=Path(previous['path']);actual=sha(p);assert actual==previous['sha256']
        protected.append(dict(path=str(p),sha256=actual,unchanged=True))
    links=re.findall(r'(?:src|href|poster)="([^"]+)"',html)
    assert all((D/x).is_file() for x in links)
    manifest={str(p.relative_to(EXP)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in EXP.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(EXP/'manifest.json',dict(files=manifest,portable_source_snapshot=False,review=str(D/'review.html')))
    validation=dict(links_exist=len(links),protected=protected,video_complete=True,strict8_failed=True,
        selected_retained=True,subjective_player_review=False,html_rendered_browser_checked=False,
        free_gib={drive:shutil.disk_usage(drive+':/').free/2**30 for drive in ['C','E']})
    save(D/'review_validation.json',validation)
    print(json.dumps(dict(review=str(D/'review.html'),decision=decision,free_gib=validation['free_gib'],temporal=[dict(scene=r['scene'],metrics=r['temporal']) for r in summary['reports']]),ensure_ascii=False))

if __name__=='__main__':main()
