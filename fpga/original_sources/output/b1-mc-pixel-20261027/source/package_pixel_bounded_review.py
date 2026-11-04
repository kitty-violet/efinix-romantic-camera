"""Package both bounded-sharpening candidates and their actual integer videos."""
from pathlib import Path
import hashlib,json,shutil,re
from html.parser import HTMLParser

OUT=Path(__file__).resolve().parents[1]
D=OUT/'pixel-r39-video-review'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def cleanup_interrupted():
    # Only this earlier evaluation's own, known frame names; never recursive.
    parent=(OUT/'cache').resolve();p=parent/'pixel-r37-corridor'
    if not p.exists():return 0
    assert p.resolve().parent==parent and p.resolve()==p and not p.is_symlink()
    files=list(p.iterdir())
    assert len(files)<=184
    for f in files:
        assert f.is_file() and not f.is_symlink() and f.resolve().parent==p
        assert re.fullmatch(r'(native|comparison)_\d{4}\.png',f.name) or f.suffix=='.ffconcat'
    size=sum(f.stat().st_size for f in files)
    for f in files:f.unlink()
    p.rmdir()
    return size

def main():
    r=read(D/'summary.json');assert r['complete']
    assert len(r['encoded'])==4 and all(e['frames']==e['unique_encoded_frames']==91 for e in r['encoded'])
    static=read(OUT/'pixel-r39-bounded-sharp/summary.json')
    exports={k:OUT/'exports'/f'pixel-A-sharp-{k}-r39' for k in ['medium','strong']}
    verification={k:read(p/'verification.json') for k,p in exports.items()}
    for k,v in verification.items():
        assert v['small_oracle_scalar_comparisons']==68800
        assert all(x['max_error']==0 for x in v['full720_qat_vs_decoded_integer_nodes'].values())
        assert v['sha256_padded']==sha(exports[k]/'coefficients-padded.bin')
    labels={'corridor':'室内走廊','outdoor':'室外街景','dusk':'傍晚','park':'公园','foliage':'树叶'}
    metric=[]
    for scene in r['reports']:
        t=scene['temporal']
        metric.append(dict(scene=scene['scene'],frames=scene['frames'],values={k:t[k] for k in ['original_A','r36','medium_r39','strong_r39']}))
    cards=[]
    for scene in ['corridor','outdoor']:
        cards.append(f'''<section><h2>{labels[scene]} · 3 秒</h2>
<p class="muted">更强档 · 固定 B1 整数输出 · 1280×720 · 91 个不同帧</p>
<video controls playsinline preload="metadata" poster="{scene}_native_0000.png" src="{scene}_native.mp4"></video>
<p><a href="{scene}_native.mp4">更强档单独播放</a> · <a href="{scene}_medium.mp4">中强档单独播放</a> · <a href="{scene}_native_0000.png">无损原始帧</a></p>
<details><summary>看原锐度与更强档的同步对照</summary><p class="muted">左侧 R36，右侧更强 R39；上面是全景，下面是同位置原生像素局部。</p>
<video controls playsinline preload="metadata" src="{scene}_comparison.mp4"></video></details>
<details><summary>看中强档</summary><video controls playsinline preload="metadata" src="{scene}_medium.mp4"></video></details></section>''')
    table=[]
    for s in metric:
        vals=s['values'];table.append('<tr><td>'+labels[s['scene']]+'</td>'+''.join(f'<td>{vals[k]["warp_mae"]:.3f} / {vals[k]["hp_error_over_signal"]:.3f}</td>' for k in ['r36','medium_r39','strong_r39'])+'</tr>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>像素 R39 · 明显提高锐度</title><style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#11161d;color:#edf1f6;font:16px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:28px 20px 64px}h1{font-size:29px;margin:0 0 10px}h2{font-size:22px;margin:0}p{margin:8px 0}.muted{color:#acb9ca;font-size:14px}section{background:#1c2531;border-radius:12px;margin:22px 0;padding:18px}video{width:100%;height:auto;display:block;background:black;margin:12px 0;image-rendering:pixelated}a{color:#9bc2ff}summary{cursor:pointer;font-weight:600}details{border-top:1px solid #354352;margin-top:14px;padding-top:12px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #354352}.scroll{overflow:auto}img{max-width:100%;height:auto;image-rendering:pixelated}</style>
<main><h1>像素 R39：中强 / 更强锐化</h1>
<p>这轮直接加强像素色块交界的明暗反差。两档都保留偏黄修正；中强更克制，更强档的边缘更突出。</p>
<p>检查结果：锐化差异明显，但两档的时序指标都比 R36 退步，更强档退步更多。本页用于比较锐化强度，尚不满足“更锐且保持原稳定性”的定稿条件。</p>
<p class="muted">这是实际网络系数输出，没有视频后滤镜。原曲线 A 和 R36 均保留，尚未替换用户已选版本。</p>'''+''.join(cards)+'''
<section><h2>变化与取舍</h2><p>静态三场景测得，原生像素格相邻亮度差相对轻量 R37：中强约 1.5 倍，更强约 2 倍。这个数值表示边界反差，不能解释为分辨率或细节翻倍。</p>
<p>中强档额外增强最多约 8 个 RGB 灰度级，更强档约 15 级。已放弃不限幅的 R38，因为会出现很重的亮边。限幅后更强档仍会出现可见的亮暗边缘，这是这档强锐化的取舍。</p>
<p>均匀灰色已经严格 R＝G＝B；相对原曲线 A，均匀灰阶均值改动最多 0.67 级。浅色色度第一档从 16 收到 12，减轻暖灰跨档后突然黄褐化；原本的暖光和黄色物体保留。</p>
<details><summary>看三档静帧对照</summary><img src="../pixel-r39-bounded-sharp/corridor_comparison.jpg" alt="室内原锐度、中强和更强对照"><img src="../pixel-r39-bounded-sharp/outdoor_comparison.jpg" alt="室外原锐度、中强和更强对照"></details></section>
<section><h2>时间稳定性实测</h2><p class="muted">单元格：运动对齐误差 / 高频误差与信号之比，数值越低越好。锐化会放大边界变化，不能只因更锐就声称更稳定。</p><div class="scroll"><table><tr><th>场景</th><th>R36</th><th>R39 中强</th><th>R39 更强</th></tr>'''+''.join(table)+'''
</table></div><p class="muted">室内外各 91 帧，约 3.03 秒；另外三场景各 31 帧、约 1 秒。原时间戳、全帧推理，编码后完整解码且帧唯一，无降帧或重复帧。光流仅离线测量。已审查静帧和逐帧数值，未作播放器连续运动的主观验收。</p></section>
<section><h2>交付依据</h2><p>固定 B1 的 10,819 个参数、24 卷积、29 节点和 720p 接口保持。两档都通过 68,800 次独立标量比较及 720p 全层整数一致性检查。本轮是现有系数调整，没有新增梯度训练，没有修改硬件或部署。</p>
<p><a href="../exports/pixel-A-sharp-strong-r39/README.md">更强档包说明</a> · <a href="../exports/pixel-A-sharp-medium-r39/README.md">中强档包说明</a> · <a href="summary.json">完整视频记录</a></p>
<p class="muted">MP4 为有损观看编码。PNG 为无损实际输出；最终效果来自系数包。MC 路线保持终止。</p></section></main></html>'''
    (D/'review.html').write_text(html,encoding='utf-8')
    protected=read(OUT/'pixel-r36-video-review/review_validation.json')['protected']
    for item in protected:assert sha(Path(item['path']))==item['sha256']
    for mode,p in exports.items():
        v=verification[mode];cap=3 if mode=='medium' else 6
        text=f'''# 纯像素 R39 · {'中强' if mode=='medium' else '更强'}锐化候选

未定稿、未替换部署；软件整数验证通过，不代表本轮实板验证。锐化目标已增强，但视频时序指标相对R36退步，当前不满足同时保留原稳定性的定稿条件。

- 继承曲线 A、S130 和像素结构；灰色整数中性点已修正，256 级 R=G=B，均值相对原 A 最大变化 0.67。
- 浅色色度首档 16→12，强色度其余档保留；对比度曲线未另加拉伸。
- 固定 B1 现有通道实现限幅亮度高通：两路 ReLU 提取正负高频，再以 ReLU(p)-ReLU(p-cap) 限幅，现有 PW 合成。cap={cap}，无新增算子或推理后处理。原生 4×4 同色单元保持。
- 本轮系数构造，没有新增梯度训练。不宣称纹理/分辨率增加；强档仍可能有可见亮暗边缘。时序详见完整报告，不能宣称完全无闪动。
- 10,819 参数=10,256 权重+563 偏置；24 卷积/29 节点/1280×720。
- CRC32 padded `{v['crc32_padded']}`；SHA256 `{v['sha256_padded']}`。
- 6 组独立标量检查合计 68,800 值、720p 全层 QAT/导出整数精确一致；BIN/HEX往返及补零校验通过。
- 两个主视频各91帧、3.0302秒；附加三个场景各31帧约1秒。完整解码、帧唯一、原PTS通过。光流仅评估，推理无历史帧。
- `residual_encoding.json` 随包保留；未修改硬件、Flash、已接受A/水墨包或V14部署。

观看：`../../pixel-r39-video-review/review.html`。MP4有损，PNG无损。`source_snapshot`依赖原项目隔离环境和原A/R28/R37资产，并非独立可安装工程。
'''
        (p/'README.md').write_text(text,encoding='utf-8');snap=p/'source_snapshot';snap.mkdir(exist_ok=True)
        for name in ['core.py','probe_pixel_colour_sharp.py','probe_pixel_bounded_sharp.py','evaluate_pixel_bounded_sharp.py','verify_export.py','package_pixel_bounded_review.py']:
            shutil.copy2(OUT/'source'/name,snap/name)
        save(p/'manifest.json',dict(candidate=p.name,user_accepted=False,MC_closed=True,protected_unchanged=True,temporal_regression=True,
             files={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='manifest.json'}))
    selected=read(OUT/'SELECTED_PIXEL_CURVE.json')
    selected.update(latest_preview_export=str(exports['strong']),latest_preview_name=exports['strong'].name,
        latest_preview_review=str(D/'review.html'),latest_preview_approved=False,
        alternative_preview_export=str(exports['medium']))
    save(OUT/'SELECTED_PIXEL_CURVE.json',selected)
    save(D/'DECISION.json',dict(status='STRONGER_SHARPNESS_PREVIEW_TEMPORAL_REGRESSION_NOT_ACCEPTED',
        sharpen_request_demonstrated=True,temporal_acceptance=False,deploy=False,
        explanation='Both sharpened candidates increase temporal errors against R36. Keep original accepted selection; show strength trade-off without claiming stability success.'))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            self.paths.extend((D/v).resolve() for k,v in attrs if k in ['src','href','poster'])
    parser=Links();parser.feed(html);assert all(p.is_file() for p in parser.paths)
    cleaned=cleanup_interrupted()
    save(OUT/'pixel-r37-video-review/STOPPED.json',dict(status='superseded_by_user_stronger_sharpening_request',
        full_video_audit_completed=False,do_not_present_as_final=True,cleaned_cache_bytes=cleaned,replacement=str(D)))
    save(D/'review_validation.json',dict(links_exist=len(parser.paths),protected=protected,video_complete=True,
        temporal=metric,subjective_motion_player_review=False,html_rendered_browser_checked=False,
        cleaned_interrupted_cache_bytes=cleaned,free_gib={k:shutil.disk_usage(k+':/').free/2**30 for k in ['C','E']}))
    print(json.dumps(dict(review=str(D/'review.html'),cleanup_bytes=cleaned,links=len(parser.paths),
        crcs={k:v['crc32_padded'] for k,v in verification.items()}),ensure_ascii=False))

if __name__=='__main__':main()
