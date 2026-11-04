"""Package the verified pixel candidate and a local video review. No inference edits."""
from pathlib import Path
import hashlib
import json
import shutil
from html.parser import HTMLParser

OUT = Path(__file__).resolve().parents[1]
D = OUT / 'pixel-r36-video-review'
EXPORT = OUT / 'exports/pixel-A-crisp-r36'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    report = read(D/'summary.json')
    static = read(OUT/'pixel-r36-crisp-edges/hard-flat/report.json')
    verification = read(EXPORT/'verification.json')
    assert report['complete'] and report['MC_route_closed']
    assert static['gray_curve_A_exact'] and not static['trained']
    assert verification['crc32_padded'] == '08534d00'
    assert sha(EXPORT/'coefficients-padded.bin') == verification['sha256_padded']
    assert all(e['frames'] == e['unique_encoded_frames'] == 91 for e in report['encoded'])
    assert all(r['all_frames_preserved'] for r in report['reports'])
    protected = {
        Path('E:/fpga/model-training-workspace/accepted/B1-A-20261008/coefficients-padded.bin'): 'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523',
        Path('E:/fpga/model-training-workspace/accepted/B1-水墨风-20261027/coefficients-padded.bin'): 'd81910c309e14e4ecd965e39554cd13209f061c0e7e0b19c71e28c470d81b445',
        Path('E:/fpga/output/deployment/board-full-150-video-standalone222-20261024/styles/V14/coefficients-padded.bin'): 'b62919cf6973530bf3d419e4c509c47321e8b42207384c728d1b9c2b613ada0f',
    }
    protected_result = []
    for p, expected in protected.items():
        actual = sha(p)
        assert actual == expected, p
        protected_result.append(dict(path=str(p), sha256=actual, unchanged=True))
    for r in report['reports']:
        assert r['temporal']['crisp_r36']['warp_mae'] < r['temporal']['original_A']['warp_mae']
        assert r['temporal']['crisp_r36']['hp_error_over_signal'] < r['temporal']['original_A']['hp_error_over_signal']
    labels = {'corridor':'室内走廊', 'outdoor':'室外街景', 'dusk':'傍晚', 'park':'公园', 'foliage':'树叶'}
    rows = []
    for r in report['reports']:
        old = r['temporal']['original_A']; new = r['temporal']['crisp_r36']
        rows.append(f"<tr><td>{labels[r['scene']]}</td><td>{r['frames']}</td><td>{old['warp_mae']:.3f} → {new['warp_mae']:.3f}</td><td>{old['hp_error_over_signal']:.3f} → {new['hp_error_over_signal']:.3f}</td></tr>")
    cards = []
    for scene in ['corridor','outdoor']:
        cards.append(f'''<section><h2>{labels[scene]} · 3 秒</h2>
<p class="sub">R36 实际 B1 整数输出 · 1280 × 720 · 91 帧 · 约 3.03 秒</p>
<video controls playsinline preload="metadata" poster="{scene}_native_0000.png" src="{scene}_native.mp4"></video>
<p><a href="{scene}_native.mp4">单独播放视频</a> · <a href="{scene}_native_0000.png">查看无损原始帧</a></p>
<details><summary>与原曲线 A 对照：左旧版，右新版</summary>
<p class="sub">上排看全景，下排看同位置的原生像素局部。对照视频没有插帧或降帧。</p>
<video controls playsinline preload="metadata" src="{scene}_comparison.mp4"></video></details></section>''')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>纯像素 R36 · 色层边界整理</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#11151b;color:#e9edf3;font:16px/1.75 "Microsoft YaHei",sans-serif}
main{max-width:1320px;margin:auto;padding:28px 20px 64px}h1{font-size:30px;margin:0 0 10px}h2{font-size:22px;margin:0 0 6px}
p{margin:8px 0}.sub{color:#aab7c8;font-size:14px}.tag{display:inline-block;background:#253a35;color:#aee5cd;padding:2px 12px;border-radius:20px;font-size:13px}
section{margin-top:24px;background:#1b222c;padding:18px;border-radius:12px}video{display:block;width:100%;height:auto;margin:14px 0;background:#000;image-rendering:pixelated}
a{color:#91bcff}details{border-top:1px solid #34404d;padding-top:12px;margin-top:14px}summary{cursor:pointer;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #34404d}.scroll{overflow:auto}
code{font-family:Consolas,monospace}ul{padding-left:22px}img{max-width:100%;height:auto;image-rendering:pixelated}
</style><main><span class="tag">只完善纯像素 · MC 已终止</span><h1>保留原版块面，收干净色层边界</h1>
<p>保留你选定的曲线 A 和饱和度 130%，清理孤立杂色、边缘多余的中间色阶与明暗修饰。这次没有扩张整片亮度色块。</p>
<p class="sub">原版仍保留。R36 是这轮待看的候选，没有替换已选版或部署文件。</p>''' + ''.join(cards) + '''
<section><h2>这次具体改了什么</h2><ul>
<li>清理小片孤立色边，减少颜色像多层颜料叠在一起的感觉。</li>
<li>在现有网络末端收紧色阶边界，去掉额外的方向性边缘明暗。</li>
<li>256 级均匀灰块的输出与曲线 A 完全一致；末端二值区域归属不变。原生 4×4 单元内部保持同色。</li>
</ul><details><summary>查看像素局部：原曲线 A / 去孤点 R28 / 本次 R36</summary>
<img src="../pixel-r36-crisp-edges/outdoor_crop.png" alt="三版室外同位置局部，以最近邻放大，不做平滑"></details></section>
<section><h2>视频检查</h2><p>相同输入、相同时间戳逐帧推理。两段主视频均为 91 个不同输出帧；无降帧、无重复帧。其余三段各约 1 秒作补充短测。</p>
<div class="scroll"><table><thead><tr><th>场景</th><th>帧数</th><th>运动对齐误差，越低越好</th><th>高频误差 / 信号，越低越好</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table></div>
<p class="sub">均为原 A → R36。相对 R28，硬边会让部分高频误差略升；室外高频绝对误差相对原 A 也有约 0.09% 的小幅增加。这是边缘更硬的代价，不能据此宣称所有指标都改善或完全无闪动。</p>
<p class="sub">已检查静帧和逐帧统计；未作播放器连续运动的主观验收。光流只用于离线测量，不进入推理。</p></section>
<section><h2>实际网络与导出</h2><p>固定 B1，10,819 个参数。效果写在网络系数里，没有额外滤镜。本轮为已有学生网络的系数调整，没有新增梯度训练。</p>
<p>68,800 次独立标量比较、720p 逐层整数一致性检查均通过。CRC32：<code>08534d00</code>。这是软件验证候选，未作本轮实板验证。</p>
<p><a href="../exports/pixel-A-crisp-r36/README.md">候选包说明</a> · <a href="summary.json">完整视频记录</a> · <a href="../exports/pixel-A-crisp-r36/verification.json">整数验证记录</a></p>
<p class="sub">MP4 是观看编码，颜色边界可能受有损压缩影响；无损 PNG 和系数包保留实际输出证据。</p></section></main></html>'''
    (D/'review.html').write_text(html, encoding='utf-8')
    sources = EXPORT/'source_snapshot'; sources.mkdir(exist_ok=True)
    filenames = ['core.py','probe_pixel_crisp_edges.py','probe_minimal_pixel_cleanup.py',
        'evaluate_pixel_crisp.py','verify_export.py','evaluation_utils.py','evaluate_pixel.py',
        'evaluate_r3_r4.py','r13_region_cuboids.py','package_pixel_crisp_review.py']
    for name in filenames:
        shutil.copy2(OUT/'source'/name, sources/name)
    (EXPORT/'README.md').write_text('''# 纯像素曲线 A · R36 候选

状态：离线软件整数验证通过，尚未获用户定稿，也不是实板验收。MC 路线已由用户否决并终止。

保留曲线 A 的 256 级均匀灰阶响应及 S130 设置。沿用原版块面，先清理色度孤点，再收紧末端阈值掩码并去掉 RGB 头的方向性明暗修饰。相对 R28，末端二值区域归属保持；并非每个 RGB 像素不变。本轮通过构造现有 B1 系数实现，没有新增梯度训练。

- 固定 B1：24 卷积、29 节点、1280×720；10,819 参数（10,256 权重 + 563 偏置）。
- 效果全部来自系数；推理没有后滤镜、光流、历史帧或分割器。
- 原生 4×4 同色单元，没有声称改成全局 6×6 网格。
- 系数有效长 17,792 字节，补零传输长 20,480 字节；CRC32 padded `08534d00`。
- SHA256 padded `12af10eab0bff9978c26112f926c0ab1acb9aef3868cbbb8815c8ee8f25d7d9a`。
- `verification.json`：6 组独立标量算例共 68,800 值和 720p 全层整数输出一致，codec 往返及补零校验通过。
- `residual_encoding.json` 必须随包保留，不能仅更换系数后默认程序配置必然兼容；未改硬件、Flash、已接受的 A 包/水墨包或现行部署。
- 视频：室内走廊、室外街景各 91 帧、约 3.03 秒，保留原时间戳，全帧推理，编码后完整解码且各 91 帧唯一。补测傍晚、公园、树叶各约 1 秒。
- 相对原曲线 A，五场景运动对齐误差和高频归一化误差都下降。相对 R28，部分高频误差上升；室外绝对高频误差相对原 A 微升约 0.09%。未据此宣称完全无闪动。
- 已检查静帧与数值；尚无播放器连续观看的主观验收记录。

观看入口：`../../pixel-r36-video-review/review.html`。无损样例 PNG、原时间戳、模型绑定和视频 SHA 在该目录。视频仅供观看，为有损 H.264 编码。

代码快照在 `source_snapshot/`，依赖项目内现有 `model-training-workspace` 与原 A/R28 资产。它是来源快照，不是独立可安装训练工程。`r13_region_cuboids.py` 在此仅提供通用颜色连通域统计，不表示重启 MC。

当前用户已选版本仍为 `pixel-user-curve-A`。本候选仅供比较，未覆盖 accepted 或 deployment。
''', encoding='utf-8')
    selected_path = OUT/'SELECTED_PIXEL_CURVE.json'; selected = read(selected_path)
    selected.update(latest_preview_export=str(EXPORT), latest_preview_name='pixel-A-crisp-r36',
        latest_preview_approved=False, latest_preview_review=str(D/'review.html'), MC_route_closed=True)
    save(selected_path, selected)
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.paths=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src','poster'] and value:
                    self.paths.append((D/value).resolve())
    parser=Links();parser.feed(html)
    assert all(p.is_file() for p in parser.paths), [str(p) for p in parser.paths if not p.is_file()]
    files = [p for p in EXPORT.rglob('*') if p.is_file() and p.name!='manifest.json']
    save(EXPORT/'manifest.json', dict(candidate='pixel-A-crisp-r36', user_accepted=False,
        deployment_changed=False, MC_route_closed=True, protected=protected_result,
        files={str(p.relative_to(EXPORT)):sha(p) for p in files}))
    save(D/'review_validation.json', dict(local_links_checked=len(parser.paths),all_exist=True,
        review_sha256=sha(D/'review.html'), encoded_verified=report['encoded'],protected=protected_result,
        no_rendered_browser_audit=True,static_images_visually_reviewed=True,
        free_gib={drive:shutil.disk_usage(drive+':/').free/2**30 for drive in ['C','E']}))
    print(json.dumps(dict(review=str(D/'review.html'),files=len(files),links=len(parser.paths),
        protected_unchanged=True,accepted=False),ensure_ascii=False))

if __name__=='__main__':main()
