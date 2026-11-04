"""Package the selected A result without changing or postprocessing pixels."""
import argparse,html,json,os,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--data',required=True);ap.add_argument('--current',required=True);a=ap.parse_args()
    root=Path(a.root);dest=root/'delivery';dest.mkdir(exist_ok=False);selected=root/'A';package=selected/'package';video=root/'selected-A-videos'
    num=json.loads((package/'review.json').read_text('utf-8'));assert num['integer'] and num['independent_oracle_error']==0 and all(r['export_node_error']==0 for r in num['records'])
    videos=json.loads((video/'results.json').read_text('utf-8'));assert {r['id'] for r in videos['results']}=={'stairs','plaza','new_hallway'}
    assert videos['sha256']==num['package_sha256'] and not videos['postfilter'] and not videos['teacher_inference']
    for name in ('coefficients-padded.bin','load_words.hex','review.json'):shutil.copy2(package/name,dest/name)
    shutil.copy2(selected/'candidate.pt',dest/'selected-A.pt')
    assert file_hash(dest/'coefficients-padded.bin')==num['package_sha256'] and (dest/'coefficients-padded.bin').stat().st_size==20480
    words=(dest/'load_words.hex').read_text('ascii').splitlines();assert len(words)==2560 and b''.join(int(w,16).to_bytes(8,'little') for w in words)==(dest/'coefficients-padded.bin').read_bytes()
    edge=json.loads((root/'edge-audit/results.json').read_text('utf-8'))['summary'];narrow=json.loads((root/'narrow-audit/results.json').read_text('utf-8'))['summary'];motion=json.loads((root/'motion-audit/review.json').read_text('utf-8'))['summary']
    summary={}
    for mode in ('A','B','C'):
        key=root.name+'/'+mode;summary[mode]=dict(edges=edge[key],narrow=narrow[key],short_motion=motion[mode])
    declaration=dict(selected='A',selection_scope='current_and_three_requested_revisions_only',rationale='A preserves the most narrow structures, has near-best boundary width and the lowest sampled contour drift of A/B/C; B rounds and removes more structure, C has worse sampled motion.',
        core_bytes=17792,transport_bytes=20480,le64_words=2560,package_sha256=num['package_sha256'],contract_sha256=m.t.r.CONTRACT_SHA,
        unchanged_B1_graph=True,new_inference_operators=0,extra_output_filter=False,geometry='manually_constructed_coefficients',rgb_head='calibrated_on21_TRAIN_frames_from7_sources',
        not_end_to_end_teacher_distillation=True,actual_board=False,complete_teacher_match=False,known_limitations=['tree_sky_pale_bands_remain','some_narrow_structures_missing','some_corners_rounded'],comparison=summary)
    write_json(dest/'selection.json',declaration)
    data=m.load_data(a.data);by={r['id']:r for r in data['validation']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        paths=[('目标',Path(by[rid]['target'])),('修改前',Path(a.current)/(rid+'_output.png')),('选中A · 实际B1',selected/(rid+'_output.png'))]
        for j,(title,path) in enumerate(paths):board.paste(Image.open(path).crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(dest/'selected-comparison.png')
    rel=lambda p:Path(os.path.relpath(p,dest)).as_posix()
    scene_names={'stairs':'台阶与建筑','plaza':'广场与树木','new_hallway':'走廊'};sections=[];video_records=[]
    for scene in ('stairs','plaza','new_hallway'):
        row=next(r for r in videos['results'] if r['id']==scene);path=video/(scene+'_integer1080.mp4');assert file_hash(path)==row['video']['sha256']
        assert row['video']['fps']==30 and row['video']['size']==[1920,1080] and 10<=row['video']['seconds']<=20
        video_records.append(dict(scene=scene,path=str(path),seconds=row['video']['seconds'],sha256=file_hash(path),full_decode_frames=row['video']['frames']))
        sections.append(f'<section><h2>{scene_names[scene]} · {row["video"]["seconds"]:g}秒</h2><div class="pair"><figure><figcaption>原片</figcaption><video controls muted playsinline preload="metadata" src="{rel(row["source"])}"></video></figure><figure><figcaption>A · 真实B1系数回放</figcaption><video controls muted playsinline preload="metadata" poster="{rel(video/(scene+"_090_integer1080.png"))}" src="{rel(path)}"></video></figure></div><button onclick="playPair(this)">两边从头同步播放</button><a href="{rel(path)}">打开A版完整1080p视频</a></section>')
    reserved=json.loads((root/'reserved-A-review/review.json').read_text('utf-8'));extra=[]
    assert reserved['package_sha256']==num['package_sha256'] and reserved['not_blind']
    for row in reserved['records']:
        rid=row['id'];extra.append(f'<details><summary>{html.escape(rid)}</summary><img loading="lazy" src="{rel(root/"reserved-A-review"/(rid+"_comparison.png"))}"><a href="{rel(root/"reserved-A-review"/(rid+"_output.png"))}">打开实际720p输出</a></details>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B1 色块风格 · 选中A</title><style>body{max-width:1540px;margin:24px auto;padding:0 22px;background:#131b25;color:#e8edf4;font:17px/1.6 system-ui}h1{font-size:30px}h2{font-size:23px;margin-top:30px}p{max-width:1100px;color:#c4cedb}a{color:#87c9ff}button{background:#c8e2ff;color:#102235;border:0;border-radius:6px;padding:10px 16px;margin:12px 18px 12px 0;font:inherit;cursor:pointer}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;min-width:0}figcaption{padding:7px 0}video,img{width:100%;background:#090d13}details{padding:10px 0}summary{cursor:pointer}.notice{border-left:3px solid #e5ba77;padding-left:15px}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style><h1>B1 色块风格 · 三版中选 A</h1><p>本页视频由同一份20 KiB系数包重新加载运行。输入1080p → 720p B1处理 → 1080p输出，没有额外输出滤镜。当前为软件整数回放，未上板。</p><p class="notice">A在这次三版中更清楚、保留的细层更多。树冠浅色带、部分细线缺失和转角变圆仍有残留；这不是“已完全达到教师”的声明。</p>'''+''.join(sections)+'''<h2>原尺寸局部对比</h2><p>左：目标；中：修改前；右：本次选中A。点击图片看原尺寸。</p><a href="selected-comparison.png"><img src="selected-comparison.png"></a><details><summary>查看当前版与A/B/C全部对比</summary><a href="../comparison.png"><img src="../comparison.png"></a></details><h2>另两个历史来源 · 同一包复查10张</h2><p>这些来源曾用于历史重建实验，不能称为盲测；本轮没有用它们拟合颜色或再次调参。</p>'''+''.join(extra)+'''<h2>实际系数与检查记录</h2><p><a href="coefficients-padded.bin" download>系数包 BIN · 20 KiB</a>　<a href="load_words.hex" download>2560条 LE64 HEX</a>　<a href="selected-A.pt" download>对应模型状态</a>　<a href="selection.json">选择依据与包标识</a></p><p>区域处理系数采用人工构造；RGB输出头由21张训练图片校准。全部运算在原B1内执行，并非教师蒸馏已收敛的权重。</p><script>function playPair(b){const vs=[...b.previousElementSibling.querySelectorAll('video')];document.querySelectorAll('video').forEach(v=>v.pause());vs.forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});}</script></html>'''
    (dest/'watch.html').write_text(page,encoding='utf-8')
    text=f'''# B1三版选择：A

已完成当前版与A/B/C的同图、同相邻帧比较，选择A。这里交付当前最优结果，不宣称所有缺陷已消除或已上板。

- 主系数：coefficients-padded.bin，20480字节；有效17792字节；load_words.hex为2560个小端64位字。
- SHA-256：{num['package_sha256']}
- 合同：{m.t.r.CONTRACT_SHA}
- 原B1 8/16/32 R3、24卷积/29节点；没有硬件改动、额外推理算子或输出滤镜。
- 几何系数手工构造；RGB头只在7个TRAIN来源共21张图片上校准。不是端到端教师蒸馏训练收敛的声明。
- 三版六整图逐节点误差0、独立整数oracle误差0、HEX往返及长度/CRC/填充错误拒绝通过。
- 视频来自本包重载，1080整数area720 → B1 INT8/Q17 → 整数bilinear1080，30fps。台阶12秒、广场15秒、走廊12秒，全部解码完成。
- A直边宽中位2.563px（720域），修改前3.644；细结构对比低于目标一半的比例67.15%，修改前82.30%。数值仅覆盖所列样本，不代表全部细节恢复。
- 剩余：树冠浅色带、部分细线缺失、部分转角圆化，与目标仍有差距。未宣称完整教师匹配、硬件资源/时序新放行或上板实测。

打开watch.html可看全部视频/图片；本地HTTP服务根是runs、仅127.0.0.1:8883。视频使用相对引用，不重复复制大文件。

复现构造：python -B -m network_ab.training.run_b1_three_requested_revisions（完整参数保存在上级review.json）。
原始比较：../comparison.png；edge-audit、narrow-audit、motion-audit均在上一级。
历史reserved来源已用同一包复查10张，但因历史参与重建，不能称为盲测。
'''
    (dest/'README.md').write_text(text,encoding='utf-8');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'delivery.json',dict(selected='A',package_sha256=num['package_sha256'],videos=video_records,extra_sources_reviewed=len(reserved['records']),user_acceptance_pending=True,actual_board=False))
    print('THREE_REVISION_DELIVERY',dest,num['package_sha256'],flush=True)

if __name__=='__main__':main()
