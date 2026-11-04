"""Assemble honest two-route review, verify assets and protected packages."""
from core import *
import ast,html

def main():
    assert shutil.disk_usage(ROOT).free>20*2**30
    result=json.loads((OUT/'two-routes-video/summary.json').read_text('utf-8'))
    assert result['all_two_complete'] and len(result['clips'])==2
    for clip in result['clips']:
        assert all(n==clip['frames'] for n in clip['raw_unique_frames'].values())
        assert all(n==0 for n in clip['consecutive_duplicates'].values())
        assert all(v['frames']==clip['frames'] and v['max_pts_error_seconds']<.001 for v in clip['video'].values())
    protected=[]
    for folder in [WORK/'accepted/B1-A-20261008',WORK/'accepted/B1-水墨风-20261027']:
        acceptance=json.loads((folder/'acceptance.json').read_text('utf-8'))
        actual=sha(folder/'coefficients-padded.bin')
        expected=acceptance.get('package_sha256') or acceptance.get('sha256_padded') or acceptance.get('coefficient_sha256')
        # Use acceptance if keyed, otherwise the package hash manifest below.
        if expected:assert actual==expected,(folder,actual,expected)
        protected.append(dict(path=str(folder/'coefficients-padded.bin'),sha256=actual,acceptance_matched=bool(expected)))
    oldhash=json.loads((ROOT/'output/b1-ink-wash-20261026/audit.json').read_text('utf-8'))['deployment_sha256']
    assert sha(DEPLOY)==oldhash;protected.append(dict(path=str(DEPLOY),sha256=oldhash,known_earlier_hash_matched=True))
    images=[]
    for p in OUT.rglob('*'):
        if p.suffix.lower() in ['.png','.jpg']:
            with Image.open(p) as im:im.verify()
            images.append(str(p.relative_to(OUT)))
    scripts=list((OUT/'source').glob('*.py'))
    for p in scripts:ast.parse(p.read_text('utf-8-sig'),filename=str(p))
    params=sum(p.numel() for p in B1().parameters());assert params==10819
    cachefiles=list((OUT/'cache').rglob('*'));cachefiles=[p for p in cachefiles if p.is_file()];assert not cachefiles
    table=[]
    for c in result['clips']:
        stats=c['temporal_summary']
        table.append('|'+c['id']+'|'+str(c['frames'])+'|'+f"{c['duration_seconds']:.3f}"+'|'+
            '|'.join(f"{stats[k]['warp_mae']:.3f}" for k in ['source','v14','large_pixel','local_blocks'])+'|')
    report='''# 纯像素放大 / MC方块轮廓：两条独立路线

用户最新要求：纯像素学生放大像素块并微调亮度调色；MC另行探索整体方块轮廓、块面明暗，不匹配材质。两路不能混称成功。

## A：纯像素放大

- 首轮4×4纯像素学生已拟合并导出；250步选225步，验证RGB MAE0.985/255，独立720p图MAE约1.25–1.28。
- 6×6目标已生成，保持原6级/通道分档，饱和度1.03、gamma0.97为轻微调整。文件在flat-pixel-v2/targets。
- 400步放大试训选150步，验证MAE4.077；独立室外/室内MAE9.740/5.463。输出是学生整数近似，仍有4像素原生阶梯，**不能标为严格6×6学生**。
- grid_phase_audit.json给出边界内4像素平移测试：两个B1逐值等变；6×6教师出现MAE8.353、最大123的不同。固定图总stride4决定严格全局6格与之冲突；这不排除内容相关的大色块或接近6的视觉尺度，不能外推成所有像素风都不可行。

## B：MC方块轮廓

- references/mc-shape-concept.png是图像生成工具重绘的视觉目标，树和人物被重建为方块体。**不是学生输出，也不是严格配准的逐像素训练标签**。
- 已做三组各1000步的圆润外形到矩形/块面明暗构造试训。普通RGB误差偏爱保留圆形；增强角部40倍权重并引入门限/方向特征后仍有模糊和圆角，未通过整体方块几何目标。
- local-blocks为另一个可对齐的局部方块轮廓目标：按低分辨率颜色门限做方形闭运算，合并细碎外形，并加微弱顶/侧方向明暗。CPU形态学只生成训练标签。
- 同类局部运算已通过原有DW/PW/ReLU/饱和系数构造进入B1，再做250步QAT选50步；独立室外/室内目标MAE1.265/1.059。实际推理不调用形态学、渲染器、分割、光流或额外滤镜。
- 这个学生只达到**局部直角块面抽象**；不是图像生成目标里的全局语义3D方块重建，不能宣称真正MC已达标。

## 同源连续视频与时序

原图/旧V14/像素放大近似/局部方块候选来自同一独立原始帧，保持原PTS。两段都约3秒，每个新学生都有原生1280×720纯整数视频，另有四列对照。全部解码帧唯一，无跳帧或重复帧；软件视频帧率不等于板卡吞吐率。

|片段|帧数|秒|原图warpMAE|旧V14|像素放大近似|局部方块候选|
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

warp使用双向光流一致性/光度阈值屏蔽遮挡，仅供评价。屏幕固定网格的跳格也会进入该误差。静态区与高频相对误差见JSON，不靠色块简化后的低误差宣称稳定PASS。当前两路都未获视觉/时序验收。

## 工程与复现

- 固定B1：8/16/32通道，3残差块，24卷积/29节点，10819参数，1280×720；未改算子、拓扑、接口或硬件。
- 三个exports目录分别存4×4初版、6×6目标试拟合、MC局部方块候选。有效17792字节、补齐20480字节；各有BIN、HEX、checkpoint、残差字段、合同和CRC/SHA验证。
- 每个正式exports候选六组小图共68800标量与独立标准库参考逐节点一致；720p全部节点与QAT/解码整数一致。数值通过不等于风格通过，不等于实板放行。
- Python使用项目env。先pixel_targets.py、train_pixel.py和verify_export.py；放大目标/试训为make_large_pixel_targets.py、train_large_pixel.py；MC为probe_mc_shapes.py与mc_local_blocks.py；视频为render_two_routes.py。脚本参数/源数据划分见JSON。
- 原A、水墨、旧V14和部署路径只读；本轮无依赖安装、无板卡/Flash/RTL/部署修改。临时帧缓存验证编码后清理。

## 参考

- Minecraft官方方块风格及像素光影说明：https://www.minecraft.net/en-us/vibrant-visuals-update
- 官方版本说明：https://feedback.minecraft.net/hc/en-us/articles/37393460002957-Minecraft-Bedrock-Edition-1-21-90-Chase-the-Skies
- Pixelated Image Abstraction作者资料（背景研究，并非本轮完整复现）：https://cragl.cs.gmu.edu/pixelate/
'''
    (OUT/'两条路线说明.md').write_text(report,encoding='utf-8')
    cards=[]
    for s,label in [('outdoor','室外'),('indoor','室内')]:
        cards.append(f'<h2>{label}：同源约3秒对照</h2><p>原图 / 旧V14 / 像素放大近似 / 局部方块候选；后两者均为纯B1整数输出，尚未达最终目标。</p><video controls preload="metadata" src="two-routes-video/{s}_comparison.mp4"></video>')
        cards.append(f'<p><a href="two-routes-video/{s}_large_pixel.mp4">像素放大近似：原生720p</a> · <a href="two-routes-video/{s}_local_blocks.mp4">局部方块：原生720p</a></p>')
        cards.append(f'<h3>{label}：6×6目标与学生近似</h3><img src="flat-pixel-v2/{s}_comparison.jpg"><h3>{label}：MC局部方块轮廓目标与学生</h3><img src="mc-voxel/local-blocks/{s}_comparison.jpg">')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>纯像素与MC：双路线评审</title><style>body{background:#141922;color:#e6edf3;font:17px sans-serif;max-width:1600px;margin:24px auto;padding:0 18px}img,video{width:100%;height:auto}a{color:#8ed0ff}</style><h1>两条路线，分别验收</h1><p>A：纯像素放大；B：MC整体方块轮廓。严格6×6与真正MC都未宣称达标。</p>'+''.join(cards)+'<h2>MC视觉概念目标（AI重绘，非学生）</h2><img src="mc-voxel/references/mc-shape-concept.png"><p><a href="两条路线说明.md">完整结果、限制和复现说明</a></p>',encoding='utf-8')
    save_json(OUT/'review_checks.json',dict(syntax_checked=len(scripts),images_decoded=len(images),cache_files=0,
        parameters=params,protected=protected,disk=disk_guard(),all_videos_complete=True,
        visual_accepted=False,strict6_achieved=False,semantic3D_MC_achieved=False,board_tested=False))
    print('TWO_ROUTE_REVIEW_READY',flush=True)

if __name__=='__main__':main()
