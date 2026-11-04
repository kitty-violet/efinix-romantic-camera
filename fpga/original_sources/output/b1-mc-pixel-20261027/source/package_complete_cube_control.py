"""Freeze the local complete-cube capacity evidence, separate from scene MC."""
from core import *
import ast

def main():
    setup(271165);d=OUT/'exports/mc-full-cube-capacity-r27';v=json.loads((d/'verification.json').read_text('utf-8'))
    assert v['crc32_padded']=='bc033008' and all(z['max_error']==0 for z in v['full720_qat_vs_decoded_integer_nodes'].values())
    proof=OUT/'r27-full-cube-planes';report=json.loads((proof/'report.json').read_text('utf-8'))
    assert report['positive_control_pass'] and report['face_plane_uniformity_pass'] and not report['complete_real_scene_MC_pass']
    source=d/'source';source.mkdir(exist_ok=True)
    for p in (OUT/'source').glob('*.py'):
        ast.parse(p.read_text('utf-8-sig'),filename=str(p));shutil.copy2(p,source/p.name)
    for p in proof.iterdir():
        if p.is_file() and p.suffix in ['.png','.json']:shutil.copy2(p,d/p.name)
    shutil.copy2(OUT/'mc-complete-cube-target-audit/audit.json',d/'old-target-and-RF-audit.json')
    (d/'README.md').write_text('''# 单个完整立方体：固定B1容量对照 R27

CRC bc033008；24卷积、29节点、10819参数、1280×720部署接口均未修改。普通卷积、ReLU和原上采样系数完成，无外置遮罩/滤镜/光流/历史帧。此包是构造系数，未训练。

范围：在规定的48像素正方形输入正面、24像素右上纵深条件下，以同一个二值正面抽取顶边和右边，再沿同一个方向展开，按正面优先和共同交界得到三个完整色面。判别输入仍是一个固定亮度阈值，不能推断语义物体或真实深度。

均匀正面、规定范围的正面渐变、规定浅色色块三组输出一致；外轮廓IoU0.9639、三个面内部RGB标准差均0。QAT/BIN全部720p节点一致，另有68800次独立标准库整数标量比较。此处斜边仍是原生4像素采样台阶。

对照R26发现，仅外轮廓IoU0.969仍会漏掉侧面部分被误当顶面的错误（侧面内部标准差24.43）；R26不通过完整立方体检查。本轮将面内部一致性加入硬考核，R27通过。

限制：光斑和硬阴影负例仍会被挤出；固定24像素纵深不等于根据物体尺寸形成统一大方块。真实场景的多个物体、遮挡、共享平面及合理的尺度选择未解决。严禁把此构造正例当作真实场景MC已成功。

当前室内外连续场景视频使用既有R25，见../../mc-latest-r25-review；它们不是R27视频。未修改任何硬件、Flash、A包、水墨包或现行部署基点。
''',encoding='utf-8')
    paths=[p for p in d.rglob('*') if p.is_file() and p.name!='manifest-sha256.json']
    save_json(d/'manifest-sha256.json',{str(p.relative_to(d)):sha(p) for p in paths})
    # Keep the current scene model and the newer bounded proof as different refs.
    progress=dict(scene_model='mc-geometry-only-r25',scene_crc='4c66dbd6',scene_video_dir=str(OUT/'mc-latest-r25-review'),
        bounded_capacity_model='mc-full-cube-capacity-r27',bounded_crc='bc033008',bounded_proof=str(d/'uniform_comparison.png'),
        complete_scene_MC_accepted=False,latest_user_correction='Target must be a complete cube, not a front plate with narrow extrusion.',
        next='Develop one shared object/plane assignment for the corrected complete-face representation; do not keep optimising support classification alone.')
    save_json(OUT/'mc-current-review-and-capacity.json',progress)
    # Verify every recorded new package and review hash.
    total=0
    for root in [d,OUT/'mc-latest-r25-review']:
        for name,h in json.loads((root/'manifest-sha256.json').read_text('utf-8')).items():assert sha(root/name)==h;total+=1
    print('CUBE_CAPACITY_PACKAGE_VERIFIED',total,'files',disk_guard(),flush=True)

if __name__=='__main__':main()
