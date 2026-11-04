"""Package the shortened-transition video against the immediately preceding one."""
import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from . import inspect_coarse_people_video as base
from .publish_person_blocks import OUT,SOFT,SOFT_HASH
from .runtime import file_hash,write_json


def main():
    assert file_hash(SOFT)==SOFT_HASH
    base.OUT=OUT;base.OLD=SOFT;base.main()
    soft=json.loads((SOFT.parent/'video_verification.json').read_text(encoding='utf-8'))
    current=json.loads((OUT/'video_verification.json').read_text(encoding='utf-8'))
    decode=json.loads((OUT/'decode_verification.json').read_text(encoding='utf-8'))
    # Same source footage, frame range and automatic mask; this is the right
    # immediate-previous reference for the user's transition feedback.
    comparison=dict(previous_soft=soft['person_flow_diagnostic']['current'],current_blocks=current['person_flow_diagnostic']['current'],
        note='Colour separation itself changes residual magnitude; this is not a visual stability acceptance test.')
    write_json(OUT/'soft_vs_blocks_motion.json',comparison)
    cases=[(210,(195,645,660,1045)),(330,(965,420,1430,820)),(330,(0,660,465,1060))]
    board=Image.new('RGB',(930,1308),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    bindings=[]
    for row,(frame,box) in enumerate(cases):
        for col,(folder,title) in enumerate([(SOFT.parent,'刚才：过渡较柔'),(OUT,'本次：缩短过渡，明确色块')]):
            path=folder/f'frame_{frame:03d}_1080p.png';bindings.append(dict(path=str(path),sha256=file_hash(path)))
            draw.text((col*465+8,row*436+4),title,font=font,fill='white');board.paste(Image.open(path).crop(box),(col*465,row*436+36))
    board.save(OUT/'people_comparison.png');write_json(OUT/'comparison_sources.json',bindings)
    html=(OUT/'preview.html').read_text(encoding='utf-8')
    html=html.replace('人物粗化 · 15秒对照','人物色块 · 缩短颜色过渡').replace('人物减少细碎刻画，衣服归并成更大的色块','缩短人物的颜色过渡，让明暗形成更明确的色块')
    html=html.replace('本次人物粗化','本次明确色块').replace('上一版对照','刚才柔和版')
    html=html.replace('衣物减碎纹；脸部使用较轻的简化，保留主要结构。','衣物采用更少、更明确的颜色分级；脸部保留稍细一级的分级，减少五官被吞没。')
    html=html.replace('上一版与本次人物粗化的实际同帧裁剪','刚才柔和版与本次缩短颜色过渡的实际同帧裁剪')
    (OUT/'preview.html').write_text(html,encoding='utf-8')
    prev,new=comparison['previous_soft'],comparison['current_blocks']
    report=f'''# 人物色块：降低颜色过渡

用户否定前一版人物色彩过渡过柔，要求降低过渡。本次在同一实际学生输出上调整人物颜色分级，旧柔和版保留作同帧对照。

## 实际修改

- 身体保持原TV强度0.10/100迭代；明暗步长15、分级斜率13、混合0.98；色度步长10、斜率10、混合0.92。大幅缩窄中间过渡，形成更明确的颜色平台。
- 肤色范围采用固定较细一级：明暗步长8、斜率9、混合0.84，色度步长6、斜率8、混合0.75；TV最低0.014。肤色提示是启发式色彩范围，不等于可靠人脸语义分割。
- 分级前3×3中值和Lab双边d9/range8/spatial3去除小色斑，肤色保留50%原颜色引导，以免小五官被吞没。没有人物坐标变形，没有原片RGB粘贴回人物。
- 自动人物分割、掩膜时域处理、原始视频、学生权重均与前版相同。无新训练、无教师推理、无RGB跨帧混合。

## 视频与核验

15秒、450帧、30fps、1920×1080、H264 CRF16、无音频。人物仍经SDR1080→整数area720→真实FP32学生→人物处理→整数bilinear1080。背景复用最初block-separation视频的1080p解码结果，合成再编码；不是全链新渲染。

30帧近景短检查通过，完整视频及独立检查各解码450帧，相邻重复0。编码前后5帧MAE范围{min(r['codec_mae_255'] for r in decode['selected_codec_checks']):.3f}–{max(r['codec_mae_255'] for r in decode['selected_codec_checks']):.3f}/255。

相同源光流及人物区域诊断，刚才柔和版mean={prev['mean']:.5f}、帧内P95平均={prev['p95']:.5f}；本版mean={new['mean']:.5f}、帧内P95平均={new['p95']:.5f}。更强颜色分界会改变差异幅度，本指标不等于无闪烁或稳定验收。video_verification.json内previous仍指最初block-separation参考；与刚才柔和版的准确对照是soft_vs_blocks_motion.json。

当前只是软件效果预览，新增人物分割、颜色处理尚未融入模型或FPGA，未验证定点、资源和时序。实际权重不变。源码、参数和输入哈希均已保留。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    shutil.copy2(__file__,OUT/'source_code'/('extra_'+Path(__file__).name))
    files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    write_json(OUT/'manifest.json',dict(files=files,total_bytes=sum(r['bytes'] for r in files)))
    print('PERSON_BLOCKS_READY',comparison,flush=True)


if __name__=='__main__':main()
