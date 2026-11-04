"""Write a bounded engineering result, preserving failures and the V1 fallback."""
import json
from pathlib import Path
from .audit_b1_sampling_v2 import OUT,FALLBACK
from .runtime import file_hash,write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    final=OUT/'temporal-margin';package=final/'package';videos=final/'videos'
    verification=read(package/'verification.json');comparison=read(videos/'comparison.json')
    assert len(comparison['records'])==2 and comparison['package_sha256']==verification['sha256']==file_hash(package/'coefficients-padded.bin')
    assert verification['learned_steps']==11700 and verification['convolutions']==24 and verification['nodes']==29
    assert verification['export_all_node_error']==verification['independent_oracle_error']==0
    for r in read(FALLBACK/'manifest.json')['files']:
        assert file_hash(FALLBACK/r['path'])==r['sha256']
    thin=read(final/'sampling-check/thin_structures.json')['results']
    rows=[]
    for width in (1,2,3,4,6,8):
        ranges=[]
        for model in thin:
            r=[v for v in model['records'] if v['width_720']==width]
            ranges.append(f"{100*min(v['minimum'] for v in r):.1f}%–{100*max(v['maximum'] for v in r):.1f}%")
        rows.append(f'| {width} | {ranges[0]} | {ranges[1]} |')
    motion=[]
    for r in comparison['records']:
        v1,v2=r['motion']['v1'],r['motion']['v2']
        motion.append(f"| {r['scene']} | {v1['mean']:.3f} → {v2['mean']:.3f} | {100*v1['over16']:.3f}% → {100*v2['over16']:.3f}% | {v1['p95']:.3f} → {v2['p95']:.3f} |")
    report=f'''# B1 采样与颜色层级修正：实际结果及边界

## 本轮判定

V1按用户要求封存为第一版兜底，认可的是视频连续性，不是整体画质。新版增加了亮度层级，并修复部分色偏和内部数值截平；人物轮廓和“包浆”尚未解决，不替换V1为已验收版本。

最终候选：`{package}`，SHA256 `{verification['sha256']}`。
推理只有原B1网络和既定1080→整数area720→网络→整数bilinear1080。没有额外滤镜、教师、人物分割或运行时光流。

## 已做的修改

1. 原六个二态亮度特征改为六个三态特征，明暗代表值由7层扩展为13层；另外两个色度特征仍独立学习，不代表整个RGB图只有13种颜色。
2. 教师标签的7×7多数改为3×3；末端DW卷积从冻结改为可学习，量化监督使用实际量化后的核。
3. 训练覆盖两级stride2的全部裁剪相位，并直接监督真实360p格点；压缩中段隐藏数值尺度，在后层补偿，降低提前饱和。
4. 加强色度约束；最后1200步加入块内分类阈值余量、真实相邻帧概率一致性和实际RGB的一致性。光流仅用于训练和离线诊断。

本轮新增3600 FP32 + 2800 QAT = 6400步；包括父模型共6600 FP32 + 5100 QAT = 11700步。使用原42训练帧、21真实相邻帧对；回归图曾反复检查，不能称盲测。

## 对采样的实测结论

V1合成阶跃在第二次降采样后平均10%–90%过渡宽约5.87个720p像素，最终约3.11像素。格点修正版最终约3.21像素，不能据此宣称边缘明显变锐。此前上采样的对称核确实可能引入格点偏移，但V1已训练权重的平均最终偏移约−0.18像素，不能把简单插值模型的偏移直接套到它身上。

给最后360→720解码段注入正确目标时，人物和建筑外形能够保留；实际网络输入该段的预测已有偏差。这是定位用的oracle旁路，不是可部署图，也不能证明完整180p瓶颈保留了全部信息。诊断图与真实小网图严格分开。

自然图中，格点修正版的up2_low_dw饱和比例由约43%降到14%–19%；色度Co平均偏差由约+2.7至+3.3降到+0.48至+0.99，Cg由约−1降至−0.42至+0.02。数值范围修复确实生效，但这些指标不等于画质通过。

以下测试是最终权重对灰色高对比细线的对比度保留，范围涵盖四种相位与两个方向。它依赖本测试颜色和权重，不是通用分辨率上限。

| 输入细线宽度（720p像素） | V1对比度保留 | 最终候选对比度保留 |
|---|---|---|
'''+ '\n'.join(rows)+'''

## 视频复核

两段均为实际字节重载后逐帧生成：台阶360帧/12秒，人物450帧/15秒，1920×1080、30fps。源视频SHA256与V1相同。完整解码、帧数和编码前后误差已核对。

| 场景 | 运动对齐平均误差：V1→候选 | 超过16级变化比例 | P95 |
|---|---|---|---|
'''+ '\n'.join(motion)+f'''

这些是诊断值，不是主观无抽动的证明；不能只挑一个更好的数字放行。请结合`{videos/'compare_v1_v2.html'}`和两段原尺寸视频审查。

## 验证和未完成事项

- FP32/QAT均实际通过720p GPU前向、反向、小批次和新实例磁盘恢复。最终时序续训恢复最大差3.73e-9；CUDA光流反向有非逐位确定性警告，未声称所有恢复严格0。
- 24卷积/29节点不变；17792B有效、20480B传输；全节点字节复演和独立整数oracle误差0，CRC/坏包拒绝和2560个小端64位装载字回读通过。
- 未修改硬件、共享网络合同或硬件工作流。没有本轮RTL/P&R/时序/上板PASS；既有ModelSim启动WinError623保留，未重复启动或修系统。
- 残差削弱对照出现明显色偏；末端半插值/最近邻/轻锐化未修好人物，不作为最终候选。所有干预均在B1系数内执行，未冒充重新训练。
- 剩余重点仍是进入末端前的区域形状预测，以及小于几像素结构随采样相位的损失。不能用升码率、盲目锐化、只增色阶或降低总损失替代这项工作。

## 资料依据

- 实际模型及逐层核：本目录`sampling_audit.json`、`lattice-recovery/sampling-check/sampling_audit.json`。
- 正确目标旁路：`reconstruction_oracles.png`及`lattice-recovery/reconstruction-check`，仅诊断。
- [Making Convolutional Networks Shift-Invariant Again（作者项目）](https://richzhang.github.io/antialiased-cnns/)与[官方实现](https://github.com/adobe/antialiased-cnns)：用于研究降采样、混叠和移动一致性；没有将其额外BlurPool算子直接塞入硬件图。
- [PyTorch interpolate官方文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.interpolate.html)：用于核对像素格点约定。实际视频缩放仍以项目整数参考及其逐点测试为准。
'''
    (OUT/'engineering_review.md').write_text(report,encoding='utf-8')
    files=[package/'coefficients-padded.bin',package/'coefficients.bin',package/'load_words.hex',package/'verification.json',
           package/'v1_v2_teacher.png',package/'trees_people_v1_v2.png',final/'checkpoint_001200.pt',
           videos/'stairs_integer1080.mp4',videos/'plaza_integer1080.mp4',videos/'comparison.json']
    write_json(OUT/'delivery_manifest.json',dict(status='PARTIAL_IMPROVEMENT_NOT_FINAL_ACCEPTANCE',
        fallback_preserved=str(FALLBACK),candidate=str(package),files=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in files],
        remaining=['soft/incorrect small contours','person appearance still smeared','no actual board validation'],
        graph_unchanged=True,external_postprocessing=False,training_complete=True,visual_accepted=False))
    print('FINAL_REVIEW_WRITTEN',str(OUT/'engineering_review.md'),flush=True)


if __name__=='__main__':main()
