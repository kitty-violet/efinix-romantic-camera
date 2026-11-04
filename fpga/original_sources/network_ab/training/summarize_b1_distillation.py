"""Publish verified B1 trial results without converting proxy metrics to acceptance."""
import json
from pathlib import Path
from .runtime import configure_workspace


def main():
    root=configure_workspace();run=root/'runs/b1-distillation-20261003'
    v=json.loads((run/'verification.json').read_text());f=json.loads((run/'final-2000/verification.json').read_text())
    assert v['status']=='NUMERICAL_AND_DELIVERY_CHECKS_PASS_VISUAL_QUALITY_SEPARATE'
    curve=[]
    for p in sorted((run/'train').glob('validation_*/report.json')):
        curve.append((int(p.parent.name.split('_')[-1]),json.loads(p.read_text())['mean_teacher_mae_rgb255']))
    video_rows='\n'.join(f"|{r['id']}|{r['seconds']:.0f}s / {r['frames']}帧|{r['teacher_mae']:.3f}|{r['student_temporal_p95']:.3f}|{r['teacher_temporal_p95']:.3f}|" for r in v['videos'])
    detail_rows='\n'.join(f"|{r['id']}|{r['teacher_edge_mean']:.3f}|{r['b1_edge_mean']:.3f}|" for r in v['still_detail'])
    minimum_psnr=min(r['rgb_psnr_db'] for r in v['encoding_samples'])
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())/2**20
    text=f'''# B1 2000步 FP32 蒸馏：已完成的效果实验

本轮实际训练并导出了板子对应的小网络，完成走廊12秒、台阶12秒、广场人物15秒的连续视频。结论是：**本轮动画风格蒸馏未达标。用户观看后指出主要是画质模糊，没有动画风格；无损图对照支持这一判断。** 部分色彩变化不能算完成风格迁移。它不证明B1永远不能实现，也不是INT8量化损失——当前全程FP32。

查看页：[原片 / 教师 / B1 同时间切换](http://127.0.0.1:8883/b1-distillation-20261003/index.html)。直接文件见本目录 `videos-2000`，每个场景均有独立教师和B1的1920×1080 MP4。教师和学生使用相同板式缩放；B1没有叠加教师、原图、白斑滤波、锐化或时间融合。

## 实际执行

- 固定当前合同：8/16/32通道、3残差块、24卷积、10,256卷积权重；没有改硬件网络或其他工作流。
- 原始视频重新解码：7个训练源、95相邻帧对、190张原生720p输入与目标。目标来自完整720p的已认可240步教师，再做白斑修正。旧版目标未复用。
- 400×400原生尺度配对裁剪，40px边界不计损失，2对4帧/普通步；每20步使用完整720p两帧。固定BN运行统计、训练全部卷积及BN仿射；EMA用于最终推理。
- 配对/梯度/高频/VGG目标/弱时序损失；无GAN、材质分类或人工贴脸。初始化来自已有RGB通路，仍是同一个合同网络。
- GPU FP32前向、反向、小批次和恢复预检通过；恢复重放误差 `{v['resume_error']:.12g}`。正式500步退出后在新进程恢复，日志1–2000连续，95组都实际采样，100次完整720p更新。
- 优化器更新合计 `{v['optimizer_update_seconds']:.2f}` 秒，峰值分配显存 `{v['peak_allocated_mib']:.1f}` MiB。不是端到端准备/渲染耗时，不是板上FPS。
- 三个评估源均不在训练源中，但以前已查看过，属于源级隔离回归，不能称盲测泛化验收。

## 静帧结果和不能回避的问题

6个完整720p回归样例的教师平均RGB绝对误差从 `{curve[0][1]:.3f}` 降至 `{curve[-1][1]:.3f}`（灰度级）。仅像素距离下降不能证明纹理或游戏感充分。后半程改善很小，不能只靠不断加步数宣称在解决问题。

台阶、植被与人物细节在编码前的无损720p图中已经偏软。以下是同帧平均相邻像素梯度幅度，作为细节损失的辅助证据，不能单独作审美分数：

|样例|教师|B1|
|---|---:|---:|
{detail_rows}

另外，受控平移同一输入图1–3像素后，B1对齐输出仍平均相差2.06–2.92灰度级，P95约9.94–11.90；移动4像素则误差0。这检出了两次stride2带来的取样相位敏感。它是需要针对性训练/检验的因素，但没有单凭这个测试把所有视频抽动归因于网络，或判定现有结构永远不行。原始数据见 `translation_probe.json`。

补充检查见 `softness-diagnosis/report.json`：六张初始输出均逐像素等于原图每4像素取样后复制放大；RGB通路初始化不是无损恒等映射。训练后六张图的平均梯度幅度均进一步下降。第2000步原始权重与EMA的输出平均差仅0.431–0.732灰度级，梯度幅度几乎一致，因此权重平均不是主要模糊来源；1:1裁剪图没有做锐化。

这些检查仍没有分离初始化、损失配置、数据覆盖和网络表达能力各自的影响。下一项应在保持合同的前提下，针对同一清晰原生尺度样本分别做原图重建和配对动画目标拟合，先验证清晰结构能否被拟合，再做独立来源视频；平移相位也必须纳入。不能因图更平滑就宣布稳定性成功，也不能把弱风格直接拿去量化。

## 视频检查

所有视频都逐帧运行网络，没有复制帧或插帧延长时长；B1累计1170帧、教师1170帧。链路为1080→整数面积720→网络→整数双线性1080，H264 CRF16。30fps为播放帧率，不是实时推理成绩。

|场景|时长/帧数|B1-教师平均RGB差|B1时序残差P95|教师时序残差P95|
|---|---:|---:|---:|---:|
{video_rows}

时序项使用相同源光流与遮挡/重建误差掩码，每3帧取一次相邻帧对，比较风格相对源图的变化。**模糊或风格变弱也会降低这个指标，不能据此宣布无抽动。** 应与完整视频及无损细节一起看。

{len(v['encoding_samples'])}个编码前后样本最低RGB PSNR `{minimum_psnr:.2f}` dB；编码会损失信息，但上述偏软在编码前已经存在，不能归咎于码率。完整帧数、尺寸、时长、hash见 `videos-2000/report.json`。

## 真实权重与复现证据

- 文件：`final-2000/b1_style_fp32.pt`，`{v['weights_bytes']}`字节。
- SHA256：`{v['weights_sha256']}`。
- 6张完整720p图独立导出重载输出差0；BN折叠最大实数差 `{max(r['fold_max'] for r in f['records']):.12g}`。
- 完整续训断点：`train/checkpoint_002000.pt`；配置：`config.json`；冻结主要训练源码：`train/source`。
- 推理用 `network_ab.contract_torch.build_fp32_model()` 和权重的 `model` 字段；始终 `.eval()`。输入/输出及板式缩放说明见 `final-2000/README.md` 和 `inference.json`。
- 当前只有FP32实验参数，未完成QAT、INT8生产导出、硬件逐值对拍或上板。保留旧教师与原4850方向。

总核验在 `verification.json`，本轮目录约 `{size:.1f}` MiB。没有新依赖下载、没有修改硬件文件，也未覆盖其他工作流的状态。
'''
    (run/'RESULTS.md').write_text(text,encoding='utf-8')
    print('RESULTS_WRITTEN',run/'RESULTS.md')


if __name__=='__main__':main()
