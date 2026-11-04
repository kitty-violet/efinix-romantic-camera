"""Write the final measured reconstruction report, without claims of optimality."""
import json
from pathlib import Path
from . import research_b1_reconstruction as s
from .runtime import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    root=s.OUT; final=read(root/'final_results.json'); verified=read(root/'final_verification.json')
    old=final['metrics']['old_integer']; new=final['metrics']['new_integer']
    reduction=100*(1-new['validation']['mean_mse']/old['validation']['mean_mse'])
    lines=['# 固定B1原图恢复：实际研究结果','',
        '本轮只做输入原图的重建诊断，不是动画风格成品。硬件合同、采样结构、算子、通道数和20 KiB传输包保持不变。结果来自真实包的软件整数离线回放，尚未上板。','',
        f'旧版不是这套结构已测到的最好水平。最终实际包在6张已查看验证图上的像素均方误差减少 **{reduction:.1f}%**。下面同时列出来源隔离的预留测试，避免只看调参图。细草叶、小人脸、密集栏杆仍有损失，不能保证无损，也不能把有限实验称作数学最高。','',
        '## 直接查看','',
        '- [图片和同步视频入口](watch.html)：原片、上一版和本轮实际包；人物15秒、台阶12秒。',
        '- [720p原尺寸裁图](final-review/comparison.png)：原图／上一版实际INT8／本轮实际INT8。',
        '- [1080p外部缩放与网络分开对照](final-review/resize-versus-network.png)。',
        '- [预留来源各自PSNR最低帧对照](final-review/test-worst-per-source.png)：白天与夜间，两张均已目视检查。',
        '- [人物15秒](videos/plaza_integer1080.mp4)、[台阶12秒](videos/stairs_integer1080.mp4)。',
        '- [真实权重包](delivery/coefficients-padded.bin)、[64bit小端装载hex](delivery/load_words.hex)。','',
        '## 实测质量','',
        '本表以网络输入的720p原图为基准，比对同尺寸输出；后文1080p视频帧的数值另算，不能混用。PSNR由整组像素MSE合并计算；SSIM是RGB、7×7均匀窗的平均值。梯度误差越低越好。这些数值用于交叉核对，不能替代看图。','',
        '| 模型 | 验证PSNR dB | 验证SSIM | 验证梯度MAE | 预留测试PSNR dB | 预留测试SSIM |','|---|---:|---:|---:|---:|---:|']
    names={'old_fp32':'上一版FP32','old_integer':'上一版实际INT8包','new_fp32':'本轮最高保真FP32（单独保留）','new_integer':'本轮选定实际INT8包'}
    for label,item in final['metrics'].items():
        a,b=item['validation'],item['reserved_test']
        lines.append(f'| {names[label]} | {a["pooled_psnr"]:.4f} | {a["mean_ssim"]:.5f} | {a["mean_edge"]:.4f} | {b["pooled_psnr"]:.4f} | {b["mean_ssim"]:.5f} |')
    lines+=['','FP32最高保真断点和整数包属于不同后续分支；不能把两者差值全部解释为对同一权重的一次量化损失。精确父链见 selection.json 与各实验 config.json。','',
        '训练使用原来7个视频的190帧，目标就是输入自身。6张验证图来自独立来源，但历史已经查看，不叫盲测。最终预留测试来自070032、192624两个原test来源，各5帧；在本轮模型冻结后才统一评估，未用于权重更新或选择。watch.html列出全部10帧，不按输出好坏筛选。','',
        '## 做过的训练与对照','',
        f'本研究共实际执行 **{verified["total_executed_updates"]:,} 次正式参数更新**，不包含此前旧版的10,350步、预检步和逐图诊断。下表更新数包含两次磁盘保护后从断点重放的步骤；这些重放不重复算入模型继承步数。','',
        '| 实验目录 | 实际更新次数 | 分支结束步 | 最佳PSNR dB | 最佳PSNR步 | 停止原因 |','|---|---:|---:|---:|---:|---|']
    for t in verified['trials']:
        lines.append(f'| {t["name"]} | {t["executed_updates"]} | {t["final_step"]} | {t["best_psnr"]:.4f} | {t["best_psnr_step"]} | {t["stop"]} |')
    lines+=['','不同损失分支还按结构综合分保存best.pt，所以上表PSNR最高断点不一定等于交付断点。原始逐帧值、停止检查与断点在各目录history.json、completion.json、config.json中。','',
        '观察与解释：','',
        '1. 旧权重只加入频率损失并续训5000步，约28.91 dB，几乎没提升。仅换损失不够。',
        '2. 独立的正负颜色／方向初始化、纹理采样、相位辅助训练使恢复明显改善。相位辅助头只在训练使用，导出时没有它。',
        '3. 同初值无辅助头9000步也达到31.08 dB；旧初值加辅助头达到30.20 dB。因此不能把改善全部归于某一种损失或辅助头，初始化和训练路径都有关。',
        '4. RGB通道置换时输入和目标一起变换，修正中途蓝伞偏灰的问题；然后去掉辅助头继续保真训练。',
        '5. 原图VGG特征、优化潜变量作为训练监督、低学习率长尾均实际尝试。后两阶段的收益已降到约0.07、0.06 dB；VGG分支没有超过最高保真分支。',
        '6. 量化范围校准不是万能：早期权重均衡有帮助，后期直接套用反而更差。测量后做了逐层偏差校正、固定倍率QAT、结构损失QAT及已有Q17倍率的联合优化。',
        '7. 另外测试训练专用平移一致性，约束输入移动1／2像素后，输出对应位置不要额外改变细节。保留原图目标与结构损失；没有在推理端混帧或平滑。',
        '8. 没有外部锐化、生成式补纹理、人脸修复、分割或高清旁路。原图恢复的真实性优先于增加看似锐利的假细节。','',
        '## 采样限制和没有夸大的结论','',
        '输入输出仍是720p。180是网络最深处特征图的高度，不是把全部网络输入改成180p。网络内部两次stride2、两次最近邻上采样未改。',
        '下采样前的可学习卷积确实能把部分细节编码进特征通道，不能简单断言“下采样后必然只能糊”。周期6像素的灰度细条纹测试，旧FP32原图MAE约39，新FP32约6；改善是对应原图的恢复，不只是边缘能量变大。',
        '但不能保证任意原图无损恢复。构造的2／4像素周期纹理经过两次stride2后，内部特征在远离边界处成为空间常量；后续最近邻与共享卷积无法从该常量恢复原本的周期位置变化。本轮测试也未恢复这类反例。这不等于所有小于4像素的自然细节均不可恢复。',
        '另对两张训练纹理crop分别做2500步潜变量优化和2500步单图权重拟合；潜变量优化改善更明显，但密集细节仍有损失。它使用已知原图逐图优化，只是定位问题的诊断，不是可部署通用网络，也不证明全局最优。',
        '自然场景的最高可恢复程度仍取决于数据、权重和整数误差。当前只能说这是这一批实测路线得到的最好交付候选，不能声称已穷尽所有算法或所有权重。','',
        '## 时序连续性、编码和外部缩放','',
        '| 视频 | 时长/帧数 | 原片运动补偿MAE | 输出运动补偿MAE | 编码RGB MAE范围 |','|---|---|---:|---:|---:|']
    for row in verified['video_rows']:
        lines.append(f'| {row["scene"]} | {row["seconds"]}秒 / {row["frames"]}帧 | {row["motion_source"]:.4f} | {row["motion_output"]:.4f} | {min(row["encoding_mae"]):.3f}–{max(row["encoding_mae"]):.3f} |')
    lines+=['','光流估计的运动补偿误差仅用于诊断，不构成“完全不闪烁”的证明。视频为1920×1080、30fps播放，不能据此推断FPGA实测帧率。编码前90／91／92及末帧PNG已保存；这些PNG仍存在的软化不能归于码率。','',
        '| 第90帧 | 仅1080→720→1080 PSNR | 经过本轮B1后 PSNR |','|---|---:|---:|']
    for row in verified['resize_diagnostic']:
        lines.append(f'| {row["scene"]} | {row["resize_only"]["psnr"]:.4f} | {row["actual_network"]["psnr"]:.4f} |')
    lines+=['','## 交付与验证','',
        f'- 包：17,792有效字节，20,480传输字节；SHA-256 `{final["package_sha256"]}`。',
        f'- 合同SHA-256 `{final["contract_sha256"]}`。',
        '- 每条正式训练路线均通过实际720p GPU前向／反向、小批次、新实例读盘恢复后下一步比对，恢复误差为0。',
        '- 交付真实字节重载，6张720图全29节点对齐误差0；独立12×12整数参考误差0。',
        '- 截断包、CRC破坏、非零padding拒绝；hex按2560个64bit小端字回读一致。',
        '- 两段视频完整解码、帧数／分辨率／FPS／源与输出哈希检查通过；第90帧重新跑实际包与编码前PNG逐像素一致。',
        '- 没有修改硬件或上板；资源fit、布局布线和严格时序不在此软件结果中冒充通过。',
        '- 没有安装或下载依赖；仅清理自己pip/wheels缓存583,090,332字节。两次磁盘保护中断及恢复均保留日志。','',
        '## 研究资料与复现','',
        '[研究文献与采用边界](research-plan.md)记录CAR、IRN、Focal Frequency Loss、恢复损失、感知损失、量化均衡／偏差校正和LSQ的原始论文及作者资料。每种训练的完整参数与冻结代码见对应目录config.json和research_b1_reconstruction.py；恢复源码另存resumed_source文件。','',
        '入口代码在 `E:/fpga/network_ab/training/`：`research_b1_reconstruction.py`（训练）、`audit_b1_reconstruction_research.py`（采样/相位/图像检查）、`publish_b1_reconstruction_research.py`（真实包与最终测试）、`render_b1_region_trial.py`（整数视频）、`verify_b1_reconstruction_research.py`（交付绑定）。使用项目隔离Python并加 `-B`；不要原地覆盖已完成实验目录。','',
        '[全部质量数值](final_results.json) · [交付核验与实验更新次数](final_verification.json) · [选定父链](selection.json)']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('RECON_RESEARCH_REPORT_WRITTEN',flush=True)


if __name__=='__main__':main()
