# AI 模型训练工作区

> **2026-10-08 当前结果已确定：用户接受 B1 的画风修订 A。**
> 唯一当前交付入口为 `E:\fpga\model-training-workspace\accepted\B1-A-20261008\README.md`，
> 部署提示词为同目录 `发给FPGA部署对话的提示词.md`。
> A 是效果修订名，硬件拓扑仍为 B1。旧实验的大型产物已经按用户要求清理，
> 下方早期流程和历史报告不代表当前交付路径；现用代码、环境、A复现数据与定稿保留。

本目录由 `model-training` 对话独占。共享 `model_b_contract.json`、`contract_torch.py`、
整数参考、硬件和其他工作流只读；唯一网络为 B / 720p / 8-16-32 / R3。

用户于 2026-09-29 指定根目录 `E:\fpga\model-training-workspace`：

| 子目录 | 用途 |
|---|---|
| `env` | 隔离 Python 3.10 环境 |
| `cache/pip`、`cache/wheels` | 包管理缓存、已验证的官方安装包 |
| `cache/torch` | 官方 VGG19 权重 |
| `cache/cuda`、`tmp` | CUDA 缓存、临时文件 |
| `data/incoming/content` | 用户内容素材上传，保留原始文件 |
| `data/incoming/holdout` | 可选的独立验收素材上传，禁止用于训练 |
| `data/processed` | 720p 母版、过滤记录和数据划分清单 |
| `runs` | 权重、检查点、日志、验证图片和视频 |

所有命令从 `E:\fpga` 运行，使用 `-B` 防止在共享源码旁生成 pycache。
每个命令自行设置指定根目录内的 TEMP/TMP、pip、Torch 和 CUDA 缓存。

```powershell
# 已获指定目录内安装授权；官方索引 SHA256 校验，绝不绕过完整性错误。
& 'C:\Program Files\Python310\python.exe' -B -m network_ab.training.setup_environment `
  --root 'E:\fpga\model-training-workspace'

$trainingPython = 'E:\fpga\model-training-workspace\env\Scripts\python.exe'
& $trainingPython -B -m network_ab.training.preflight
& $trainingPython -B -m network_ab.training.prepare_data
& $trainingPython -B -m network_ab.training.verify_gpu `
  --output 'E:\fpga\model-training-workspace\runs\gpu-validation-UNIQUE-NAME'
```

GPU 检查实际执行合同模型的 29 节点、720p 前向、VGG 损失反向、有效 batch4、
Adam、检查点保存及下一步恢复比较，并报告速度、显存和警告。
其内容是明确标记的程序生成诊断图案，checkpoint 只能是 `diagnostic_only`；
正式训练器拒绝将其作为训练权重恢复。它不代表任何风格画质结论。
CUDA replicate-padding 反向可能有原子加法差异；恢复误差实测报告，
容差为绝对 1e-6 / 相对 1e-5，不虚称所有 GPU 反向逐位确定。

## 数据到位后的流程

1. `prepare_data` 清点上传文件，保存完整源 SHA256。按拍摄场次/源视频分组，
   每个源只能属于 train/val/test 之一。没有足够独立素材时不随机拆相邻帧凑数。
2. 在工作目录制作 `model-training-source-plan-v1`、`status=reviewed` 的源计划：
   每源提供 path/sha256/group/split，视频另提供 sample_fps；结合素材审计固定模糊、
   曝光、dHash 和颜色均值去重阈值。执行 `prepare_data --plan <源计划> --name <版本>`。
3. 预处理保留原文件、先留出验证/测试、输出等比例裁切的完整 720p 母版与 draft 清单。
   检查图片审查表和场景覆盖后再把清单标记 approved。风格集不作内容集。
4. 用 `verify_real_data` 做实际内容的短轮训练与保存恢复检查，再开始正式 FP32。
   `--stop-after` 仅限制本次执行长度，不改变余弦学习率总计划。

```powershell
& $trainingPython -B -m network_ab.training.verify_real_data `
  --manifest '<审核后的manifest.json>' --gpu-report '<同版本GPU检查result.json>' `
  --output 'E:\fpga\model-training-workspace\runs\real-content-UNIQUE-NAME'

& $trainingPython -B -m network_ab.training.train `
  --manifest '<审核后的manifest.json>' --gpu-report '<同版本GPU检查result.json>' `
  --run 'E:\fpga\model-training-workspace\runs\fp32-001' --stop-after 8

& $trainingPython -B -m network_ab.training.train `
  --manifest '<同一manifest.json>' --gpu-report '<同一GPU检查result.json>' `
  --run 'E:\fpga\model-training-workspace\runs\fp32-001' `
  --resume 'E:\fpga\model-training-workspace\runs\fp32-001\checkpoint_000008.pt'
```

FP32 默认主裁块 640×360 占 90%、960×540 占 10%，每 1000 步使用完整 720p；
microbatch1、累积4次。GPU 检查必须证明全部尺寸可执行，不会静默缩小部署分辨率。
这组显存参数在实际验证后冻结。loss 精确遵循 Q038-Q040：VGG19 relu2_2 内容，
relu1_2/2_2/3_3/4_3 等权 Gram，Gram 除以 C×H×W，仅首次按输出梯度约 2:1 校准。
校准必须使用训练态的实际首个有效 batch4，合并四个微批次输出梯度的平方范数；
校准结束还原全部模型/BN、采样位置及 Python/CPU/CUDA RNG。不能用未训练 BN 的
推理态单图来代替。`verify_real_data` 实测检查此还原及实际断点恢复。
不使用 AMP/TF32，不加 TV、时间一致性或第二套风格。

每次启动、恢复、训练步骤及保存均校验源码/合同指纹；变化时停止，由用户协调。
完整检查点包含 BN、Adam、scheduler、Python/Torch/CUDA RNG、采样位置和增强 RNG、
配置及数据/代码/合同/VGG 哈希。中途中断不保存半步采样状态；从最近完整检查点恢复。

验证输出是完整720p并排图片/视频、内容/Gram、饱和率和边缘能量诊断。
边缘能量不等于结构正确，验证损失也不是综合画质总分；LPIPS、光流视频稳定性、
严重伪影清单和用户盲评仍需在真实素材到位后完成，不自动批准 QAT。
当前不实现量化导出；不同尺度残差生产编码仍属于共享硬件合同待解决项。

依赖来源：PyTorch 官方 CUDA 12.8 wheel 索引及模型站；其余 Python 包使用 PyPI。
固定主依赖：torch2.11.0、torchvision0.26.0、numpy2.2.6、Pillow12.3.0、
imageio-ffmpeg0.6.0（打包 ffmpeg）、opencv-python-headless4.12.0.88（后续图像审计/光流）。
完整传递依赖版本和来源/hash 在 `runs/environment/pip-install-report.json` 与 freeze 文件。

测试：`python -B -m unittest discover -s network_ab/training/tests -v`。
当前真实进度及准确停点以 `PROJECT_MEMORY/streams/model-training/STATE.md` 为准。

素材审查可使用 `review_uploads --input <incoming内目录> --output <全新runs目录>`。
脚本全解码并记录实际 PTS，生成约1秒间隔预览；需要时使用 `--start/--end/--interval`
细审局部。预处理同样按实际 PTS 取帧，未经单独审查的非16:9源会拒绝直接裁切。
筛选和风格数据划分不等同于手势类别真值，原视频始终保留。

## 2026-09-30 画质退化修正

两轮默认训练1000步均有灰蓝退化。有界2×2固定样本对照记录于
`E:\fpga\model-training-workspace\runs\画质退化诊断与修正_20260930.md`。
`fp32_recovery_config.json`仅新增RGB通路参数初始化和固定BN运行统计策略；
BN仿射参数与卷积继续训练，同源拓扑/损失/学习率计划/数据划分保持一致。
所有正式入口和两项GPU验证入口都应显式使用这一配置，不能混用旧检查报告。

```powershell
$recoveryConfig = 'E:\fpga\network_ab\training\fp32_recovery_config.json'
& $trainingPython -B -m network_ab.training.verify_gpu --config $recoveryConfig `
  --output 'E:\fpga\model-training-workspace\runs\gpu-recovery-UNIQUE'
& $trainingPython -B -m network_ab.training.verify_real_data --config $recoveryConfig `
  --manifest 'E:\fpga\model-training-workspace\data\processed\content-20260929-v1\manifest.approved.json' `
  --gpu-report 'E:\fpga\model-training-workspace\runs\gpu-recovery-UNIQUE\result.json' `
  --output 'E:\fpga\model-training-workspace\runs\real-recovery-UNIQUE'
& $trainingPython -B -m network_ab.training.train --config $recoveryConfig `
  --manifest 'E:\fpga\model-training-workspace\data\processed\content-20260929-v1\manifest.approved.json' `
  --gpu-report 'E:\fpga\model-training-workspace\runs\gpu-recovery-UNIQUE\result.json' `
  --run 'E:\fpga\model-training-workspace\runs\fp32-recovery-UNIQUE' --stop-after 1000
```

初始化输出不是训练成果；`diagnose_quality`生成的像素/内容诊断权重禁止作为正式权重。
最终画质仍须检查独立完整720p图片视频；修复训练策略不等同于允许量化。

Windows观察程序可能短暂占用JSON导致原子替换失败：写入对5/32/33错误有界重试，
非关键progress占用不会中止训练，核心检查点仍严格检查；此外每250步保存完整恢复快照。
这不改变优化器/学习率/数据/损失。历史run源码指纹不会静默放宽；仅有明确I/O修复且
数值函数AST/所有其他源码哈希一致时，另建run并保存完整状态迁移记录、重新GPU及
真实恢复验证、对照重做步骤的数值与采样。当前实例见runs/fp32-content-20260930-03/migration.json。
