# DS-ResStyle A/B结构规格

## 当前定义

- A版：`A_BASELINE_8_16_32_R3`
  - 1280×720输入/输出；
  - 8/16/32通道；
  - 3个深度可分离残差块；
  - 解码器为最近邻放大后执行DW3×3+PW1×1；
  - 最终普通3×3输出RGB。
- B版：`B_BALANCED_DECODER_8_16_32_R3`
  - 与A版共用完全相同的输入、编码器和3个残差块；
  - 解码器改为低分辨率DW3×3、低分辨率PW1×1降通道、最近邻放大、高分辨率DW3×3细化；
  - 最终PW1×1输出RGB。

## 理论预算

| 版本 | GMAC/帧 | 720p30 | 裸卷积权重 |
|---|---:|---:|---:|
| A | 1.3704192 | 41.112576 GMAC/s | 10,232字节 |
| B | 0.9667584 | 29.002752 GMAC/s | 10,256字节 |

B版相对A版减少29.46%卷积MAC。B版权重略多24字节，因为增加两个高分辨率DW细化核；其算力收益来自把主要通道混合移到低分辨率，而不是删掉权重。

## 公平比较规则

A/B训练必须共用：

- 完全相同的内容集、风格集以及训练/验证/测试划分；
- 完全相同的输入裁剪、缩放、颜色处理和数据增强；
- 完全相同的内容损失、风格损失和损失权重；
- 完全相同的优化器、学习率计划、批量大小、训练步数和随机种子；
- 完全相同的FP32、QAT、INT8校准和导出流程；
- 完全相同的固定图片与视频测试集。

唯一允许变化的是解码器与输出头的结构。首轮仍不加入新增细节一致性或循环一致性损失，避免无法判断差异来自结构还是损失。

正式评价至少包含：结构保持、轮廓、细纹理、材质、颜色稳定、最近邻块效应、视频闪烁、FP32/INT8差异、理论MAC、FPGA微基准有效吞吐和整帧预测时间。

## 使用

不需要第三方Python包：

```powershell
python E:\fpga\network_ab\model_ab_spec.py
python E:\fpga\network_ab\model_ab_spec.py --model B --json
python -m unittest discover -s E:\fpga\network_ab\tests -v
```

2026-09-29阶段2：当前B网络的完整数学定义已落入 `model_b_contract.json`，由 `model_contract.schema.json` 与stdlib校验器严格检查。`model_ab_spec.build_model_b()`从它生成旧接口兼容视图；现有调度、尺度审计、系数布局通过该接口取得相同形状/顺序，不再另写一套B拓扑。A候选仅保留历史比较规格。训练器、硬件匹配FakeQuant和真实权重导出器尚未实现，合成系数中的激活标志不可当成训练网络定义。

从当前硬件到FP32、QAT、部署和完整基线上板的启动条件，见`E:\fpga\output\toolchain\从当前硬件到训练部署与基线上板路线_2026-09-29.md`。旧工作流中的640×480、12/24/48方案不得用于当前720p、8/16/32候选。

## 阶段2：共用逐层定义与生成边界

唯一输入是 `model_b_contract.json`；含29个图节点：24个普通/DW/PW量化算子、3次残差相加、2次最近邻复制。完整720p仍为966,758,400 MAC、10,256个裸权重。图记录输入边、卷积核/stride/replicate、BN、训练bias、ReLU、量化点、融合组和DDR提交边界。每层HWC形状、OC8布局、有效lane/激活掩码和静态配置由校验器派生，不人工维护第二份层表。

本轮明确选择（等待FP32画质验证，不代表已有训练结果）：

- 除RGB头外23个卷积使用有可冻结running statistics的affine BatchNorm，训练卷积不另带冗余bias；部署前按eval统计折叠。普通层在BN后ReLU；每个残差块最后PW保持线性，残差相加后不加ReLU，以保留有符号恒等旁路。
- RGB头有训练bias、无BN/无ReLU；FP32输入/最终输出以scale=1/128对应有符号INT8。输出实数Clamp为`[-1,127/128]`；部署`q_out+128`得到RGB888。VGG输入精确为`(128*x+128)/255`，不能把这个域误按`(x+1)/2`转换。
- 权重per-OC、激活per-tensor、zero-point=0；bias先有符号33位相加并饱和INT32，再用signed18 Q17/固定shift17、half-away、INT8饱和和指定ReLU。不能静默把超范围倍率截成18位。
- 残差两支各自转共同INT32尺度后相加，再做一次输出RQ。实际scale均待校准/QAT；`alignment_encoding=null`，生产残差适配仍OPEN。通用整数参考用精确有理数验证这个公式，**不保证任意变倍率/移位已可由当前RTL实现**。
- 训练参数为11,379个：10,256卷积权重+1,120 BN仿射参数+RGB bias3；另有1,120个BN running-stat值。它们不是INT8部署镜像容量估计。

```powershell
# 仅标准库；生成配置，不产生权重或训练尺度
& E:\fpga\.venv-media\Scripts\python.exe -B E:\fpga\network_ab\model_contract.py `
  --output-dir E:\fpga\network_ab\generated\model_b_v1

# 重新计算并拒绝人工篡改/旧生成物
& E:\fpga\.venv-media\Scripts\python.exe -B E:\fpga\network_ab\model_contract.py `
  --output-dir E:\fpga\network_ab\generated\model_b_v1 --check

& E:\fpga\.venv-media\Scripts\python.exe -B -m unittest discover `
  -s E:\fpga\network_ab\tests -v
```

生成物包括 `training_structure.json`、`deployment_static.json`、`model_b_contract.h/.vh`、`layer_table.md`和源文件hash清单。静态层表不是已训练部署包：权重/偏置/倍率为空，残差valid=false，`complete_network_start_allowed=false`。硬件地址和生产调度仍须按实际实现验证，不能把静态宏直接当作已完成MMIO。

SV常量头已编译/断言通过。C头已由`validate_c_header.py`调用E盘现有GCC完成C11/Wall/Wextra/Werror无诊断编译，再实际执行24条表并与部署JSON逐字段对照；证据在`tmp/phase2-c-header-final-20260929`。这只证明主机C配置格式，不是Sapphire固件或MMIO已实现。C结构的`groups`字段表示OC8组数，对应JSON的`oc8_groups`，不是训练Conv2d的分组数。

```powershell
& E:\fpga\.venv-media\Scripts\python.exe -B E:\fpga\network_ab\validate_c_header.py `
  --compiler E:\fpga\tools\verilator\ucrt64\bin\gcc.exe `
  --output (Join-Path E:\fpga\tmp ('contract-c-' + (Get-Date -Format yyyyMMdd-HHmmss-fff)))
```

`contract_reference.py`为独立stdlib小张量整数参考，覆盖replicate/stride2、普通/DW/PW、NN、bias/Q17、不同尺度残差及完整图执行；没有提供完整参数和scale则拒绝执行。它不是GPU训练器或高性能720p整数引擎。`contract_torch.py`提供从同一合同构建的FP32源，只有调用 `build_fp32_model()`时才import torch；当前只完成源码与结构检查，**尚未执行PyTorch前向/反向、训练或真实QAT**。QAT需以冻结BN统计并按部署折叠后的权重语义做FakeQuant，不能在导出时才首次改变其数值分布。

旧 `model_ab_spec.py` 原件保存在 `historical/model_ab_spec_pre_contract_20260929.py`，SHA256仍为 `BE78BEA79A645696038D5F685BEE92793566022527D9A666E91AAE41F93E3202`；回归逐字段确认新旧B兼容视图完全一致，历史物理/数字证据不被新hash覆盖。`coefficient_layout.py`的按索引奇偶ReLU、极值bias和负倍率仍是历史诊断刺激；没有改它或旧17792B合成镜像来冒充生产参数。

手势/恶灵/显示仍属完整基线，本机器定义只描述主风格网络。旧手势24通道需要完整的32通道零填充适配或新增24/G3模式，GAP/小FC与共享时隙另验；本定义不宣称其已实现。

## 固定Q17尺度审计

`q17_scale_audit.py`用于检查QAT导出的逐层/逐输出通道重标定系数能否由当前固定Q17硬件表示。输入JSON必须提供每层`input_scale`、逐输出通道`weight_scales`和`output_scale`，并显式证明输入、权重、输出zero-point全为0；如果同时提供校准集上的含bias INT32累加样本，还会逐样本比较理想实数scale和固定Q17整数链的INT8输出。对正式A/B模型，审计器还会核对完整算子顺序和每层输出通道数，缺层、乱序或少通道都会拒绝执行。

```powershell
python E:\fpga\network_ab\q17_scale_audit.py `
  E:\fpga\network_ab\q17_scale_audit_example.json `
  --json-output E:\fpga\output\toolchain\q17_scale_audit_example_result.json `
  --fail-on-unfit
```

示例文件只用于验证审计器，明确不是训练结果。审计器的`hardware_range_pass`只说明乘数范围可表示；即使校准样本零差异，`quality_gate`仍固定为`UNMEASURED`，必须另外完成固定图片/视频上的FP32与INT8画质验收，才能决定Q17是否可部署。

## 隔离训练环境

2026-09-29用户安排：训练环境暂缓，等待用户指定安装位置；当前不安装、不下载、不迁移或删除现有环境。已有`E:\fpga\.venv-train`只是历史目录，不代表已指定今后安装位置，也不代表PyTorch已就绪。环境、wheel缓存、TEMP/TMP、PyTorch/VGG缓存、数据和训练输出均须提前按用户指定的目录配置，不能因虚拟环境在E盘就默认缓存不会落到C盘。

`setup_train_env_verified.ps1`是历史安装入口，当前不应执行。它从PyTorch官方索引下载并校验wheel；今后使用前须核验当时的版本兼容性，并改为显式接收上述路径。以下仅保留历史命令，不构成当前安装授权：

```powershell
# 只下载并校验，不安装
& E:\fpga\network_ab\setup_train_env_verified.ps1 -DownloadOnly

# 已校验wheel后安装，并实际执行CUDA张量运算
& E:\fpga\network_ab\setup_train_env_verified.ps1
```

当前环境状态以`E:\fpga\output\toolchain\QAT训练环境检查.md`为准；截至2026-09-28，torch尚未安装，不能运行上述训练或QAT流程。
