# 基于易灵思的浪漫相机

基于 Efinix Titanium Ti60F225 与 Sapphire RISC-V 的视频图像处理作品。当前代码为 **production-r15（2026-11-03）**，包含 Efinity 工程、C 固件、CPU ROM、六模式显示与控制、测试源码，以及 V9-P1 相机外壳。

本仓库用于 2026 年全国大学生嵌入式芯片与系统设计竞赛 FPGA 创新设计赛道的项目归档与团队交接，对应易灵思赛题一“AI 赋能的「实时绘画」——风格化视频渲染系统”。当前仓库为私有仓库。

![V9-P1 正式打印件与装配分解](mechanical/enclosure_v9/03_三维核对/打印零件_装配分解核对.png)

*上图为正式打印网格的装配示意，屏幕和电子件不属于打印件；实物照片和演示视频尚未纳入此版本。*

## 当前版本

| 项目 | 内容 |
| --- | --- |
| FPGA 器件 | Efinix Titanium Ti60F225 |
| 代码版本 | `production-r15`，来自 `sapphire-native8h-r15-code-20261103.zip` |
| 顶层模块 | `ti60f225_oob_top` |
| CPU | 官方 Sapphire `trim3_no_clint`，RV32I + Zicsr/Zifencei，100 MHz，8 KiB OCR |
| 硬件计算域 | 150 MHz；风格处理及原始分辨率路径见工程模块 |
| 模式 | 原包说明为六模式，包含中文名称、FPS、全/半屏切换 |
| 控制 | C 固件负责相机配置、目录校验、装载调度、事件及模式切换；BLE/实体按键输入 |
| Efinity | `2026.1.132`；工程器件 Ti60F225、时序模型 I3 |
| 固件工具链 | RV32 GCC 15.2.0，已附对应预编译固件和四个 ROM lane 文件 |
| 外壳 | V9-P1，2026-10-28，毫米，1:1 |

视频输出时序与 AI 推理帧率是两种指标。旧版约 17.1 AI FPS 的记录不作为 r15 的实测值；新包没有附本版实测帧率、切换延迟或综合时序结果，也不能凭六个显示模式推定有六套独立网络权重。

本次将新包 **89 个文件原样收录**于 `fpga/`，已具备项目 XML、外设配置、引脚/时序资料、RTL、C 固件和测试源码。[原包 README](fpga/README.md)和[工程使用说明](docs/fpga-project.md)给出构建入口。包内未附最终 FPGA 位流及外部 Flash 目录/模型/ISP 数据镜像，故不能仅用 CPU `control.bin` 完成整机烧录。本次没有重新综合、仿真或实板测试。外壳仍是原 V9-P1 首件试装交付，完整边界见[赛题对照](docs/competition-checklist.md)。

## 从哪里开始

| 需要做的事 | 文档 |
| --- | --- |
| 下载完整文件、配置已有环境 | [环境与获取方式](docs/environment.md) |
| 使用已有已烧录设备 | [使用说明](docs/usage.md) |
| 打开 FPGA 工程、构建固件 | [工程使用说明](docs/fpga-project.md)、[原包说明](fpga/README.md) |
| 实现手机/串口控制 | [蓝牙命令协议](docs/ble-protocol.md) |
| 打印和装配外壳 | [V9-P1 外壳入口](mechanical/enclosure_v9/README.md) |
| 准备比赛提交、核对缺项 | [赛题对照](docs/competition-checklist.md)、[交接与提交](docs/submission.md) |
| 核验文件是否完整 | [文件完整性说明](docs/verification.md) |
| 了解代码和 IP 使用范围 | [第三方与权利说明](THIRD_PARTY_NOTICES.md) |

## 下载与校验

完整下载需要 Git 和 Git LFS。先在自己的电脑上确认已有工具可用，再执行：

```powershell
git clone https://github.com/kitty-violet/efinix-romantic-camera.git
cd efinix-romantic-camera
git lfs pull
python -B scripts/verify_repository.py
python -B fpga/tools/verify_package.py
```

私有仓库需要 GitHub 账号已获访问权限；命令中的 `git` 和 `python` 应替换为本人已确认的可执行文件路径。校验脚本仅使用 Python 3.10+ 标准库，无需 PyTorch、CUDA 或训练环境。

大网表、STEP/STL/BREP 使用 Git LFS。r15 顶层网表约 174.8 MiB；没有拉取 LFS 时，它们可能只是几行指针文本。完整下载及磁盘预算见[环境说明](docs/environment.md)。

## 仓库结构

```text
.
├── fpga/
│   ├── project/                 Efinity XML、RTL、外设/SDC、CPU ROM
│   ├── firmware/                C/汇编/链接脚本及本版预编译文件
│   ├── sapphire/                Sapphire 配置与 BSP
│   ├── tests/                   测试台、仿真模型及测试数据
│   ├── tools/                   固件构建与原包校验脚本
│   └── MANIFEST.json            原包 SHA-256 清单，原样保留
├── mechanical/enclosure_v9/
│   ├── 01_正式打印零件/         六种正式零件，STEP/STL 各一份
│   ├── 02_先打螺纹试配件/       T1/T2 试配环
│   ├── 03_三维核对/             正式件分解图、历史 V9 确认图
│   ├── 04_核对与源文件/         CAD 生成源码、参数、BREP、检查记录
│   └── 整机装配_V9_P1.stp       装配参考，不作为整体打印件
├── docs/                        环境、使用、协议、赛题及提交文档
├── manifests/source-assets.json 原始资产字节数和 SHA-256
├── scripts/verify_repository.py 文件、LFS 指针和文档链接检查
└── .github/workflows/           自动执行仓库完整性检查
```

当前收录 123 个原始资产：r15 代码包 89 个、V9 外壳 34 个，共约 271.5 MiB。未打包个人环境、账号凭据、工具安装包或历史训练实验。文件来源与哈希见[资产清单](manifests/source-assets.json)。11 月 2 日的五风格无 RISC-V 版本保留在 Git 历史提交 `423d07a`，不与新工程混用。

## 维护

先阅读[贡献与更新约定](CONTRIBUTING.md)。每次替换网表、CAD 或参数文件时，保持版本说明、校验清单和对应验证证据一致；不要将归档校验通过解释成时序收敛或实板验收通过。

版本变更记录见 [CHANGELOG](CHANGELOG.md)。本仓库未授予统一开源许可证，Efinix 原语和受保护 IP 的权利说明见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
