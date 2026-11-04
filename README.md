# 基于易灵思的浪漫相机

基于 Efinix Titanium Ti60F225 与 Sapphire RISC-V 的实时风格化视频系统。当前归档为团队确认的 **production-r15-complete（2026-11-03）**：整机工程、C 固件、可读 CNN 源码、五套模型权重、运行数据、开发来源及 V9-P1 相机外壳。

本仓库用于 2026 年全国大学生嵌入式芯片与系统设计竞赛 FPGA 创新设计赛道，对应易灵思赛题一“AI 赋能的「实时绘画」——风格化视频渲染系统”。仓库已按团队要求公开，供评委直接浏览和下载；公开访问不等于授予统一开源许可证。

![V9-P1 正式打印件与装配分解](mechanical/enclosure_v9/03_三维核对/打印零件_装配分解核对.png)

*上图为正式打印网格的装配示意，屏幕和电子件不属于打印件；实物照片和演示视频尚未纳入此版本。*

## 当前版本

| 项目 | 内容 |
| --- | --- |
| FPGA 器件 | Efinix Titanium Ti60F225 |
| 代码版本 | `production-r15-complete`，来自 `sapphire-native8h-r15-complete-20261103.rar`，2026-11-04 更新入库 |
| 顶层模块 | `ti60f225_oob_top` |
| CPU | 官方 Sapphire `trim3_no_clint`，RV32I + Zicsr/Zifencei，100 MHz，8 KiB OCR |
| 硬件计算域 | 150 MHz；`S4R1_8_32_8_V1`，11 卷积、13 节点、1 残差块 |
| 模型规模 | 每风格 3808 权重 + 227 偏置 = 4035 参数，另 227 个 Q17 乘数 |
| 风格 | 漫画、阿凡达、热成像、像素动画、阴曹地府，共五套 checkpoint/整数权重 |
| 原图模式 | 第六槽位走 1080p bypass，装载占位数据不是第六套 CNN |
| 视频 | 风格计算 1280×720 RGB；HDMI 按源码时序为 1920×1080@30Hz |
| 控制 | C 固件负责相机配置、目录校验、装载调度、事件及模式切换；BLE/实体按键输入 |
| Efinity | `2026.1.132`；工程器件 Ti60F225、时序模型 I3 |
| 固件工具链 | RV32 GCC 15.2.0，已附对应预编译固件和四个 ROM lane 文件 |
| 外壳 | V9-P1，2026-10-28，毫米，1:1 |

HDMI 30Hz 是显示时序，**不代表 AI 每秒生成 30 个新画面**。本次没有重新测量实际采集帧率、AI 吞吐、切换延迟或综合时序；旧版约 17.1 AI FPS 不作为 r15 实测。

新包 **3240 个文件逐字节保留**于 `fpga/`，比上一代码包补齐五套模型、42 个可读核设计源、调度器、计算/DMA 描述符、生产运行数据和原始开发源码。原包 [README](fpga/README.md)、[网络结构导航](fpga/网络结构与源码导航.md)、[来源清单](fpga/SOURCE_ORIGINS.json)均原样保留。

**烧录边界：**已附的 `flash-user-data.bin` 仅是 Flash `0x340000` 起的数据区，不是整片 Flash 镜像，不能从地址 0 烧录。包内仍无最终 FPGA 位流；来源记录将该数据标为 `NOT_BOARD_VERIFIED`，应由硬件维护者与匹配位流、相机配置及回读验收一起确认，详见[工程说明](docs/fpga-project.md)。外壳仍是原 V9-P1 首件试装交付，验收边界见[赛题对照](docs/competition-checklist.md)。

## 从哪里开始

| 需要做的事 | 文档 |
| --- | --- |
| 下载完整文件、配置已有环境 | [环境与获取方式](docs/environment.md) |
| 查看五风格网络、参数与图像推理 | [网络与模型](docs/network.md) |
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
python -B fpga/tools/verify_complete.py
git lfs fsck
```

公开库浏览和克隆不需要协作者权限；命令中的 `git` 和 `python` 应替换为本人已确认的可执行文件路径。两级 Python 校验仅用 Python 3.10+ 标准库，无需 PyTorch/CUDA。原包的 `verify_package.py` 基础校验也继续保留。

大网表、STEP/STL/BREP 使用 Git LFS。r15 顶层网表约 174.8 MiB；没有拉取 LFS 时，它们可能只是几行指针文本。完整下载及磁盘预算见[环境说明](docs/environment.md)。

## 仓库结构

```text
.
├── fpga/
│   ├── project/                 Efinity XML、RTL、外设/SDC、CPU ROM
│   ├── firmware/                C/汇编/链接脚本及本版预编译文件
│   ├── network/
│   │   ├── model.py             包内独立 S4R1 网络类
│   │   ├── models/              五风格 checkpoint/整数权重，原图槽位占位
│   │   ├── rtl/                 可读共享 CNN 计算核
│   │   ├── programs/            计算/DMA 描述符与二进制
│   │   ├── scheduler/           调度与程序打包源码
│   │   └── runtime_data/        六槽目录和 Flash 数据区镜像
│   ├── original_sources/        原训练、导出、集成等开发源码，保留来源层次
│   ├── sapphire/                Sapphire 配置与 BSP
│   ├── tests/                   测试台、仿真模型及测试数据
│   ├── tools/                   完整性、模型/核检查、固件及运行数据构建入口
│   ├── verification/            来源编译/展开记录，不是本轮上板结果
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

共 **3274 个原始资产，323,843,953 字节，约 308.8 MiB**：最终包 3240 个、V9 外壳 34 个。不含个人环境、账号凭据、许可证或工具安装包。`original_sources/output/` 是队友交付的开发源码归档，不是本轮产生的缓存；其中历史绝对路径需自行适配，不应整目录自动执行。来源与哈希见[资产清单](manifests/source-assets.json)。历史版本可在 Git 提交记录查阅，不与当前工程混用。

## 维护

先阅读[贡献与更新约定](CONTRIBUTING.md)。本轮执行文件/ROM/模型转换/描述符及运行数据一致性检查，没有重新训练、安装工具、综合、HDL 仿真、烧录或实物测试。文件校验通过、来源历史验证与本轮实板验收必须分别表述。

版本变更记录见 [CHANGELOG](CHANGELOG.md)。本仓库未授予统一开源许可证，Efinix 原语和受保护 IP 的权利说明见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
