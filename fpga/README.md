# Sapphire + native8h r15 完整代码

## 先看网络结构

- **`网络结构与源码导航.md`**：逐层尺寸、运算、残差连接和 FPGA 模块对应关系。
- **`network/model.py`**：可直接读取本包五套 `selected.pt` 的 PyTorch 网络类 `S4R1`。
- **`network/graph.json`**：13 个节点的结构定义；各风格完整结构及量化参数同时保留在 `network/models/slot*/graph.json`。
- **`network/rtl/cnn_production_frame.sv`**：可读 FPGA 计算核入口，下面的 MAC、DW/PW、缓存、DMA 和重定量模块均随包提供。

## 工程目录

| 目录 | 内容 |
|---|---|
| `project/` | 第十五版实际整机工程、展平顶层、Sapphire 核、接口 RTL、ROM、引脚和时序约束 |
| `firmware/` | 生产 C 程序、相机表、启动汇编、链接脚本和匹配的预编译固件 |
| `network/models/` | 五风格 checkpoint、整数权重、BIN/HEX、逐层结构、测试向量；slot5 为原图模式装载 ABI 占位 |
| `network/rtl/` | 共享 CNN 计算核的完整可读模块及实际选定映射输入 |
| `network/programs/` | 计算/DMA 描述符、字段定义、存储分配和已生成二进制 |
| `network/scheduler/` | 调度与描述符编译源码 |
| `network/runtime_data/` | 匹配六槽目录、计算程序、系数和 RGB 表的运行数据 |
| `original_sources/` | 网络定义、训练、导出、像素修正及集成工具的原始开发源码，保留目录层次 |
| `sapphire/` | 官方 CPU 配置和 BSP |
| `tests/` | 控制、CPU、相机、DMA 测试及对应模型/数据 |
| `tools/` | 包完整性、模型数值、源码依赖检查与固件构建入口 |
| `source_binding/` | r15 模式缓冲修改与原计算核对应的来源记录 |

## 使用

Efinity 2026.1.132 打开 `project/ti60f225_oob.xml`；器件 Ti60F225/I3，顶层 `ti60f225_oob_top`，CPU 100 MHz，计算域 150 MHz。实际整机入口是 `project/`；`network/rtl/` 供分层阅读和单独计算核开发，已在整机展平文件中实现。

使用 Python 3.10+ 检查所有文件、项目输入、逐层图、权重转换、描述符和运行数据：

```powershell
python tools/verify_complete.py
```

在已有 PyTorch/NumPy/Pillow 环境中打开网络和检查五套 checkpoint：

```powershell
python network/model.py --slot 0
python tools/check_network_model.py
```

模型输入为 NCHW 的 `RGB_uint8 / 128 - 1`，输出乘 128 加 128 后恢复 RGB。图像推理入口为 `network/infer_package.py`，支持 `--style-root network/models/slot0 --input image.png --output result.png --device cpu`，输入为 1280×720 RGB。

C 固件重新编译入口：`python tools/build_firmware.py --toolchain E:/path/to/gcc`，使用 RV32 GCC 15.2.0。生成的四个 ROM 文件放回 `project/`。原始训练脚本的素材和历史输出路径保留在 `original_sources/`，当前模型的独立加载与推理使用上述包内入口。

模式顺序为漫画、阿凡达、热成像、像素动画、阴曹地府、原图。文件清单见 `MANIFEST.json`，来源对应见 `SOURCE_ORIGINS.json`，本次检查结果见 `检查结果.json`。

运行数据重建：`python tools/build_runtime.py`，从包内五套系数和描述符字段重建六槽运行数据，并逐字节对照交付文件。调度算法入口为 `network/scheduler/compile_schedule.py`，当前选定程序的打包入口为 `network/scheduler/pack_program.py`；早期实验和迁移脚本保留于 `original_sources/`。

可读计算核检查：`python tools/check_core.py --questa E:/path/to/questa/win64 --license E:/path/to/license.dat`。所需六个官方 Verilog 原语模型已在 `tests/vendor/`。`network/rtl/fixed_video_compute.sv` 仅将 `start` 的声明提前以兼容 Questa；原文件与修改记录均随包保存。

新说明文件为 UTF-8；整机工程保留原始字节，`project/i2c_master_bit_ctrl.v` 的中文注释使用原 GB18030 编码。
