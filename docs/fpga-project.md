# r15 完整工程与固件

版本 `production-r15-complete`，2026-11-03，2026-11-04 入库。来源为团队提供的 `sapphire-native8h-r15-complete-20261103.rar`。包内 3240 个文件原样保留于 [fpga](../fpga/)，包括原 README、MANIFEST、模型和来源记录；没有修改 HDL、C 固件、权重或受保护 IP。与上一代码包相比，原工程/固件文件未变，主要补齐可读计算核、模型、调度器及运行数据。

## 工程入口

使用已有 **Efinity 2026.1.132** 打开 [ti60f225_oob.xml](../fpga/project/ti60f225_oob.xml)，顶层为 `ti60f225_oob_top`，器件 Ti60F225，时序模型 I3。工程的设计源、SDC 和外设 XML 引用均为包内相对路径，归档时已检查其存在性。

保留整个 `project/` 结构，尤其是四个 CPU ROM lane 文件和 `outflow/`。后者包含本包提供的引脚、外设和物理约束资料，不能当成普通缓存省略。编译时建议使用数据盘上的英文短路径，在工作副本中操作。

| 文件/目录 | 用途 |
| --- | --- |
| [native222.v](../fpga/project/native222.v) | r15 大网表与顶层集成，Git LFS |
| [EfxSapphireSoc.v](../fpga/project/EfxSapphireSoc.v)、[sapphire_candidate.sv](../fpga/project/sapphire_candidate.sv) | 综合用 Sapphire 核和 APB 封装 |
| [rv32_control_system.sv](../fpga/project/rv32_control_system.sv) | CPU、控制跨域桥、服务后端、SPI 和相机连接 |
| [control_rpc_bridge.sv](../fpga/project/control_rpc_bridge.sv)、[rv32_service_backend.sv](../fpga/project/rv32_service_backend.sv) | 原子请求跨域、寄存器/事件/事务后端 |
| [service_word_dma.sv](../fpga/project/service_word_dma.sv) | 批量传输、CRC 与系数服务 |
| [camera_service_bridge.sv](../fpga/project/camera_service_bridge.sv)、[camera_i2c_transaction.sv](../fpga/project/camera_i2c_transaction.sv) | 相机事务跨域和 I2C |
| [frame_pair_owner4.sv](../fpga/project/frame_pair_owner4.sv)、[native_capture_select.sv](../fpga/project/native_capture_select.sv)、[native_output_bypass.sv](../fpga/project/native_output_bypass.sv) | 帧所有权、原始/风格路径切换 |
| [legacy222_names.sv](../fpga/project/legacy222_names.sv)、[five_style_name_osd.sv](../fpga/project/five_style_name_osd.sv) | 六槽位中文名称与帧边界锁存 |
| [firmware/main.c](../fpga/firmware/main.c) | 相机初始化、目录/CRC 检查、装载、模式切换及中断调度 |
| [firmware/control.h](../fpga/firmware/control.h) | MMIO 寄存器和状态/事件定义 |

CPU 为官方 Sapphire `trim3_no_clint`，RV32I + Zicsr/Zifencei、100 MHz，片上 OCR 8 KiB；视频计算域 150 MHz。固件与硬件职责应以源码为准，CPU 不承担逐像素卷积运算。

## 可读 CNN 与整机的关系

[network/rtl/cnn_production_frame.xml](../fpga/network/rtl/cnn_production_frame.xml)给出 42 个可读核设计源，供分层阅读或单独核开发；它不是包含相机、HDMI 和 CPU 的整机工程。核已在整机展平文件中集成，不应把两套核重复加入同一工程。

网络为五套同拓扑的 S4R1，输入/输出为 1280×720 RGB；显示前放大到 1080p，原图模式走 native1080 bypass。HDMI 总时序 2200×1125、像素时钟 74.25 MHz，对应 1080p30；这不等于 CNN 吞吐 30fps，前端可重复显示最近完成的帧。网络、量化和推理入口见[模型说明](network.md)。

来源 [source_binding](../fpga/source_binding/)记录计算核与 r15 模式控制修改的对应关系。`network/rtl/fixed_video_compute.sv` 在原包中仅提前 `start` 声明以兼容 Questa，原文件及调整前后哈希均保留；本轮没有另改。

## 固件构建

可直接使用 [firmware/prebuilt](../fpga/firmware/prebuilt/) 中本版 CPU 固件和四个 ROM 文件，不需要先安装新工具。要修改 C 固件时，先核对自己已有的 **RV32 GCC 15.2.0**，原脚本是 Windows 入口，调用 `riscv-none-elf-gcc.exe` 和 `objcopy.exe`。

```powershell
# 当前目录为仓库根，路径替换为本人已有工具。
& 'E:\your-python\python.exe' -B .\fpga\tools\build_firmware.py --toolchain 'E:\your-rv32-gcc'
```

`--toolchain` 指向含 `bin/` 和 `lib/gcc/riscv-none-elf/15.2.0/rv32i/ilp32/libgcc.a` 的工具链根目录。编译采用 `-march=rv32i_zicsr_zifencei -mabi=ilp32`，不能随意换为 RV32IMAC 固件。

输出位于 `fpga/build/firmware/`，包含 ELF、BIN、map 和四个 ROM lane。确认结果后，在独立工程副本中将四个 `EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol*.bin` 一起复制到 `project/`，再构建 FPGA 工程。不要只更新一个 lane 或混用不同版本。若正式纳入修改，应同时更新预编译交付、两个哈希清单和版本说明。

[memory.ld](../fpga/firmware/memory.ld)规定 OCR 从 `0xf9000000` 开始、大小 8 KiB，并检查至少 768 字节栈余量。当前 `control.bin` 为 6168 字节；ROM 是将它补零至 8192 字节后，按四个字节 lane 拆分得到的二进制文本，**不是 FPGA 位流**。

## 运行数据与生成配置

`main.c` 从 SPI Flash `0x340000` 读取目录，校验后调度程序/系数/ISP 数据的装载。完整包已附 [flash-user-data.bin](../fpga/network/runtime_data/flash-user-data.bin)及其[清单](../fpga/network/runtime_data/manifest.json)：

| 项目 | 内容 |
| --- | --- |
| 内容 | 六槽目录、计算/DMA 程序、五风格驻留系数/原图占位、768 字节 RGB/ISP 表 |
| 文件长度 | 2,797,568 字节 |
| Flash 起始地址 | `0x340000` |
| 结束地址（不含） | `0x5EB000` |
| SHA-256 | `b29b4fc95a590cb51572ea087f280ebe148071997039da6fa39451dabfd74016` |
| 来源验收状态 | `DATA_ONLY_CAMERA44_V2_PREPARED_NOT_BOARD_VERIFIED`，`hardware_access: false` |

这是**数据区镜像**，不是完整 8MB Flash 文件、FPGA 配置位流或 CPU ROM。**不能从 Flash 地址 0 写入，也不能只用它完成整机烧录。**包内仍无最终 FPGA `.bit/.hex/.lbf` 位流。

[来源 result.json](../fpga/network/runtime_data/result.json)还要求在匹配的当前配置位流中，用官方 BRAM updater 落实两个相机寄存器字节，并完成最终镜像回读及有人监督的视频验收。由硬件维护者核对位流、相机表、数据区与烧录配置后再操作；本轮不执行烧录，不把来源待验状态写成已上板通过。`tests/data/` 的 HEX 仅供仿真，不能替代生产数据。

标准库可从包内描述符字段和系数重建数据，并逐字节对照交付文件。在仓库根目录运行，输出放个人工作目录：

```powershell
python -B fpga/tools/build_runtime.py --output D:/your-work/r15-runtime
```

该命令会创建输出文件，不改包内交付。当前选定计算/DMA 程序的打包源码在 [pack_program.py](../fpga/network/scheduler/pack_program.py)，调度算法在 [compile_schedule.py](../fpga/network/scheduler/compile_schedule.py)。保留的旧实验脚本并非本版整机一键构建流程。

[soc_config](../fpga/sapphire/soc_config)中的 `--ramHex` 保留了作者的 Efinity 安装绝对路径，指向厂商 bootloader。这是核生成来源记录；若重新生成 Sapphire，需改为本人同版合法安装中的路径，并重新落实本版固件 ROM，不能让默认 bootloader 替换 r15 控制程序。预编译 map/lst 的作者路径属于来源记录，不是要求复刻的电脑目录。

## 仿真与验收

测试源见 [tests](../fpga/tests/)。仿真用 `tests/vendor/EfxSapphireSoc.v`，综合用 `project/EfxSapphireSoc.v`，不要把两者同时加入同一编译清单。生产固件测试运行目录需要四个 ROM lane 及 `tests/data/` 三个 HEX。

控制测试未附统一运行入口；部分测试还依赖原 runner 的实现/连接检查，例如 snapshot 测试引用的 `zeroaddr_snapshot20`。保留这些测试源用于恢复测试，但不将源码中的 PASS 字符串当成已执行结果。

新增 [tools/check_core.py](../fpga/tools/check_core.py)可使用本人已有合法 Questa 及许可证，对可读核编译/展开；六个厂商仿真模型在 `tests/vendor/`。来源记录位于 [verification/core](../fpga/verification/core/)，编译/展开成功不等于功能仿真或综合时序通过。本轮未重新执行该工具。

本次只重新执行完整性、工程引用、ROM、整数权重转换/参考向量、描述符及内存重建检查，没有安装工具、编译 C、执行 HDL 仿真、综合布局布线或烧录设备。性能及赛题证据见[对照表](competition-checklist.md)。
