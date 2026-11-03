# Sapphire + native8h 第十五版代码

版本：production-r15，日期：2026-11-03。

## 目录

- `project/`：完整 FPGA 工程、Verilog/SystemVerilog 源码、官方 Sapphire 核、ROM 初值、引脚和时序约束。
- `firmware/`：C 控制程序、相机配置表、启动汇编和链接脚本；`prebuilt/` 为本版对应的 ELF、BIN、反汇编和 ROM 文件。
- `sapphire/`：官方核生成配置、CPU 配置和 BSP。
- `tests/`：控制桥、服务后端、DMA、相机及固件测试源码；`data/` 为测试数据，`vendor/` 为对应的仿真模型。
- `tools/`：固件编译及文件完整性检查工具。
- `MANIFEST.json`：文件 SHA-256 清单。

## 工程参数与功能

器件为 Efinix Titanium Ti60F225，工具版本为 Efinity 2026.1.132，顶层为 `ti60f225_oob_top`。CPU 为官方 Sapphire trim3_no_clint，RV32I + Zicsr/Zifencei，100 MHz、8 KiB OCR；视频计算域为 150 MHz。

C 固件负责相机配置、目录检查与装载调度、事件及模式切换。硬件保留底层传输、CRC、地址保护、流水计算和显示。工程包含六模式、中文名称、FPS、BLE/按键控制和全半屏切换。r15 的模式控制网采用 16 支组合缓冲，其源码已包含在 `project/native222.v` 中。

## 打开与使用

1. 解压到本地目录，用 Efinity 2026.1.132 打开 `project/ti60f225_oob.xml`。
2. 工程文件使用包内相对路径；RTL、四个 CPU ROM 文件及外围接口文件已放在对应位置。
3. C 固件入口为 `firmware/main.c`，寄存器定义在 `firmware/control.h`，内存布局在 `firmware/memory.ld`。
4. 如需重新编译固件，使用 Python 3 和 RV32 GCC 15.2.0 工具链：

   ```powershell
   python .\tools\build_firmware.py --toolchain "E:\path\to\gcc"
   ```

   输出位于 `build/firmware/`。将其中四个 `EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol*.bin` 复制到 `project/`，即可供 FPGA 工程使用。

5. 文件完整性检查：

   ```powershell
   python .\tools\verify_package.py
   ```

## 测试源码

`tb_control_rpc.sv`、`tb_service_backend.sv`、`tb_service_word_dma.sv` 和 `tb_camera_service.sv` 分别对应控制跨域桥、服务后端、DMA 和相机接口；`tb_control_system.sv`、`tb_production_firmware.sv` 对应 CPU 相邻接口与 C 固件测试。仿真时使用 `tests/vendor/EfxSapphireSoc.v`，综合工程使用 `project/EfxSapphireSoc.v`。固件测试的运行目录需要放置 `tests/data/` 下的三个 HEX 文件和本版四个 CPU ROM 文件。
