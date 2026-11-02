# FPGA 代码说明

本目录完整保留团队提供的 `github代码` 中的四个 HDL 文件及[原始说明](说明.txt)，没有修改逻辑、文件字节或受保护 IP。

## 文件与职责

| 文件 | 作用 |
| --- | --- |
| [rtl/native222.v](rtl/native222.v) | 冻结顶层 Verilog 网表；顶层 `ti60f225_oob_top`；含 Efinix 原语及受保护 IP；使用 Git LFS |
| [rtl/ble_uart_buttons.sv](rtl/ble_uart_buttons.sv) | UART 接收与 `A5 CMD ~CMD` 校验，将三个命令译码为单周期按键事件 |
| [rtl/legacy222_names.sv](rtl/legacy222_names.sv) | 在帧开始时锁存风格槽与就绪状态，清除旧标签区域并连接新 OSD |
| [rtl/five_style_name_osd.sv](rtl/five_style_name_osd.sv) | 五种中文风格名称字模和像素叠加，使用 `EFX_DPRAM10` |

`native222.v` 是大体积网表，不是按功能拆分、便于继续开发的完整源工程。浏览或评审时建议先看三个小模块，再通过顶层实例查找连接关系；无需先用编辑器展开整个网表。

## 已能从源码核对的连接

- 蓝牙 UART 接收模块参数为 150 MHz、115200 波特、8N1。
- `key_start`、`key_style`、`key_view` 分别与已有物理按键事件合并，送入原有控制逻辑。
- 顶层 `ble_uart_tx` 固定为高，当前接口没有实现命令 ACK。
- 中文风格名称在帧边界锁存槽位和就绪状态，不在帧内直接跟随选择信号变化。

这些代码事实只能说明所列接口，不能替代对整条图像通路、DDR、时钟、时序、风格切换延迟的测量。

## 重新构建的材料边界

网表文件头带有 Efinity `2026.1.132` 属性，依赖 Titanium 器件原语。当前版本缺少：

1. 原 Efinity 工程 XML、完整文件/IP 清单及生成配置。
2. 实际板卡对应的引脚、电压、时钟、DDR/MIPI/HDMI 外设配置。
3. SDC 约束、完整时序与布局布线报告。
4. 已验证位流、Flash 镜像、独立权重/数据映射及烧录参数。
5. 原始可维护设计源、模型结构与量化导出记录、仿真测试台。

拿到这些材料前，本仓库适用于代码查阅、接口交接与版本归档。不能仅用通用 Ti60 示例工程替代原配置，也不能将 `native222.v` 直接交给下载器作为位流。后续恢复工程的顺序见[环境说明](../docs/environment.md)。

## 相关入口

[使用说明](../docs/usage.md) · [蓝牙协议](../docs/ble-protocol.md) · [赛题对照](../docs/competition-checklist.md) · [IP 说明](../THIRD_PARTY_NOTICES.md)
