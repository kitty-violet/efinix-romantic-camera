# 团队交接与比赛提交

## 本次已交付

| 内容 | 入口 |
| --- | --- |
| r15 complete 原包全部 3240 个文件与说明 | [FPGA 目录](../fpga/README.md)、[工程入口](fpga-project.md) |
| Sapphire/C 固件、预编译 ELF/BIN/ROM | [固件目录](../fpga/firmware/) |
| Efinity 工程、外设/引脚与 SDC | [project](../fpga/project/) |
| 五风格 checkpoint、整数权重、量化/结构与参数表 | [网络与模型](network.md)、[models](../fpga/network/models/) |
| 42 个可读计算核设计源与调度器 | [核工程](../fpga/network/rtl/cnn_production_frame.xml)、[scheduler](../fpga/network/scheduler/) |
| 计算/DMA 程序、六槽 Flash 数据区 | [programs](../fpga/network/programs/)、[runtime_data](../fpga/network/runtime_data/) |
| 原训练/导出/集成开发源码与来源记录 | [original_sources](../fpga/original_sources/)、[SOURCE_ORIGINS](../fpga/SOURCE_ORIGINS.json) |
| 测试台、测试数据与厂商仿真模型 | [tests](../fpga/tests/) |
| V9-P1 正式打印件、试配件、装配 STEP | [机械目录](../mechanical/enclosure_v9/README.md) |
| CAD 源码、参数和历史检查 | [源文件目录](../mechanical/enclosure_v9/04_核对与源文件/) |
| 环境和完整下载方法 | [环境文档](environment.md) |
| 已烧录设备的使用方式 | [使用文档](usage.md) |
| 手机控制对接协议 | [蓝牙协议](ble-protocol.md) |
| 赛题逐项对照 | [要求对照](competition-checklist.md) |
| 资产来源、大小、SHA-256 | [源资产清单](../manifests/source-assets.json) |
| 仓库自动校验 | [验证说明](verification.md) |

归档包含原样 r15 complete 包和原 V9 外壳，共 3274 个原始资产，约 308.8 MiB。个人 Python/Conda 环境、Efinity 安装包和许可证、GitHub 凭据及完整训练素材未纳入。`project/outflow`、`firmware/prebuilt`、`network/`、原始开发源码及来源验证日志全部保留；V9 的筛选范围与初次归档相同。

原交付历史报告描述原始包的检查结果，不能当作当前仓库全文件清单。当前收录资产使用新清单校验。

## 已补齐与仍需补充

相比 11 月 3 日代码包，完整包补齐五套网络与权重、可读计算核、调度/描述符、六槽数据区及来源检查记录。工程/固件与原 V9 外壳未变。下表只列当前尚未交付或未重新验证的部分：

| 类别 | 当前缺项 |
| --- | --- |
| 构建验证 | 工程引用检查已通过；还需同版 Efinity 的综合、布局布线、时序/资源报告及合法 IP 工具支持 |
| 板上运行交付 | 仍需 r15 最终 FPGA 位流、正式烧录配置、匹配相机更新和回读验收；已附数据区 `0x340000` 起的 BIN，但其来源为未板验，不是整片镜像 |
| 网络与模型 | 已附结构、五风格参数/量化图/导出来源及映射；仍需正式数据集说明、PSNR/SSIM 等对照评价及端到端性能证据 |
| 软硬件协同证据 | Sapphire 和 C 实现已提供，仍需实际运行记录、固件/镜像与演示对应关系 |
| 测试证据 | 已附测试源码、参考向量/输出图和来源核编译/展开记录；仍需控制测试完整 runner、功能仿真、原图对照、长视频、FPS/切换延迟/掉帧实测 |
| 手机端 | 已验证 App/APK 或独立应用源码、蓝牙模块配置 |
| 机械实物 | 打印材料/工艺、首件试配、装配照片、实测接口与线缆空间 |
| 作品申报 | 按比赛系统要求的报告、团队信息和其他表单；当前未提供，未虚构填写 |

缺项不需要靠复制整个电脑环境解决。由相应成员从已验证工程导出，注明版本及必要配置，再按[更新约定](../CONTRIBUTING.md)纳入。

`firmware/prebuilt/control.bin`是 6168 字节 CPU 固件，四个 ROM lane 用于 Sapphire 片上 RAM 初始化；都不是 FPGA 位流。`network/runtime_data/flash-user-data.bin`只覆盖 `0x340000` 至 `0x5EB000`（不含）的数据区，不能从 0 烧录；仿真 `tests/data`不能替代生产数据。具体相机配置和待验状态见[工程说明](fpga-project.md)。

## 提交仓库链接前

1. 确认使用正确版本，记录提交时的 Git commit；保留与演示镜像一致的对应关系。
2. 在一份新克隆中运行 `git lfs pull` 和完整文件校验，确认大文件能取到。
3. 核对正式比赛提交说明中的链接格式、报告/视频要求；不要将 GitHub 文件归档代替全部申报材料。
4. 仓库按所有者本次要求改为公开，评委可直接访问 `https://github.com/kitty-violet/efinix-romantic-camera`，无需邀请账号。
5. 公开前检查常见凭据和个人敏感文件；保留厂商版权/保护声明，不擅自赋予第三方 IP 开源授权，参照[权利说明](../THIRD_PARTY_NOTICES.md)。

GitHub Actions 的“Repository integrity”只检查交付文件和文档链接，不代表 Efinity 编译或硬件测试通过。比赛表格中的“已完成”应由实际证据支持。
