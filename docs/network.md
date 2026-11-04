# 网络与模型

当前为 `production-r15-complete`。原包的[逐层结构导航](../fpga/网络结构与源码导航.md)、[公共结构](../fpga/network/graph.json)和[来源映射](../fpga/SOURCE_ORIGINS.json)均保留。

## 五风格与原图

| 槽位 | OSD 名称 | 模型版本 | 入口 |
| --- | --- | --- | --- |
| 0 | 漫画 | R8-07 | [slot0](../fpga/network/models/slot0/) |
| 1 | 阿凡达 | B03 | [slot1](../fpga/network/models/slot1/) |
| 2 | 热成像 | R8-06 | [slot2](../fpga/network/models/slot2/) |
| 3 | 像素动画 | PIXEL_R39 | [slot3](../fpga/network/models/slot3/) |
| 4 | 阴曹地府 | W-T256 | [slot4](../fpga/network/models/slot4/) |
| 5 | 原图 | native1080 bypass | [slot5 说明](../fpga/network/models/slot5/identity.json) |

五风格共享 `S4R1_8_32_8_V1` 拓扑，各有独立 `selected.pt`、整数权重、量化图、参数表和参考向量。原图槽位没有独立 checkpoint；其 resident 数据只是装载 ABI 占位，不能算作第六套风格网络。

## 网络结构

输入 1280×720 RGB，stem 3×3 后经 stride4 深度卷积、逐点卷积变为 320×180×32；一个残差块后缩为 8 通道，nearest×4 恢复空间尺寸，再经深度卷积和 RGB 逐点输出。共 11 个卷积、13 节点、1 个残差块。

每风格 3808 个权重、227 个偏置，共 4035 个模型参数；另有 227 个 Q17 重定量乘数。参数表给出的 720p 运算量为 488,908,800 MAC/帧，不是实测吞吐。

| 文件 | 大小/用途 |
| --- | --- |
| `selected.pt` | 每风格 28,519 字节，PyTorch checkpoint |
| `coefficients.bin` | 每风格 5708 字节，整数软件推理参数 |
| `coefficients-padded.bin` | 8192 字节，对齐的整数数据 |
| `resident-padded.bin` | 20,480 字节，硬件驻留布局 |
| `graph.json` / `parameter-count.json` | 该风格拓扑、量化/残差规则和参数统计 |
| `reference-vectors.json` / `test-vectors/` | 整数参考向量及来源输出图 |

整数格式为 INT8 权重/激活、INT32 偏置、Q17 乘数，replicate padding、half-away-from-zero 舍入。不同文件布局不能互换，也不兼容早期固定 B1 二进制格式。公共 graph 中的历史标记不能替代当前整机绑定；实际映射见 [source_binding](../fpga/source_binding/)。

## 在已有环境中查看与推理

文件完整性和整数参考向量检查只需 Python 标准库。只有查看 checkpoint/执行图片推理才需已有 PyTorch、NumPy、Pillow，参考版本见 [requirements.txt](../fpga/network/requirements.txt)；本文不自动安装任何依赖。

在仓库根目录：

```powershell
# 查看包内网络和 slot0 checkpoint；slot 可取 0 到 4。
python -B fpga/network/model.py --slot 0

# 独立整数图像推理，输入必须已是 1280x720 RGB。
# 输出使用个人工作目录，不覆盖包内测试图。
python -B fpga/network/infer_package.py --style-root fpga/network/models/slot0 --input D:/your-work/input_720p.png --output D:/your-work/slot0.png --device cpu
```

`--device cpu` 不需要 CUDA；不要照抄脚本默认的 CUDA 设备而忽略本人环境。输出父目录须已存在。包内没有通用视频一键入口，不承诺本机 720p 实时速度。

若以 [model.py](../fpga/network/model.py) 的浮点接口调用，输入为 NCHW 的 `RGB_uint8 / 128 - 1`，输出乘 128 加 128 再裁剪/转 RGB。上述整数推理入口直接接收 uint8 RGB，内部采用减 128 的整数尺度，不能再手动重复归一化。

## FPGA 与来源

[可读核工程](../fpga/network/rtl/cnn_production_frame.xml)列出 42 个设计源，供阅读和单独核开发；整机工程仍打开 `fpga/project/ti60f225_oob.xml`，不要把可读核和已展平的核重复加入。

[programs](../fpga/network/programs/)提供 6546 条计算描述符和 8277 条 DMA 描述符，以及字段、分配记录和 BIN；[scheduler](../fpga/network/scheduler/)提供调度/打包源码。[运行数据](../fpga/network/runtime_data/)将程序、驻留系数及 RGB/ISP 表组织为六槽目录。烧录地址和限制见[工程说明](fpga-project.md)。

[original_sources](../fpga/original_sources/)保留 2972 个原训练、导出和集成开发文件，不包含完整素材集或个人环境。历史脚本可能引用作者磁盘路径和旧实验；复现当前已选模型优先使用包内独立入口，不要批量执行历史脚本。

来源 [检查结果](../fpga/检查结果.json)包含模型数值检查记录。本轮重新执行的是标准库转换/整数 oracle/数据一致性检查，没有重新执行 PyTorch checkpoint 数值测试、训练或板端推理。
