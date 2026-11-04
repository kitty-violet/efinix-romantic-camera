# 环境与获取方式

## 按用途选择环境

| 用途 | 需要的工具 | 版本与限制 |
| --- | --- | --- |
| 浏览文档和源码 | 浏览器、文本编辑器 | 不需要训练环境 |
| 完整克隆 | Git、Git LFS | 仓库公开，无需协作者权限；归档工具版本仅作参考，不要求照搬个人路径 |
| 完整性校验 | Python 3.10+ | 只用标准库；归档使用 Python 3.12.14 |
| 构建 FPGA 工程 | Efinity 2026.1.132、Titanium 支持和合法授权 | r15 已提供 XML、外设、SDC、ROM 及源文件，打开 `fpga/project/ti60f225_oob.xml` |
| 修改 C 固件 | Python 3、Windows RV32 GCC 15.2.0 | 原脚本使用 `riscv-none-elf-*.exe` 及工具链内 `rv32i/ilp32/libgcc.a` |
| 使用现成 CPU 固件 | 包内 `fpga/firmware/prebuilt` | 不需要先重新编译 C；需保持 project 四个 ROM lane 一致 |
| 板上下载调试 | 匹配的下载器、驱动、已验证位流与生产 Flash 数据 | 已附六槽数据区镜像，但未板验且没有最终 FPGA 位流；不能将数据 BIN 从地址 0 烧录 |
| 查看 checkpoint/图片推理 | 已有 PyTorch、NumPy、Pillow | 原包参考 `torch==2.11.0`，来源数值检查记录 `2.11.0+cu128`；CPU 推理不要求 CUDA，见模型说明 |
| 可读核编译/展开 | 已有合法 Questa 及许可证 | `tools/check_core.py`；不把许可证上传仓库，本轮未重新执行 |
| 查看/打印外壳 | 支持 STEP 的 CAD 查看器或支持 STL 的切片软件 | 单位 mm、比例 1:1，按正式件数量打印 |
| 修改并重建外壳源码 | Python、CadQuery、NumPy | 本机已有依赖元数据记录 CadQuery 2.8.0、cadquery-ocp 7.9.3.1.1；仅供匹配参考，本次未重建 |
| 复现训练 | 按选定来源脚本核对已有依赖和素材 | 已附原训练开发源码，但历史路径和数据需适配；本轮不训练、不自动安装 |

不打包任何虚拟环境、Conda 目录、Efinity 安装包、许可证、账号令牌、SSH 配置或个人环境说明。使用者应复用自己电脑上已有的兼容工具，先核对版本；本文不要求自动安装或升级工具。

## 获取完整内容

推荐在有足够空间的数据盘创建工作目录。中文目录可用于保存仓库和 CAD；本次没有验证 Efinity 整条工具链对中文路径的兼容性，恢复工程时宜另用简短的英文路径，如 `D:\fpga\romantic_camera`。

```powershell
# 在 PowerShell 中，将可执行文件改为自己电脑上已确认的路径。
$GitExe = 'D:\git\Git\cmd\git.exe'
$PythonExe = 'E:\your-python\python.exe'

& $GitExe --version
& $GitExe lfs version
& $PythonExe --version
& $GitExe clone https://github.com/kitty-violet/efinix-romantic-camera.git 'D:\fpga\romantic_camera'
Set-Location -LiteralPath 'D:\fpga\romantic_camera'
& $GitExe lfs pull
& $PythonExe -B scripts/verify_repository.py
& $PythonExe -B fpga/tools/verify_complete.py
```

以上 Python 路径是占位示例，不是需要新建的目录。公开仓库的浏览/克隆不需登录；提交和管理权限仍由所有者控制。不要把 token 写在命令、URL 或仓库文件里。

若先前克隆时跳过了 LFS，进入仓库后运行 `git lfs pull` 再校验。若已有 Git LFS 但当前仓库未初始化其钩子，可用 `git lfs install --local` 仅配置当前仓库。

GitHub 网页“Download ZIP”对 LFS 的处理可能与仓库设置有关，因此请以校验结果确认取到的是实文件。内容以 `version https://git-lfs.github.com/spec/v1` 开头的几行小文本是 LFS 指针，不是网表或 CAD。

## 磁盘和内存

3274 个导入资产共 323,843,953 字节，约 308.8 MiB，其中 r15 顶层网表约 174.8 MiB；完整 FPGA 包约 214.4 MiB，保留机械资产约 94.4 MiB。每套 checkpoint 约 27.9 KiB，运行数据约 2.67 MiB。精确大小记录在[资产清单](../manifests/source-assets.json)。完整工作副本与 `.git/lfs` 缓存会同时占用磁盘，旧提交的 LFS 缓存也可能继续保留。

建议为首次完整克隆和日常更新先预留至少 1 GiB 磁盘；保留多版本 CAD、复制验证仓库时需另加空间。这是归档操作预算，不包括 Efinity 安装、综合、布局布线或 CAD 重建临时文件。

文件总大小不是内存需求。浏览小模块、完整性/整数参考校验不需要 GPU；大网表哈希逐块读取。720p 模型推理的特征图及运算缓存远大于 checkpoint；大网表编辑、CAD 查看和 Efinity 编译的峰值内存也取决于工具及配置。本次未运行训练、720p 推理或综合布局布线，不给出未经测量的峰值 RAM/显存保证。

## 打开 Efinity 与构建固件

r15 已补齐前一归档缺少的工程 XML、外设、SDC、引脚资料、C 固件和测试源码。使用已有环境按以下顺序操作：

1. 确认具体板卡版本、FPGA 完整料号、晶振/接口连接，与当前成功实板一致。
2. 完整拉取 LFS，运行仓库与原包两级校验，保留整个 `fpga/project` 目录。
3. 使用来源版本 Efinity 2026.1.132 及相应 Titanium 支持，打开 `fpga/project/ti60f225_oob.xml`，确认顶层和器件。
4. 现成 ROM 可直接供该工程使用；只有修改固件时才需要已有 RV32 GCC 15.2.0，命令见[工程说明](fpga-project.md)。
5. 在工作副本中检查工程加载与 IP 解析，再做综合、布局布线、时序检查；本次只验证文件引用，没有代为完成这些构建。
6. 下载前补齐对应已验证 FPGA 位流和烧录配置，核对已附数据区镜像的 `0x340000` 地址、相机配置和来源待板验状态；C 固件 BIN、测试 HEX 和数据区 BIN 不能相互替代。

`fpga/sapphire/soc_config` 保留原核生成时的 Efinity 绝对安装路径。重新生成核时需按本人安装适配，不能把原 bootloader 默认 ROM 当成本版控制固件。不要更改原包来源文件后仍宣称原始哈希一致；具体操作与仿真限制见[工程说明](fpga-project.md)。

## 模型环境

先在本人已有环境中确认 `torch`、`numpy`、`PIL` 可导入，再按[网络与模型](network.md)运行包内入口。`fpga/network/requirements.txt`是参考依赖，不要求直接安装或升级；仅下载/核验本仓库不需要这些包。CPU 图片推理显式传 `--device cpu`，GPU 使用者自行匹配已有驱动与 PyTorch。

`original_sources/`中约 2972 个历史开发文件保留作者目录层次，不包含完整训练素材；其中脚本可能写入原工作目录。阅读后按个人环境选用，不将目录当作可直接运行的批处理，也不复制作者的整套虚拟环境。

## 外壳设计环境

使用现成 STEP/STL 不需要 CadQuery。若修改设计，可参考已有 CadQuery 2.8.0 / OCP 7.9.3.1.1 环境，在兼容的个人 CAD 环境中核对：

```powershell
& 'E:\your-cad-python\python.exe' -c "import sys,cadquery,numpy; print(sys.version); print(cadquery.__version__,numpy.__version__)"
```

源码重建会写回导出目录和报告，操作方法及副本要求见[外壳 README](../mechanical/enclosure_v9/README.md)。本次归档只验证文件字节与历史记录，没有重新执行 CAD 几何计算。
