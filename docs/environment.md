# 环境与获取方式

## 按用途选择环境

| 用途 | 需要的工具 | 版本与限制 |
| --- | --- | --- |
| 浏览文档和小模块 | 浏览器、文本编辑器 | 不需要训练环境 |
| 完整克隆 | Git、Git LFS、已授权的 GitHub 账号 | 归档时使用 Git 2.54.0.windows.1、Git LFS 3.7.1；不要求照搬个人安装路径 |
| 完整性校验 | Python 3.10+ | 只用标准库；归档使用 Python 3.12.14 |
| 恢复 FPGA 工程 | Efinity、对应器件支持和合法授权 | 网表来源标识为 2026.1.132，应优先和原工程保持一致；需先补齐下文工程材料 |
| 板上下载调试 | 与板卡匹配的下载器、驱动和已验证配置 | 本仓库没有提供镜像或下载配置 |
| 查看/打印外壳 | 支持 STEP 的 CAD 查看器或支持 STL 的切片软件 | 单位 mm、比例 1:1，按正式件数量打印 |
| 修改并重建外壳源码 | Python、CadQuery、NumPy | 本机已有依赖元数据记录 CadQuery 2.8.0、cadquery-ocp 7.9.3.1.1；仅供匹配参考，本次未重建 |
| 训练模型 | 不属于本版本交付 | 没有训练脚本、数据集、训练权重或推理 Python 程序，无需因此安装 CUDA/PyTorch |

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
```

以上 Python 路径是占位示例，不是需要新建的目录。私有库的克隆凭据由本人 GitHub 登录或 Git 凭据管理器提供，不要把 token 写在命令、URL 或仓库文件里。

若先前克隆时跳过了 LFS，进入仓库后运行 `git lfs pull` 再校验。若已有 Git LFS 但当前仓库未初始化其钩子，可用 `git lfs install --local` 仅配置当前仓库。

GitHub 网页“Download ZIP”对 LFS 的处理可能与仓库设置有关，因此请以校验结果确认取到的是实文件。内容以 `version https://git-lfs.github.com/spec/v1` 开头的几行小文本是 LFS 指针，不是网表或 CAD。

## 磁盘和内存

39 个导入资产共 229,416,618 字节，约 218.8 MiB，其中顶层网表约 124.4 MiB。各文件的精确大小记录在[资产清单](../manifests/source-assets.json)。完整工作副本与 `.git/lfs` 缓存会同时占用磁盘。

建议为首次完整克隆和日常更新先预留至少 1 GiB 磁盘；保留多版本 CAD、复制验证仓库时需另加空间。这是归档操作预算，不包括 Efinity 安装、综合、布局布线或 CAD 重建临时文件。

文件总大小不是内存需求。浏览小模块和校验脚本不需要 GPU，校验逐块读取文件；大网表编辑、CAD 查看和 Efinity 编译的峰值内存取决于工具及工程配置，本次没有完整工程，不能给出已验证的编译 RAM 数字。

## 恢复 Efinity 工程

当前无法从本仓库独立构建位流。获得队友原始工程后按以下顺序进行：

1. 确认具体板卡版本、FPGA 完整料号、晶振/接口连接，与当前成功实板一致。
2. 从队友补齐工程 XML、SDC、引脚/外设配置、原 IP 生成配置和源文件清单。
3. 优先使用来源版本 Efinity 2026.1.132 及相应 Titanium 支持；先核对已有安装和许可证，不直接升级原工程。
4. 在独立工程副本中确认顶层为 `ti60f225_oob_top`，按原清单加入四个 HDL 文件，避免重复定义，保留原语和受保护 IP。
5. 先检查工程加载与 IP 解析，再做综合、布局布线、时序检查；任何缺失约束、未绑定端口或 IP 不匹配均需回到原工程修复。
6. 只有获得通过检查的镜像、正确下载配置及对应硬件后，才能按板卡官方/团队已验证流程进行下载。

没有材料支持的引脚编号、Flash 地址、电源参数、命令行构建选项在本仓库不作猜测。

## 外壳设计环境

使用现成 STEP/STL 不需要 CadQuery。若修改设计，可参考已有 CadQuery 2.8.0 / OCP 7.9.3.1.1 环境，在兼容的个人 CAD 环境中核对：

```powershell
& 'E:\your-cad-python\python.exe' -c "import sys,cadquery,numpy; print(sys.version); print(cadquery.__version__,numpy.__version__)"
```

源码重建会写回导出目录和报告，操作方法及副本要求见[外壳 README](../mechanical/enclosure_v9/README.md)。本次归档只验证文件字节与历史记录，没有重新执行 CAD 几何计算。
