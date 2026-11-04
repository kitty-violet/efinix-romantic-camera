# 文件完整性验证

## 完整工作副本

在仓库根目录运行：

```powershell
git lfs pull
python -B scripts/verify_repository.py
python -B fpga/tools/verify_package.py
python -B fpga/tools/verify_complete.py
git lfs fsck
```

使用本人已有 Git/Python 的实际路径。脚本只使用 Python 3.10+ 标准库，不安装依赖、不改写文件、不运行 FPGA 或 CAD 工具。

检查内容包括：

- 清单路径、重复项和总字节数。
- 每个导入资产的存在性、字节数和 SHA-256。
- Markdown 中相对链接指向的文件/目录是否存在；不联网、不校验锚点。
- 机械零件 CSV 中 STEP/STL 引用及历史最终报告中的网格/STEP 校验值。
- GitHub 工作流、关键说明和 HDL 入口是否存在。
- complete 原包清单的 3239 个条目及额外 MANIFEST 自身是否全部收录且与仓库资产清单一致。
- Efinity XML 中的设计源、SDC、外设文件引用是否存在。
- CPU 固件补零至 8 KiB 后与 project/prebuilt 四个 ROM lane 是否逐字节一致。
- 五套 checkpoint/整数参数入口、S4R1 拓扑与 4035 参数统计、42 个可读核设计源引用。
- 六槽运行目录中程序/驻留系数的 SHA-256、数据区地址及长度。

原包 `verify_complete.py`还会实际检查各风格整数权重转换、参考向量 oracle、核源码来源及唯一便携调整、描述符逐位打包，并在内存中重建运行数据与交付 BIN 比对；不需要 PyTorch。

成功退出码为 0。缺文件、内容变化、未拉取 LFS、清单错误或断链均以非零退出，并列出具体路径。

## GitHub 自动检查

[工作流](../.github/workflows/verify.yml)在 push、pull request 或手动触发时执行：

```text
python -B scripts/verify_repository.py --allow-lfs-pointers
```

自动检查默认不下载大文件，而是核对 LFS 指针声明的 SHA-256 和字节数是否与清单一致，普通文件、原包清单、工程引用、网络元数据及 CPU ROM 仍做实际检查。这**不能证明 LFS 服务端对象可下载**，也不执行完整整数 oracle。完整下载后应执行上面各级校验；原包 `verify_package.py` 和 `verify_complete.py`需要真实大网表，不能在只取到指针时运行。

## 清单的含义

[source-assets.json](../manifests/source-assets.json)记录完整 RAR 原样导入的 3240 个文件，以及保留的 34 个 V9 资产。`source_root`是输入包名称，`source_path`是包内相对路径，不依赖作者电脑绝对路径。另记录 RAR 的 SHA-256 和上一代码版 commit；机械筛选范围与初次归档相同。

输入 RAR 的 SHA-256 为 `75c6239b7572f0c4f8df06577bf9398940b079b849d5f339448c678732bc7172`。本次未替换网表或机械资产，新增 3151 个原包文件，仅原包 README/MANIFEST 相对上一 ZIP 更新。

根 README、环境/使用文档和仓库检查脚本以 Git commit 跟踪；`fpga/README.md` 是 r15 原包文件，保持原样并纳入资产哈希。后续有意修改 FPGA 文件时，应同时更新原包 MANIFEST 与仓库清单，并记录版本，不能直接关闭检查。

完整性检查不能证明 HDL 功能、时序、帧率、打印几何或实物装配通过。原包的已验证声明与本次重新执行的文件检查分别记录，见[赛题对照](competition-checklist.md)。

## 验证边界

`fpga/检查结果.json`和`fpga/verification/core/`是队友包内的来源验证记录，不是本轮重新运行 PyTorch/Questa 的结果。运行数据来源明确标为 `NOT_BOARD_VERIFIED`。本轮不训练、不安装环境、不运行综合/布局布线/HDL 仿真，也不对板卡写入数据。
