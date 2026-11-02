# 文件完整性验证

## 完整工作副本

在仓库根目录运行：

```powershell
git lfs pull
python -B scripts/verify_repository.py
git lfs fsck
```

使用本人已有 Git/Python 的实际路径。脚本只使用 Python 3.10+ 标准库，不安装依赖、不改写文件、不运行 FPGA 或 CAD 工具。

检查内容包括：

- 清单路径、重复项和总字节数。
- 每个导入资产的存在性、字节数和 SHA-256。
- Markdown 中相对链接指向的文件/目录是否存在；不联网、不校验锚点。
- 机械零件 CSV 中 STEP/STL 引用及历史最终报告中的网格/STEP 校验值。
- GitHub 工作流、关键说明和 HDL 入口是否存在。

成功退出码为 0。缺文件、内容变化、未拉取 LFS、清单错误或断链均以非零退出，并列出具体路径。

## GitHub 自动检查

[工作流](../.github/workflows/verify.yml)在 push、pull request 或手动触发时执行：

```text
python -B scripts/verify_repository.py --allow-lfs-pointers
```

自动检查默认不下载大文件，而是核对 LFS 指针声明的 SHA-256 和字节数是否与清单一致，普通文件仍做实际哈希。这能发现指针/清单不一致，但**不能证明 LFS 服务端对象可下载**。拿到可用于综合或打印的完整副本，必须在已授权机器上运行不带该选项的完整检查。

## 清单的含义

[source-assets.json](../manifests/source-assets.json)记录本次从两份输入目录筛选导入的原始资产。`source_root` 是输入包名称，`source_path` 是该包内部相对路径，不依赖作者电脑绝对路径。文件原样复制，SHA-256 在复制前后核对。

新增的 README、环境/使用文档和检查脚本不属于原始资产，使用 Git commit 跟踪；它们的相对链接仍接受检查。后续若有意修改原资产，应更新来源、版本和清单，不应直接关闭检查。

完整性检查不能证明 HDL 功能、时序、帧率、打印几何或实物装配通过。原包的已验证声明与本次重新执行的文件检查分别记录，见[赛题对照](competition-checklist.md)。
