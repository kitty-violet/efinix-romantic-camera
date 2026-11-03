# 更新与团队协作

本库记录当前 production-r15 工程和指定 V9-P1 外壳。更新前先阅读[原包说明](fpga/README.md)、[工程说明](docs/fpga-project.md)、[赛题对照](docs/competition-checklist.md)和[第三方说明](THIRD_PARTY_NOTICES.md)。

## 修改约定

1. 功能修改使用分支，说明基于的代码/镜像版本、变更行为及验证方式。
2. 保留网表和受保护 IP 原貌；完整工程应由原维护者按已验证配置导出，不从无关工程猜配。
3. 替换大网表、STEP/STL/BREP 时继续使用 Git LFS，提交前检查 `git lfs ls-files`。
4. 原始资产有意变化时更新 `manifests/source-assets.json` 的大小、SHA-256、来源与版本信息；FPGA 包还需更新 `fpga/MANIFEST.json`。修改固件时保持 C 源、预编译交付及 project 四个 ROM lane 一致；CAD 更新清单、说明和对应报告。
5. 更新 `CHANGELOG.md`，运行完整文件校验，说明是否完成综合、时序、实板或打印测试。
6. 不提交环境目录、许可证、token、私钥、工具安装包、无关模型/数据或缓存。

## 拉取与验证

```powershell
git pull --ff-only
git lfs pull
python -B scripts/verify_repository.py
python -B fpga/tools/verify_package.py
git lfs fsck
```

上述命令使用自己已有环境的实际入口。若本地有未提交修改，先自行保存或提交，不用覆盖/重置命令解决。

## 补充完整工程

参照[交接缺项](docs/submission.md)，补充生产 Flash 数据镜像、位流、模型资料和验证证据。r15 已附工程与约束，不要再使用旧版四文件工程说明。源码包内 `project/outflow`、预编译 BIN 和 ROM 是有意保留的交付资料；新编译输出仍放忽略的 `fpga/build`、`work_syn`、`work_pnr` 等目录。大型镜像/权重按实际交付需要选择 LFS，不提交安装环境。

测试结果注明硬件、工具版本、输入条件和 commit。文件校验通过、原说明的历史实板声明、本次实测三者应分别表述。
