# 正确项目来源与本地同步记录

后续迁移整理已明确 `run_batch_gpu1.sh` 为大豆重建入口，并在本地更新 `requirements.txt`、`README.md` 及新增双环境清单。下文的字节一致性结果记录的是首次同步完成时的状态；最新安装入口见 [MIGRATION_GPU1.md](MIGRATION_GPU1.md)。

同步日期：2026-10-16。

本地 `deployment/Fast-PGSR` 的代码基准已切换为以下 **4090 多卡服务器上的工作目录实际内容**：

- SSH：`ubuntu2@172.26.19.165:2522`
- 已核验主机名：`ubuntu2-Super-Server`
- 正确远端项目：`/media/ubuntu2/A/LWL/code/Train/Fast-PGSR`
- GPU：1 × RTX 4090、2 × RTX 4090 D，均约 24 GB 显存。
- Git HEAD：`eb17aba468cfa5f9a477287c7867f1d214a66994`。同步的是工作目录代码，包含相对 HEAD 的修改及未跟踪源码，不能只用该提交复现。

此次指定目录是本地代码的权威来源；此前本地针对其他 3090 / L20 部署的改动已由本次服务器代码覆盖。

## 对比与覆盖结果

共同步 1,873 个源代码、配置及说明文件（约 11.73 MB）：新增 484、覆盖修改 8、内容相同 1,381。对同步文件逐一计算 SHA-256，随后重新读取远端源文件与本地文件核验，均无不一致。

8 个覆盖修改的文件是 `README.md`、`arguments/__init__.py`、`run_batch.sh`、`scene/__init__.py`、`scene/cameras.py`、`scripts/preprocess/convert.py`、`train.py`、`utils/graphics_utils.py`。

同步范围包括训练、渲染、ArUco 对齐、语义掩膜、目标导出、叶片处理及其 CUDA 扩展、Grounded-SAM-2 等源码依赖。通用植株处理代码保留，供大豆重建链路使用。

未拉取数据集、视频、图像、点云/网格结果、模型权重、日志、编译产物、缓存或远端 `.git`。模型配置源码保留；模型权重与 tokenizer 资源不在本次同步范围。未修改服务器文件、未启动或中断训练任务。

本地原目录的 3,173 个文件已在覆盖前完整归档（不含 `.git`）。本地独有的 `activate_local.sh`、`setup_server.sh`、`scripts/preprocess/extract_frames.py`、`utils/training_cache.py`、`GPU检查与修改记录.md` 已归档后移出当前项目，清理了 22 个旧 Python 缓存文件。

备份与详细比较位于工作区根目录：

```text
server-audit/4090-sync-20261016-212511/
  local-before-sync.tar.gz
  server-code.tar.gz
  manifest.json
  comparison.json
  result.json
  verification.json
```

## 双环境结论

原 `run_batch.sh` 中的 `FastGS` 在服务器上不存在，`fast-pgsr` 缺少两个训练扩展；较新的 `run_batch_gpu1.sh` / `run_batch_gpu2.sh` 实际使用 `Fast-Sam2` 和 `fast-pgsr-old`。

已额外新增 [环境导出及安装说明](environments/README.md)，包括：

- 与旧脚本 `FastGS` 环境名匹配的最小 ArUco 环境定义。
- 实际 `Fast-Sam2` 的完整包导出与可迁移安装清单。
- `fast-pgsr`、`fast-pgsr-old` 的完整导出与源码扩展安装说明。

环境文件和本报告是同步后新增的本地交付文件。1,873 个服务器来源文件保持原字节内容，脚本的原始路径、阶段开关和 GPU 配置未改写。安装清单经过结构与路径检查，但尚未在全新机器上执行安装或端到端重建。
