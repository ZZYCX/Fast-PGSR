# 4090 服务器环境核验与安装

后续已明确 `run_batch_gpu1.sh` 为大豆三维重建的正确入口；迁移安装以项目根目录的 [MIGRATION_GPU1.md](../MIGRATION_GPU1.md) 和两组 `requirements-*.txt` 为准。下文同时保留早期 `run_batch.sh` 的审计信息。

核验日期：2026-10-16。来源：`ubuntu2-Super-Server`（`172.26.19.165:2522`）上的 `/media/ubuntu2/A/LWL/code/Train/Fast-PGSR`。

## 脚本与环境的实际对应关系

| 脚本 | 对齐/分割环境 | 训练/渲染环境 | 核验结论 |
|---|---|---|---|
| `run_batch.sh` | `FastGS` | `fast-pgsr` | `FastGS` 不存在；`fast-pgsr` 缺少 `diff_gaussian_rasterization_fastgs` 和 `fused_ssim` |
| `run_batch_gpu1.sh` / `run_batch_gpu2.sh` | `Fast-Sam2` | `fast-pgsr-old` | 两个环境存在；`Fast-Sam2` 的对齐 API 和 `fast-pgsr-old` 的四个训练扩展导入通过 |

`run_batch.sh` 的 COLMAP 阶段用当前 Shell 的 `python` 调用预处理脚本，外部程序为 `colmap`；ArUco 阶段使用 `conda run -n FastGS`；训练/渲染/导出前才 `conda activate fast-pgsr`。当前开关只有抽帧和 COLMAP 开启，`RUN_ALIGN=0`、`RUN_TRAIN=0`，因此默认执行时不会触发缺失的 FastGS。

原始 `requirements.txt` 只有六项未固定版本的包，既不创建 Conda 环境，也未覆盖 `pycolmap` 和编译扩展。原始 `environment.yml` 仅描述 `fast-pgsr`，没有对齐环境。两个原文件保持服务器原样。

## 只补齐 run_batch.sh 缺失的 ArUco 环境

在 Linux / WSL 的项目根目录执行：

```bash
conda env create -f environments/environment-align.yml
conda run -n FastGS python -c "import cv2,numpy,pycolmap; print(cv2.__version__, numpy.__version__, pycolmap.__version__); assert hasattr(cv2.aruco,'ArucoDetector'); assert hasattr(pycolmap,'Sim3d')"
```

这个精简环境名为 `FastGS`，与原 `run_batch.sh` 一致；版本取自服务器上实际存在且导入成功的 `Fast-Sam2`：Python 3.10.20、NumPy 2.2.6、OpenCV 5.0.0.93、pycolmap 4.1.0。它满足 `align_aruco_refine.py` 的第三方导入，但不含分割模型依赖。相同的三个 Python 包也单独列在 `requirements-align.txt`。

这是依据现存 Fast-Sam2 环境整理的最小安装清单，**不是不存在的 FastGS 环境的原始导出**。

## 安装实际使用的完整 Fast-Sam2 环境

```bash
conda env create -f environments/environment-Fast-Sam2.conda.yml
conda activate Fast-Sam2
python -m pip install -r environments/requirements-Fast-Sam2.txt
python -m pip install --no-build-isolation --no-deps -e ./third_party/Grounded-SAM-2/grounding_dino
python -m pip check
```

`environment-Fast-Sam2.conda.yml` 保留 Conda 包版本；Python 包另列于 `requirements-Fast-Sam2.txt`，包含 CUDA 11.8 对应的 PyTorch wheel 索引。它们由完整导出拆分得到，去除了远端绝对 prefix 和需要从本项目源码安装的 groundingdino 条目。

原 groundingdino 的 editable 安装指向另一个项目目录 `/home/ubuntu2/cyb/Fast-PGSR/...`，直接 `import groundingdino` 在本次非交互核验中失败。因此不能把原 `pip freeze` 原封不动作为可迁移安装清单。这里改为安装本项目的源码。当前分割脚本同时会把 `third_party/Grounded-SAM-2` 加入 Python 搜索路径，从该目录导入 SAM2 和 GroundingDINO。模型权重与 tokenizer 资源按本次要求未同步，需要使用已有资源另行配置。

## 安装训练环境

推荐按较新的 GPU 批处理脚本安装 `fast-pgsr-old`：

```bash
conda env create -f environments/environment-fast-pgsr-old.conda.yml
conda activate fast-pgsr-old
python -m pip install -r environments/requirements-fast-pgsr-old.txt
python -m pip install --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization_fastgs ./submodules/diff-plane-rasterization ./submodules/simple-knn ./submodules/fused-ssim
```

四个扩展必须在目标 Linux 机器上针对对应 PyTorch/CUDA 重新编译。服务器实际版本为 Python 3.7.13、PyTorch 1.12.1+cu116。编译需 CUDA toolkit / nvcc 和 C++ 编译器；GPU 驱动与 PyTorch wheel 不能替代 nvcc。

`pytorch3d==0.7.1` 在原环境中由本项目外的 `/media/sx639-2/D/LWL/software/pytorch3d-0.7.1` 安装，其源码未包含在此次项目同步范围。它已从可迁移 pip 清单移至 `local-packages-fast-pgsr-old.json`，仍需在目标环境中安装匹配版本才能完成全部依赖。

若需复现原 `fast-pgsr`，对应 `.conda.yml` 和 pip 清单也已提供，但原环境缺少两个扩展，不能视为完整训练环境；同样应安装上面的四个扩展。继续使用原 `run_batch.sh` 时，需将 `TRAIN_ENV` 改为已完整安装的环境名，或把完整依赖安装到名为 `fast-pgsr` 的环境。

## 原始导出与验证范围

`exports-4090/` 保留三个实际环境的：

- `environment-*.raw.yml`：原始完整 Conda 导出（含服务器绝对路径）。
- `environment-*.no-builds.yml`：不带 build 号的完整导出。
- `conda-*.explicit.txt`：Linux Conda 包精确地址；不包含 pip 包。
- `conda-*.json`：Conda 包元数据。
- `requirements-*.freeze.txt`：原始 `pip freeze --all`，含不能直接迁移的本地路径。

可迁移清单不包含本地源码包，缺失项列在 `local-packages-*.json` 中；这避免 pip 到包索引上寻找不存在的扩展版本。原始导出供审计，安装时使用上面的拆分清单。

已验证源服务器的包导入、ArUco API、PyTorch CUDA 可用性及三个批处理脚本的 Bash 语法。未创建新环境、未在目标机重新安装、未跑训练或重建，因此完整清单尚未经过全新环境安装验证。非交互 SSH 的默认 PATH 中没有 `ffmpeg`、`colmap`、`nvcc`；运行前需初始化 Conda，并配置这些外部工具。

批处理脚本仍保留服务器原始的绝对输入/输出路径、运行开关和 GPU 编号。迁移执行前应修改路径与开关；`convert.py` 会清空场景的 images、sparse 和 database 后重新预处理。
