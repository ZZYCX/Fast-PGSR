# RTX 3090 / 32624：仅部署与环境安装

实施日期：2026-10-17（Asia/Shanghai）。

**本次仅完成项目及环境安装，未执行测试、推理或训练。**

本文是本次安装的独立记录。`DEPLOYMENT_3090_FULL_GUIDE.md`、`MIGRATION_GPU1.md`、`SINGLE_RUBIK.md` 保留为历史与使用说明；其历史测试、训练和重建结果不代表本次安装做过运行验证。

## 目标与边界

- SSH：`ssh -p 32624 root@172.26.14.81`。
- 主机名：`cuda11-8-0-cudnn8-devel-ub22-web-af67524bbf65434994-7d78d5pv6zk`。
- 本地源码：`C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\deployment\Fast-PGSR`。
- 服务器项目：`/root/Fast-PGSR`。
- 安装资料、下载缓存、编译日志：`/root/fast-pgsr-setup-only`。
- 正确业务入口：`run_batch_gpu1.sh`，本次未启动。
- 未运行测试套件、模型加载、CUDA 张量操作、视频抽帧、COLMAP 重建、分割、对齐、标签生成、训练、导出或渲染。

用户更正目标端口前，曾按最初提供的 32596 启动过安装。更正后停止本机指向旧端口的模型中转；对旧端口的 SSH 连接无法完成握手，未能送达停止其后台任务的请求。没有删除或回滚旧端口文件。后续所有安装均以 32624 为目标。32596 不作为本次成功部署的证明。

## 安装前只读检查

目标最初没有 `/root/Fast-PGSR`、`/root/miniconda3` 和历史结果目录，因此无需覆盖或移动旧项目。

| 项目 | 检查记录 |
|---|---|
| 操作系统 | Ubuntu 22.04.3 LTS，x86_64 容器 |
| GPU | NVIDIA GeForce RTX 3090，24576 MiB |
| 计算能力 | `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv` 报告 8.6 |
| 驱动 | 580.105.08 |
| CUDA toolkit | `/usr/local/cuda-11.8`，nvcc 11.8.89 |
| 初始编译器 | GCC/G++ 11.4.0 |
| 内存 | 24 GiB，无 swap |
| 初始项目盘可用空间 | 约 300 GiB |

据目标 GPU 检查结果设置 `TORCH_CUDA_ARCH_LIST=8.6` 和 COLMAP 的 `CMAKE_CUDA_ARCHITECTURES=86`。GroundingDINO 与 SAM2 的本地 setup.py 自带 70、75、80、86 多架构参数，本次保留源码参数；CUDA 11.8 不追加 sm_120。

## 源码迁移

使用本机 Python/Paramiko 经 SSH/SFTP 传输。首轮源码包大小 19,567,038 字节，初始清单为安装目录中的 `source-manifest.json`。收尾按迁移范围移除历史预览、样例照片、训练摘要及逐次运行记录共31个本次上传文件（9,420,846字节），并删除该首轮传输包。最终干净源码包为 `/root/fast-pgsr-setup-only/project-source-final.tar.gz`。

解包前确认目标项目目录不存在。首次解包后，对 3,718 个文件逐项核对大小与 SHA-256，全部与传输清单一致，其中保留 562 个 `.cpp/.cu/.h/.cuh` 编译输入。Shell 文件在打包时统一为 LF 换行。

最终项目保留 **3,687 个原部署源码/配置/依赖/文档文件**和全部562个 C++/CUDA 头文件及源文件；本次新增部署文档和安装审计资料另计。排除 `.git`、缓存、旧 build/dist、数据/输出/视频/运行日志目录、历史捕获帧和重建预览、旧 `.so/.o/.a/.obj/.pyd/.pyc` 及模型权重。必要预训练资源另行准备，目标服务器的新编译产物保留供运行使用。

期间本地 `.gitignore` 更新已同步；原服务器端文件备份在 `/root/fast-pgsr-setup-only/before-final-gitignore/.gitignore`。最终源码清单为 `source-manifest-final.json`，迁移排除与更新记录为 `final-migration-scope.json`。未改写原迁移文档或算法代码。

以下当前修复文件成套上传，未修改其算法、阈值或校验逻辑：

```text
scripts/mask_generation/instance_mask_guard.py
scripts/mask_generation/run_grounded_sam2_scene.py
scripts/mask_generation/run_rubik_scene_auto.py
scripts/validate_semantic_inputs.py
run_batch_gpu1.sh
```

源头掩码校验、输入绑定、单株空间清理及目标 PLY 渲染逻辑均保留。没有使用原始 GitHub 仓库或其他服务器代码覆盖本地版本。

## 系统与 Conda

系统安装命令记录于 `setup_only_system.sh` 与 `system.log`：

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ffmpeg build-essential cmake ninja-build pkg-config wget curl ca-certificates git libgl1 libglib2.0-0 libeigen3-dev libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libgmp-dev libmpfr-dev
```

使用 Miniconda 安装器 `Miniconda3-py310_24.7.1-0-Linux-x86_64.sh`，安装前缀 `/root/miniconda3`，关闭自动激活 base；没有覆盖已有 Conda。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/Fast-PGSR
conda env create -f environments/environment-fast-pgsr-old.conda.yml
conda env create -f environments/environment-Fast-Sam2.conda.yml
```

## Python 包与 CUDA 扩展安装方式

两个环境独立创建，要求如下；实际完成状态与版本见本文后续安装结果记录。

| 环境 | 固定 Python | 固定 PyTorch | 项目清单 |
|---|---|---|---|
| fast-pgsr-old | 3.7.13 | 1.12.1+cu116 | requirements.txt → requirements-fast-pgsr-old.txt，然后 requirements-fast-pgsr-old-source.txt |
| Fast-Sam2 | 3.10.20 | 2.7.1+cu118 | requirements-Fast-Sam2.txt，然后 requirements-Fast-Sam2-source.txt |

六个框架 wheel（各环境的 torch/torchvision/torchaudio）从 PyTorch 官方索引取得 URL 和 SHA-256，按 Linux x86_64 及 cp37/cp310 ABI 下载到 `wheelhouse/`，采用 HTTP Range 分块并发下载，合并后核对官方 SHA-256。清单保存为 `framework-wheels.json`，下载日志为 `wheels.log`。

先安装相应环境的三个 wheel，再以清华镜像安装普通依赖。项目原清单不变；仅在安装资料目录生成移除 `--extra-index-url` 行的临时清单，所有包的版本约束保持原样。

```bash
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=3 FORCE_CUDA=1
export SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
conda activate fast-pgsr-old
python -m pip install --no-deps /root/fast-pgsr-setup-only/wheelhouse/*cp37*.whl
python -m pip install -r /root/fast-pgsr-setup-only/requirements-fast-pgsr-old-mirror.txt
python -m pip install -v --no-build-isolation --no-deps -r requirements-fast-pgsr-old-source.txt
conda activate Fast-Sam2
python -m pip install --no-deps /root/fast-pgsr-setup-only/wheelhouse/*cp310*.whl
python -m pip install -r /root/fast-pgsr-setup-only/requirements-Fast-Sam2-mirror.txt
python -m pip install -v --no-build-isolation --no-deps -r requirements-Fast-Sam2-source.txt
```

PyTorch3D 使用清单指定的官方固定提交 `995b60e3b99faa1ee1bcdbe244426d54d98a7242`（0.7.1）。在线 codeload 下载停滞后，停止该下载并使用同一提交的本机缓存源码归档，SHA-256 为 `0b234e05e35f46a5ff18abb93c7f8321aa88115128267ce07a1170e4ae0a092d`，31,092,821 字节。只在安装目录生成将 URL 替换为该归档 file URI 的临时源码清单，原项目清单不变。继续命令为：

```bash
python -m pip install -v --no-build-isolation --no-deps \
  -r /root/fast-pgsr-setup-only/requirements-fast-pgsr-old-source-cached.txt
```

四个高斯/CUDA 扩展、GroundingDINO、SAM2 从本次上传的源码在目标环境编译，未传输旧服务器编译二进制。安装器自身为编译导入框架属于构建步骤，没有另行运行导入验证或模型加载。

分割环境普通依赖提前分为四组下载，一组因 `IncompleteRead` 中断。大型 NVIDIA runtime、Open3D、Triton 改按镜像 simple 索引记录的准确 wheel/ABI/大小/SHA-256 分块下载；27 项合计 4,436,991,475 字节。清单中的 cu11/cu12 runtime 版本均保留，编译工具包为 CUDA 11.8，框架 wheel 仍为指定的 cu116/cu118。下载缓存统一保存在 `wheelhouse/`，`/root/.config/pip/pip.conf` 的 `global.find-links` 指向该目录。

初次环境安装日志保留在 `environments-initial.log`，停止的源码下载退出码为 143；缓存源码恢复安装日志为 `environments.log`。这些中间下载中断记录与最终安装结果分别保存。完整 freeze 中的绝对路径只用于审计，后续迁移仍使用项目根目录的可迁移 requirements 与源码清单。

## COLMAP 安装方式

采用官方 3.9.1 源码，路径 `/root/fast-pgsr-setup-only/src/colmap-3.9.1`：

```bash
cmake -S colmap-3.9.1 -B colmap-3.9.1/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build colmap-3.9.1/build -j 3
cmake --install colmap-3.9.1/build
```

预期可执行文件 `/usr/local/bin/colmap`。仅记录构建及安装证据，不执行 COLMAP 重建。

## 模型资源准备方式

本地项目最初缺少七项资源，32596 源地址更正后无法完成 SSH 握手。采用模型官方公开下载地址；SAM2 在目标服务器直接分块下载，BERT 与 GroundingDINO 因目标服务器网络问题经本机下载中转至 32624，不读取或修改训练模型。

- BERT：`https://huggingface.co/google-bert/bert-base-uncased/resolve/main/` 下指定的五个文件。
- SAM2：`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`。
- GroundingDINO：`https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`。

按 `environments/exports-3090-32596/model-resources-sha256.json` 核对文件大小及 SHA-256；最终记录见安装结果。未通过模型加载或推理验证资源。

## 后续手动启动新视频（本次未执行）

安装全部完成后，将新视频自行放入独立目录，再使用全新数据、输出及日志目录。不要把历史结果当作新输入训练结果复用。

```bash
ssh -p 32624 root@172.26.14.81
source /root/miniconda3/etc/profile.d/conda.sh
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen
cd /root/Fast-PGSR
VIDEO_DIR=/root/new-plant-videos \
DATA_ROOT=/root/plant-new-run/data \
OUTPUT_ROOT=/root/plant-new-run/output \
LOG_ROOT=/root/plant-new-run/logs \
RENDER_VOXEL_SIZE=0.02 bash run_batch_gpu1.sh
```

`RENDER_VOXEL_SIZE=0.02` 遵循历史文档中针对 24 GiB 系统内存的建议，本次没有执行渲染以确认资源占用。默认源头实例约束和 `TARGET_SPATIAL_RADIUS=0.8` 保持不变。

## 最终安装结果

2026-10-17 18:30（Asia/Shanghai）读取安装记录：系统依赖、双环境安装、COLMAP、六个框架 wheel、27 项大型依赖下载及七项模型完整性记录的最终退出码均为 **0**。安装器已结束。本节依据包元数据、构建日志和文件哈希记录安装结果，没有进行运行验证。

| 实际环境 | Python | PyTorch | torchvision | torchaudio | NumPy | pycolmap |
|---|---|---|---|---|---|---|
| `/root/miniconda3/envs/fast-pgsr-old` | 3.7.13 | 1.12.1+cu116 | 0.13.1+cu116 | 0.12.1+cu116 | 1.21.6 | 0.4.0 |
| `/root/miniconda3/envs/Fast-Sam2` | 3.10.20 | 2.7.1+cu118 | 0.22.1+cu118 | 2.7.1+cu118 | 2.2.6 | 4.1.0 |

训练环境安装 PyTorch3D **0.7.1**；四个本地高斯/CUDA 包的 setup 元数据版本均为 **0.0.0**。分割环境安装 GroundingDINO **0.1.0**、SAM-2 **1.0**。七个目标服务器编译产物已记录路径、文件大小及 SHA-256：

```text
/root/miniconda3/envs/fast-pgsr-old/lib/python3.7/site-packages/
  pytorch3d/_C.cpython-37m-x86_64-linux-gnu.so
  diff_gaussian_rasterization_fastgs/_C.cpython-37m-x86_64-linux-gnu.so
  diff_plane_rasterization/_C.cpython-37m-x86_64-linux-gnu.so
  simple_knn/_C.cpython-37m-x86_64-linux-gnu.so
  fused_ssim_cuda.cpython-37m-x86_64-linux-gnu.so
/root/Fast-PGSR/third_party/Grounded-SAM-2/
  grounding_dino/groundingdino/_C.cpython-310-x86_64-linux-gnu.so
  sam2/_C.so
```

COLMAP **3.9.1** 位于 `/usr/local/bin/colmap`，24,143,080 字节。构建缓存记录 `CUDA_ENABLED=ON`、`CMAKE_CUDA_ARCHITECTURES=86`、`CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc`、`TESTS_ENABLED=OFF`。没有执行该二进制进行重建或功能检查。

七项模型资源均在目标项目的规定位置，总计 **1,458,752,435 字节**，全部匹配历史清单的文件大小与 SHA-256。五个必需的源头修复文件，以及 `render.py` 和 `scripts/export_target_gaussians.py` 的哈希均与部署清单一致。

完整记录同时保存在本地及服务器项目的 [environments/exports-3090-32624-setup-only/](environments/exports-3090-32624-setup-only/)：

- [installation-result.json](environments/exports-3090-32624-setup-only/installation-result.json)：最终安装退出码、关键版本、七个编译产物、模型与关键源码哈希。
- `fast-pgsr-old.pip-list.json`、`fast-pgsr-old.freeze.txt`、`fast-pgsr-old.conda.json`：训练环境完整实际版本。
- `Fast-Sam2.pip-list.json`、`Fast-Sam2.freeze.txt`、`Fast-Sam2.conda.json`：分割环境完整实际版本。
- `system-packages.tsv`、`colmap-CMakeCache.txt`、`build-install-summary.txt`：系统包、COLMAP 编译配置及安装摘要。
- `framework-wheels.json`、`framework-downloads-installed.json`、`large-wheels.json`、`installed-downloads.json`：六项框架与27项大型依赖下载记录。
- `source-manifest.json`、`model-resources-installed.json`、`pytorch3d-source.tar.sha256.json`：源码、模型、固定提交源码归档的完整性记录。
- `source-manifest-final.json`、`final-migration-scope.json`、`metadata-and-local-source-record.json`：最终迁移范围、当前本地源码及固定普通依赖实际版本的逐项记录。
- `setup_only_*.sh/.py` 与临时 requirements：实际安装脚本及路径调整记录；不含密码。

完整安装日志保留在服务器 `/root/fast-pgsr-setup-only/`：`system.log`、`environments-initial.log`、`environments.log`、`colmap.log`、`wheels.log`、`large-manifest.log`、`large-wheels.log`、`prefetch-*.log` 等。中间下载失败日志保留，未作为最终安装失败处理；最终环境安装退出码为 0。下载缓存及编译目录保留在该安装目录，不混入业务数据目录。安装后项目盘约剩余 **261 GiB**。

本次要求的安装项没有遗留失败。限制为：未执行测试、模型加载、CUDA 张量操作或任何业务流程，不能据本次安装记录声称新视频推理、训练、渲染或重建已运行成功。旧端口后台任务停止状态无法确认，详见本文目标与边界记录。

**本次仅完成项目及环境安装，未执行测试、推理或训练。**
