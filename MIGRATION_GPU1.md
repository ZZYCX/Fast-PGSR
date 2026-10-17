# run_batch_gpu1.sh 大豆三维重建：双环境迁移

2026-10-17 的 RTX 3090 实际部署、系统依赖、双环境安装、输入校验、运行修复与结果路径见 [完整迁移文档](DEPLOYMENT_3090_FULL_GUIDE.md)。目标服务器环境快照位于 `environments/exports-3090-32596/`。下文的 4090 核验记录仍作为原始依赖来源；4090 源项目未同步修改。

当前入口已在MASK保存训练PNG前加入SfM实例正负提示和SAM2逐帧细化，两个视频304帧新掩码已替换正式输入，并重新执行LABEL及绑定。详见完整文档第14节。迁移时必须同时部署两个v4分割入口、新 `instance_mask_guard.py`、校验器和批处理脚本；原v3掩码不能直接通过新MASK验收。训练后继续保留 `TARGET_SPATIAL_RADIUS=0.8` 的附加空间筛选，3090设置 `RENDER_VOXEL_SIZE=0.02`。旧检查点绑定旧掩码，后续重训必须使用新输出目录；不要把旧模型当作新掩码的训练结果。

本项目以 **`run_batch_gpu1.sh` 为正确训练入口**。2026-10-16 已重新连接 `172.26.19.165:2522`，核验主机 `ubuntu2-Super-Server` 和项目 `/media/ubuntu2/A/LWL/code/Train/Fast-PGSR`。这是依赖迁移时的核验记录；之后本地已实施单魔方适配，脚本不再与4090源项目逐字一致，4090源服务器代码未同步修改。3090目标部署已与本地迁移修复同步。

## 两个环境

| 环境名（大小写保持一致） | 脚本阶段 | Python | PyTorch | NumPy | pycolmap |
|---|---|---|---|---|---|
| `fast-pgsr-old` | COLMAP、训练、渲染、导出 | 3.7.13 | 1.12.1+cu116 | 1.21.6 | 0.4.0 |
| `Fast-Sam2` | 分割、ArUco 对齐、标签生成 | 3.10.20 | 2.7.1+cu118 | 2.2.6 | 4.1.0 |

两个环境分别冻结版本，不能合并安装到同一个环境。`run_batch.sh` 中的 FastGS/fast-pgsr 不作为此次迁移的环境依据。

## requirements 文件

- `requirements.txt`：默认安装训练环境，引用 `requirements-fast-pgsr-old.txt`。
- `requirements-fast-pgsr-old.txt`：训练环境的固定版本 Python 包。
- `requirements-fast-pgsr-old-source.txt`：四个项目 CUDA 扩展及 PyTorch3D 0.7.1 源码安装项。
- `requirements-Fast-Sam2.txt`：分割/对齐环境的固定版本 Python 包。
- `requirements-Fast-Sam2-source.txt`：本项目 GroundingDINO / SAM2 源码安装项。

普通包清单来自对应环境的完整 Conda 导出中 pip 部分，Conda 管理的包由 `.conda.yml` 保存。完整原始 `pip freeze --all`、Conda YAML 和包元数据保存在 `environments/exports-4090/`，原六项 `requirements.txt` 已备份为其中的 `requirements-project-original.txt`。

两份普通包清单包含环境中实际安装的辅助包和历史 CUDA runtime 包，不是人工删减的最小依赖清单。所有 Python 包版本按源环境保留；本地源码包移到单独清单，避免绝对路径失效及编译时找不到 PyTorch。

## 在新 Linux 服务器安装

将整个项目代码复制到新服务器，以下命令都在**项目根目录**执行。需要可用的 Conda、Git、C++ 编译器和 CUDA toolkit / nvcc。源服务器存在 `/usr/local/cuda-11.8`；编译环境应结合目标显卡、PyTorch 和源码扩展确认，不能只安装 NVIDIA 驱动。

### 1. 训练环境

```bash
conda env create -f environments/environment-fast-pgsr-old.conda.yml
conda activate fast-pgsr-old
python -m pip install -r requirements.txt
# 先安装 PyTorch 等普通依赖，再编译源码包：
python -m pip install --no-build-isolation --no-deps -r requirements-fast-pgsr-old-source.txt
python -m pip check
python -c "import torch, diff_gaussian_rasterization_fastgs, diff_plane_rasterization, simple_knn._C, fused_ssim, pytorch3d; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), pytorch3d.__version__)"
```

PyTorch3D 在源环境中报告版本 0.7.1，但其安装记录指向项目外的旧路径，原源码目录已不可访问。源码清单使用已核实的官方 v0.7.1 提交 `995b60e3b99faa1ee1bcdbe244426d54d98a7242` 的 codeload 归档，不能保证包含原目录中可能存在的未提交修改。目标机安装时需要下载该归档，或预先准备相应源码。

### 2. 分割与对齐环境

```bash
conda env create -f environments/environment-Fast-Sam2.conda.yml
conda activate Fast-Sam2
python -m pip install -r requirements-Fast-Sam2.txt
python -m pip install --no-build-isolation --no-deps -r requirements-Fast-Sam2-source.txt
python -m pip check
python -c "import torch,cv2,numpy,pycolmap; print(torch.__version__,cv2.__version__,numpy.__version__,pycolmap.__version__); assert hasattr(cv2.aruco,'ArucoDetector'); assert hasattr(pycolmap,'Sim3d')"
python -c "import runpy; m=runpy.run_path('scripts/mask_generation/run_rubik_scene_auto.py',run_name='migration_check'); m['load_base'](); print('mask imports OK')"
```

源环境的 groundingdino editable 安装记录指向另一个项目目录，因此清单改用本项目内源码。分割脚本也从本项目 `third_party/Grounded-SAM-2` 加载源码。源码扩展需要在目标机重新编译，不能复用另一台机器的 `.so` 文件。

### 3. 外部工具、模型资源和大豆路径

`requirements.txt` 无法安装 FFmpeg、COLMAP 可执行程序、NVIDIA 驱动和系统 CUDA 编译工具。`run_batch_gpu1.sh` 的抽帧直接调用当前 Shell 中的 FFmpeg；COLMAP 在 `fast-pgsr-old` 环境中运行。安装后检查：

```bash
ffmpeg -version
conda run -n fast-pgsr-old colmap -h
nvcc --version
```

SAM2 / GroundingDINO 权重及本地 BERT tokenizer 资源需要另外准备；本次没有下载模型资源。分割脚本设置离线模式，不能仅凭安装包清单自动获得这些资源。

本地适配后的入口默认 GPU 0、一个魔方，路径默认相对项目根目录（`videos/`、`data/`、`output/`、`logs/`），已移除旧服务器绿豆绝对路径。使用 `VIDEO_DIR`、`DATA_ROOT`、`OUTPUT_ROOT` 配置实际大豆目录，`GPU_ID` 可覆盖；若调度器已设置 `CUDA_VISIBLE_DEVICES` 则保留。具体命令与旧产物处理见 [SINGLE_RUBIK.md](SINGLE_RUBIK.md)。原始导出文件是历史快照，不能据此认为当前本地业务代码与服务器一致。

## 核验范围

源服务器上两个环境均执行 `pip check` 通过；已重新读取 Python、PyTorch、NumPy、OpenCV、pycolmap、PyTorch3D 版本。训练 CUDA 扩展的导入记录及本次分割入口导入检查输出保存在 `environments/exports-4090/`。

requirements 引用、源码路径和固定版本与导出文件做了一致性检查。2026-10-16～17 已在新的 RTX 3090 容器上创建两个环境、重新编译源码扩展，两个环境的 `pip check` 均通过，两个视频均完成 30000 次训练。后续渲染、导出验收及迁移中的实际修复以 [完整迁移文档](DEPLOYMENT_3090_FULL_GUIDE.md) 为准。
