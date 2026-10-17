# Fast-PGSR 大豆重建：3090 部署与再次迁移流程

本文记录本次实际部署，正确入口是 `run_batch_gpu1.sh`。环境、模型资源、视频和重建输出要分别准备；仅复制源码和安装 requirements 不能替代模型权重或 COLMAP。

实施时间：2026-10-16 晚至 2026-10-17（Asia/Shanghai）。

**当前状态：已进一步完成分割源头修复，两个视频的304帧训练用掩码全部重新生成，并已更新服务器正式 `data/` 目录的掩码、语义初始化和绑定。见第14节。** 第13节的30000次模型是使用旧掩码训练后清理的历史结果，本次没有重新训练；后续必须使用新输出目录训练，不能直接复用旧模型。第10节保留首次部署记录。

## 本次目标与输入

| 项目 | 本次配置 |
|---|---|
| SSH | `ssh -p 32596 root@172.26.14.81` |
| 项目 | `/root/Fast-PGSR` |
| 系统 | Ubuntu 22.04 容器 |
| GPU | RTX 3090，24GB，可见编号 0 |
| 编译工具包 | `/usr/local/cuda-11.8` |
| Conda | `/root/miniconda3` |
| 数据/输出/日志 | 项目的 `data/`、`output/`、`logs/` |

从本地 `E:\大豆拍摄视频\2026.9.18\LWL` 按文件名升序选择前两个视频：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `A1087-1-R3-0918.mp4` | 96,536,079 | `fc1429acac1a11cfa87eb8df71ba222555e82b00ecb2c67667aa8d5af2547f81` |
| `A1087-2-R3-0918.mp4` | 109,543,022 | `4dd17c8276690aef3bc68e21b201f1b226c695044d80cdaa34c13218fe052f2a` |

实际视频已上传到 `/root/Fast-PGSR/videos/` 并核对哈希。密码不写入本文或部署脚本。

## 1. 检查新服务器

```bash
hostname
nvidia-smi
df -h /root /tmp
free -h
command -v conda ffmpeg colmap nvcc gcc g++ git
ls -ld /usr/local/cuda*
```

确认目标目录是否已有其他项目或训练结果；存在时先备份，再同步代码。本次 `/root/Fast-PGSR` 原先不存在。其他服务器不要直接覆盖现存训练目录。

## 2. 传输本地项目与视频

本次使用 Python Paramiko 的 SSH/SFTP：打包本地项目，排除 `.git`、数据、输出、日志、视频、Python 缓存和编译产物；传到 `/root/fast-pgsr-deployment/project.tar.gz` 后解压至项目目录。视频单独上传，避免把整个本地视频目录复制到服务器。

再次迁移也可在 Windows PowerShell 使用 OpenSSH（先设置实际本地路径与目标连接）：

```powershell
$project = 'C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\deployment\Fast-PGSR'
tar -czf Fast-PGSR-code.tar.gz --exclude=.git --exclude=__pycache__ --exclude=build --exclude=dist --exclude=data --exclude=output --exclude=logs --exclude=videos --exclude='*.so' --exclude='*.pyc' --exclude='*.o' --exclude='*.egg-info' --exclude=checkpoints --exclude=gdino_checkpoints --exclude=bert-base-uncased -C $project .
scp -P 32596 Fast-PGSR-code.tar.gz root@172.26.14.81:/root/
scp -P 32596 'E:\大豆拍摄视频\2026.9.18\LWL\A1087-1-R3-0918.mp4' 'E:\大豆拍摄视频\2026.9.18\LWL\A1087-2-R3-0918.mp4' root@172.26.14.81:/root/Fast-PGSR/videos/
```

先在远端建立目录再执行上述视频上传：

```bash
mkdir -p /root/Fast-PGSR/videos
tar -xzf /root/Fast-PGSR-code.tar.gz -C /root/Fast-PGSR
sha256sum /root/Fast-PGSR/videos/*.mp4
bash -n /root/Fast-PGSR/run_batch_gpu1.sh
```

所有 shell 文件需使用 LF 换行；Windows CRLF 会导致 Bash 选项或路径解析错误。

## 3. 安装 Ubuntu 系统依赖与 Miniconda

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ffmpeg build-essential cmake ninja-build pkg-config wget curl ca-certificates git libgl1 libglib2.0-0 libeigen3-dev libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libgmp-dev libmpfr-dev

curl -fL --retry 3 https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -o /root/miniconda-installer.sh
bash /root/miniconda-installer.sh -b -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
conda config --set auto_activate_base false
```

目标已有可用 Conda 时，可使用现有安装并修改后续路径，不要再次覆盖其目录。

## 4. 创建两个独立 Conda 环境

```bash
cd /root/Fast-PGSR
source /root/miniconda3/etc/profile.d/conda.sh
conda env create -f environments/environment-fast-pgsr-old.conda.yml
conda env create -f environments/environment-Fast-Sam2.conda.yml
```

| 环境 | Python | PyTorch | 负责阶段 |
|---|---|---|---|
| `fast-pgsr-old` | 3.7.13 | 1.12.1+cu116 | COLMAP 调用、训练、渲染、导出 |
| `Fast-Sam2` | 3.10.20 | 2.7.1+cu118 | 分割、ArUco、语义初始化 |

使用随项目提供的 Conda YAML 和两组 requirements，不要用一个环境安装两套 PyTorch/NumPy。

## 5. 安装 Python 包与源码扩展

```bash
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export MAX_JOBS=4
# 此处 8.6 对应本次 RTX 3090；换 GPU 时需调整。
export TORCH_CUDA_ARCH_LIST=8.6
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

conda activate fast-pgsr-old
python -m pip install --timeout 120 --retries 3 -r requirements.txt
python -m pip install --no-build-isolation --no-deps -r requirements-fast-pgsr-old-source.txt
python -m pip check

conda activate Fast-Sam2
python -m pip install --timeout 120 --retries 3 -r requirements-Fast-Sam2.txt
python -m pip install --no-build-isolation --no-deps -r requirements-Fast-Sam2-source.txt
python -m pip check
```

先装 PyTorch 等普通包，再装源码包。四个高斯/CUDA 扩展、PyTorch3D、GroundingDINO 和 SAM2 不能依赖其他机器的旧 `.so`；需针对目标环境重新构建。

本次 GitHub 的 `git clone` 出现 TLS 中断，因此 PyTorch3D 的源码依赖改用官方同一固定提交的 codeload 归档 URL。提交仍为 `995b60e3b99faa1ee1bcdbe244426d54d98a7242`，对应 0.7.1。

### 下载缓慢时使用 wheelhouse

本次普通 PyPI 下载慢，改用清华镜像；requirements 中保留 PyTorch CUDA wheel 的官方索引。PyTorch 下载源仍慢时，可先下载**相同版本、Python ABI、Linux 架构**的 wheel 放入本地 wheelhouse：

```bash
mkdir -p /root/fast-pgsr-deployment/wheelhouse
python -m pip config --user set global.find-links /root/fast-pgsr-deployment/wheelhouse
```

本次 `Fast-Sam2` 的 torch 2.7.1+cu118 wheel 从 4090 的 pip 缓存中提取并传输；训练 torch 1.12.1+cu116 通过官方原始 CDN 分段下载，并校验官方响应 ETag 对应的 MD5 后合并使用。没有切换为 CPU wheel 或改变 CUDA 版本。

本次实际进一步把六个框架 wheel（两个环境各自的 torch、torchvision、torchaudio）预先装好，再为普通包生成**仅去掉额外 PyTorch 索引、版本完全不变**的临时清单。这样可避免 NumPy、NVIDIA runtime 等普通包从额外索引返回的慢速 PyPI 地址下载：

```bash
# 当前激活的环境必须与 wheel 的 cp37/cp310 标签一致；不要跨环境混装。
# 先用 python -m pip install --no-deps /实际路径/对应版本.whl 安装框架的三个 wheel。
# 以下以 Fast-Sam2 为例；训练环境替换为 requirements-fast-pgsr-old.txt。
python - requirements-Fast-Sam2.txt /tmp/requirements-Fast-Sam2-mirror.txt <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding='utf-8')
Path(sys.argv[2]).write_text('\n'.join(line for line in text.splitlines() if not line.startswith('--extra-index-url')) + '\n', encoding='utf-8')
PY
python -m pip install --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r /tmp/requirements-Fast-Sam2-mirror.txt
```

没有提前安装相应 CUDA 框架 wheel 时，仍应使用原 requirements 中的官方 CUDA 索引，不能直接去掉索引后期待普通镜像提供 `+cu116` / `+cu118` 包。

本次安装使用的 wheel 校验记录、目标服务器完整环境导出、系统包版本及运行验证输出见 `environments/exports-3090-32596/`。这些是实际目标环境快照；其中原始 freeze 的本地路径仅用于审计，再次安装应使用项目根目录的可迁移清单。

## 6. 编译支持 CUDA 的 COLMAP

系统普通 COLMAP 包不一定支持 CUDA；入口在 GPU 模式下要求 `colmap -h` 输出包含 `with CUDA`。本次编译版本为 3.9.1：

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
mkdir -p /root/fast-pgsr-deployment/src
cd /root/fast-pgsr-deployment/src
curl -fL --retry 3 https://codeload.github.com/colmap/colmap/tar.gz/refs/tags/3.9.1 -o colmap-3.9.1.tar.gz
tar -xzf colmap-3.9.1.tar.gz
cmake -S colmap-3.9.1 -B colmap-3.9.1/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build colmap-3.9.1/build -j 4
cmake --install colmap-3.9.1/build
colmap -h
```

`CMAKE_CUDA_ARCHITECTURES=86` 对应本次 3090。换显卡时需要设置其架构，并确认当前 CUDA toolkit 支持它。

## 7. 准备离线模型资源

以下相对项目路径必须存在：

```text
third_party/Grounded-SAM-2/
  gdino_checkpoints/groundingdino_swint_ogc.pth
  checkpoints/sam2.1_hiera_base_plus.pt
  bert-base-uncased/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    vocab.txt
```

本次从已核验的 4090 项目转移上述 7 个文件，总计约 1.46 GB，源/目标 SHA-256 一致。脚本默认离线加载 BERT，因此只有 vocab/tokenizer 而没有 BERT 模型权重也会失败。再次迁移时可从本次部署项目复制这些固定资源，或者从模型官方来源重新准备，并保存校验和。

从本次3090服务器制作资源包（不包含训练数据或输出）：

```bash
tar -czf /root/Fast-PGSR-models.tar.gz -C /root/Fast-PGSR/third_party/Grounded-SAM-2 \
  gdino_checkpoints/groundingdino_swint_ogc.pth checkpoints/sam2.1_hiera_base_plus.pt bert-base-uncased
```

将资源包复制到新服务器后，在新项目相同目录解压：

```bash
tar -xzf /实际传输位置/Fast-PGSR-models.tar.gz -C /root/Fast-PGSR/third_party/Grounded-SAM-2
cd /root/Fast-PGSR
python3 - <<'PY'
import hashlib,json
from pathlib import Path
base=Path('third_party/Grounded-SAM-2')
for item in json.loads(Path('environments/exports-3090-32596/model-resources-sha256.json').read_text()):
    p=base/item['path']
    assert p.stat().st_size==item['bytes'],str(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()==item['sha256'],str(p)
    print('OK',p)
PY
```

## 8. 安装后验证

```bash
cd /root/Fast-PGSR
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export QT_QPA_PLATFORM=offscreen
conda run --no-capture-output -n fast-pgsr-old python -c "import torch,pytorch3d,diff_gaussian_rasterization_fastgs,diff_plane_rasterization,simple_knn._C,fused_ssim; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
conda run --no-capture-output -n Fast-Sam2 python -c "import torch,cv2,numpy,pycolmap; assert hasattr(cv2.aruco,'ArucoDetector'); assert hasattr(pycolmap,'Sim3d'); print(torch.__version__,cv2.__version__,numpy.__version__,pycolmap.__version__)"
conda run --no-capture-output -n Fast-Sam2 python -c "import runpy; m=runpy.run_path('scripts/mask_generation/run_rubik_scene_auto.py',run_name='migration_check'); m['load_base'](); print('MASK_IMPORT_OK')"
conda run -n fast-pgsr-old python -m pip check
conda run -n Fast-Sam2 python -m pip check
colmap -h
ffmpeg -version
bash -n run_batch_gpu1.sh
```

确认输出 `torch.cuda.is_available()` 为 True，分割基础脚本可导入；通过导入并不等于已验证实际训练与识别效果，仍需下一步视频实跑。

## 9. 执行两个视频的完整重建

```bash
cd /root/Fast-PGSR
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export QT_QPA_PLATFORM=offscreen
export RENDER_VOXEL_SIZE=0.02
bash run_batch_gpu1.sh
```

入口默认 GPU 0、单魔方、2.5 FPS、30000 次训练、strict 导出，按株执行抽帧 → COLMAP → MASK → ALIGN → LABEL → TRAIN → EXPORT → RENDER。导出默认启用 `TARGET_SPATIAL_RADIUS=0.8` 的单株空间清理，渲染直接使用清理后的目标PLY。保持这两个视频独占 `videos/` 时，默认队列恰好为本次两个场景。

本次服务器系统内存为24GiB，使用 `RENDER_VOXEL_SIZE=0.02`。脚本保留历史默认0.01；0.01在本次输入上即使屏蔽背景，网格提取阶段仍超出24GiB。该参数只影响 TSDF 网格，不影响训练和高斯目标导出。体素单位为 ArUco 对齐后的归一化坐标单位，不能直接写作米。

SSH 断开后继续运行，可建立项目外的启动脚本：

```bash
mkdir -p /root/fast-pgsr-deployment
cat > /root/fast-pgsr-deployment/run-reconstruction.sh <<'SH'
#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen
export RENDER_VOXEL_SIZE=0.02
cd /root/Fast-PGSR
bash run_batch_gpu1.sh
status=$?
echo "$status" > /root/fast-pgsr-deployment/run-reconstruction.exit-code
exit "$status"
SH
nohup bash /root/fast-pgsr-deployment/run-reconstruction.sh > /root/fast-pgsr-deployment/run-reconstruction.log 2>&1 < /dev/null &
echo "PID=$!"
```

检查进度和最终退出码：

```bash
tail -c 5000 /root/fast-pgsr-deployment/run-reconstruction.log | tr '\r' '\n' | tail -25
cat /root/fast-pgsr-deployment/run-reconstruction.exit-code
nvidia-smi
```

退出码文件在结束后才写入；训练进度含回车字符，使用按字节截取可避免 `tail -n` 一次打印整段训练进度。本次启动/安装/验证日志统一位于 `/root/fast-pgsr-deployment/`，各阶段业务日志位于项目 `logs/single-rubik-*`。

默认重试复用已完成中间产物。上游输入发生改变时，应使用新的数据目录；明确需要覆盖自己的既有产物时才设置 `FORCE=1`。不要为了通过检查而降低标记几何质量阈值或跳过语义绑定验证。

## 10. 首次实跑记录（13:19历史结果，已由第13节清理结果替代）

### 已验证的最小修复

`align_aruco_refine.py` 原始检测器在第一段视频的 142 张去畸变图像中没有检出 ArUco，无法满足至少两个注册视角的要求。修复仅扩大自适应阈值窗口（3～65，步长4），启用亚像素角点细化，并在无检测时尝试2倍插值放大；检测角点除以2后再送入原相机模型。字典、目标 ID、最少视角数和全部几何质量阈值不变。

独立实测在 `frame_000083.jpg`、`frame_000090.jpg` 检出 ID 0，原三角化与 Sim3 拟合通过：RMSE 0.012291、最大角点误差 0.017185（标记归一化边长为1）。这验证了回退角点坐标转换和实际几何效果。

第二段视频也通过原几何阈值：RMSE 0.004770，最大角点误差 0.006273。

真实训练还发现 `utils/training_cache.py` 的两次 `torch.argsort(..., stable=True)` 不兼容训练环境 PyTorch 1.12。改为 `torch.sort(..., stable=True).indices`，保留距离优先、角度次优先及稳定处理相等值的顺序。目标 3090 上的 `scripts/test_gpu_optimizations.py` 全部9项测试通过，包括与原 NumPy `lexsort` 邻近视角顺序一致的测试。

两个失败训练尝试仅产生初始化输出，已移至 `/root/fast-pgsr-deployment/failed-stable-sort-20261017-002116/`，随后重试。没有删除或重算已成功的抽帧、COLMAP、Mask、对齐和语义初始化。

每个成功场景应至少具有：

```text
output/<scene>/point_cloud/iteration_30000/point_cloud.ply
output/<scene>/point_cloud/iteration_30000/semantic_state.npz
output/<scene>/mesh/...
output/<scene>/target_exports/strict/point_cloud.ply
```

最终供植株重建使用的是 `target_exports/strict/point_cloud.ply`，包括目标植株和一个魔方；完整场景模型保留在 `point_cloud/iteration_30000/`。检查日志退出码、文件大小、目标类别计数及视觉预览，不能仅凭“文件存在”判定成功。

### 渲染收尾修复与断点续跑

两段训练分别于 2026-10-17 00:59:31、01:39:11 完成，但首次 TSDF 融合被系统杀死。服务器 cgroup v1 的 `memory.limit_in_bytes=25769803776`（24GiB），`memory.oom_control` 记录 `oom_kill 2`，峰值超过该上限。原 `render.py` 只检查透明通道 `view.mask`；本次 JPEG 没有透明通道，虽然已加载 `view.object_mask`，背景仍被积分，导致细体素 TSDF 占用过大。

最小修复：仅在 `semantic_pipeline=True` 且目标掩码存在时，将背景标签0的深度清零，保留标签1植株和标签2魔方。普通非语义渲染行为保留。另取消全部 CPU 深度的 `torch.stack` 复制，逐张在 CPU 上应用掩码，不再反复送 GPU。该修复使第一段视频完成全部142帧融合，但0.01体素下提取网格时仍触发内存限制。因此在 `run_batch_gpu1.sh` 增加 `RENDER_VOXEL_SIZE` 环境变量，保留默认0.01，本次显式设置0.02以适配24GiB系统内存。代价是网格空间采样变粗；训练权重、语义状态和高斯导出精度不变。

修复同步到本地和 3090 后，仅重新执行 Render/Export：

```bash
cd /root/Fast-PGSR
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen
RUN_EXTRACT=0 RUN_COLMAP=0 RUN_MASK=0 RUN_ALIGN=0 RUN_LABEL=0 RUN_TRAIN=0 \
RUN_RENDER=1 RUN_EXPORT=1 FORCE=1 RENDER_VOXEL_SIZE=0.02 bash run_batch_gpu1.sh
```

这里 `FORCE=1` 只作用于已启用的渲染/导出阶段；会重建对应的 `mesh/` 和 `target_exports/`，不会重新训练。不要把这些开关省略后执行 `FORCE=1`，否则会连同上游阶段一起重做。

完整训练日志：`/root/Fast-PGSR/logs/single-rubik-20261017-002121-GGy5nE/`。0.01体素补跑失败记录：`/root/Fast-PGSR/logs/single-rubik-20261017-131125-zaY6Kn/`。部署、安装和校验记录：`/root/fast-pgsr-deployment/`。

### 本次训练结果

| 场景 | 注册图像 | 训练迭代 | 完整模型高斯数 | 训练耗时 | 训练结束时间 |
|---|---:|---:|---:|---|---|
| A1087-1-R3-0918 | 142/142 | 30000 | 92257 | 36分58.9秒 | 2026-10-17 00:59:31 |
| A1087-2-R3-0918 | 162/162 | 30000 | 105214 | 35分18.3秒 | 2026-10-17 01:39:11 |

时间按服务器日志，时区 Asia/Shanghai。日志中的训练视角 PSNR 分别为 23.3409、22.2405；这是训练集指标，不能当作独立测试集评估。

### 最终渲染与导出验收

本次0.02体素的成功补跑日志：`/root/Fast-PGSR/logs/single-rubik-20261017-131529-wWPMyN/`。两个场景的 `Render_Correct/`、`Export_Correct/` 均有成功日志；主日志报告“失败阶段数=0”。退出码文件 `/root/fast-pgsr-deployment/retry-render-export-memory.exit-code` 内容为 `0`。

| 场景 | 严格目标高斯数 | 其中植株 | 其中魔方1 | RGB渲染帧 | 后处理网格顶点/三角形 |
|---|---:|---:|---:|---:|---|
| A1087-1-R3-0918 | 37437 | 36642 | 795 | 142 | 164250 / 320069 |
| A1087-2-R3-0918 | 35887 | 35345 | 542 | 162 | 49906 / 94057 |

两个严格目标 PLY 均仅包含语义ID1（植株）和ID2（一个魔方），背景、未知、第二魔方计数均为0。已逐项比对62项高斯属性，严格导出文件的每一项都等于原完整模型中 `selection_indices.npy` 选取的对应值。PLY/语义状态顺序校验通过，坐标全部有限；每个场景的四份网格均可读且顶点/三角形非空，RGB渲染数量等于注册图像数量。数据与语义绑定检查也通过。

服务器上的两个结果根目录：

```text
/root/Fast-PGSR/output/A1087-1-R3-0918/
/root/Fast-PGSR/output/A1087-2-R3-0918/
```

每个结果根目录中：

| 相对路径 | 用途 |
|---|---|
| `target_exports/strict/point_cloud.ply` | 最终植株+魔方高斯模型，保留完整62项属性 |
| `target_exports/strict/semantic_state.npz` | 与该模型匹配的语义状态 |
| `target_exports/strict/export_manifest.json` | 导出模式、类别数和配置 |
| `target_exports/strict/selection_indices.npy` | 在完整高斯模型中的选择索引 |
| `target_exports/plant_only/point_cloud.ply` | 仅植株语义类别 |
| `target_exports/rubik_1_only/point_cloud.ply` | 仅魔方语义类别 |
| `point_cloud/iteration_30000/` | 完整训练模型及语义状态 |
| `mesh/mesh_color.ply` | 未做连通分量筛选的带颜色网格 |
| `mesh/tsdf_fusion_post.ply` | 按原逻辑保留主要连通分量的后处理网格 |
| `train/ours_30000/renders/` | 各注册视角RGB渲染图 |
| `train/ours_30000/renders_depth/`、`renders_normal/` | 深度可视化、法线图；后者同处 `train/ours_30000/` |

后处理网格保留主要连通分量，细叶或断开的结构可能被筛掉；不能用它代替严格目标高斯模型来判断所有叶片是否完整。分离导出的 `unknown_target_candidates/` 只是候选诊断结果，不属于 strict 目标。

目标高斯 PLY 文件校验和：

```text
A1087-1-R3-0918  36d53d98b91d090c1b590bf1f7f6a8176ebcdc452d7df109e42c5f262de6c4c5  9285906 bytes
A1087-2-R3-0918  f2c7eb0c27328dcaa795cc8fe60319287a3e581468f7ca7a55e18c6ae70af1d7  8901506 bytes
```

验收记录保存在 `/root/fast-pgsr-deployment/reconstruction-results.json`，随代码归档的副本是 `environments/exports-3090-32596/reconstruction-results.json`。详细审计日志为 `/root/fast-pgsr-deployment/final_audit_3090.sh.log`，包含 `RECONSTRUCTION_AUDIT_PASSED` 和 `FINAL_MESH_AND_RENDER_AUDIT_PASSED`。

本地结果副本目录：

```text
C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\reconstruction-results\3090-32596-20261017\
```

其中按场景保存完整30000次高斯模型、全部目标导出、后处理网格、三维高斯位置预览、首/中/末视角RGB渲染样本和运行日志。两个 strict PLY 在下载后再次核对SHA-256。较大的原始网格和其余渲染图保留在服务器；本地副本不是服务器输出目录的逐文件镜像。三维预览只显示高斯中心与近似颜色，不是表面渲染。

## 11. 再次迁移时如何验收

运行结束应满足：退出码0；每个场景训练日志含 `Training complete.` 和 `bind-trained OK`；渲染日志位于 `Render_Correct/`，导出日志位于 `Export_Correct/`；最终 PLY 与语义状态均通过校验。仅观察 GPU 空闲无法区分成功退出和进程崩溃。

```bash
cd /root/Fast-PGSR
for scene in A1087-1-R3-0918 A1087-2-R3-0918; do
  conda run --no-capture-output -n fast-pgsr-old python scripts/validate_semantic_inputs.py bind-trained \
    --scene "/root/Fast-PGSR/data/$scene" --model "/root/Fast-PGSR/output/$scene" --iteration 30000 --rubik-count 1
done
conda run --no-capture-output -n fast-pgsr-old python - <<'PY'
import json
from pathlib import Path
from scripts.export_target_gaussians import validate_input
for name in ['A1087-1-R3-0918','A1087-2-R3-0918']:
    p=Path('output')/name/'target_exports/strict'
    ply,xyz,sem=validate_input(str(p/'point_cloud.ply'),str(p/'semantic_state.npz'))
    m=json.loads((p/'export_manifest.json').read_text())
    assert m['selected_plant_count']>0 and m['selected_rubik_1_count']>0
    assert m['selected_background_count']==m['selected_unknown_count']==m['selected_rubik_2_count']==0
    print(name,len(xyz),m['selected_plant_count'],m['selected_rubik_1_count'])
PY
```

查看 RGB 渲染、三维点云和网格。只按语义类别执行旧 `strict` 导出不能纠正邻株误标。第13节记录了本次邻株/离群簇的诊断、清理和新结果；入口现在在语义选择后增加空间实例筛选。它不等同于逐叶人工标注，后续若用于精确表型测量，仍需复核目标完整性及几何质量。

`selection_debug.ply` 是便于检查的坐标/颜色诊断点云，不能代替包含62项高斯属性的 `point_cloud.ply`。不要用普通点云转换工具重写最终高斯 PLY，否则可能丢失球谐颜色、透明度、尺度和旋转参数。`semantic_state.npz`、`selection_indices.npy` 和 `export_manifest.json` 应与高斯 PLY 一起保存。

## 12. 结果取回与下次迁移清单

在 Windows PowerShell 中取回某场景的完整输出（包括体积较大的原始网格和全部渲染图）：

```powershell
$save = 'C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\reconstruction-results\3090-32596-20261017'
New-Item -ItemType Directory -Force -Path $save
scp -r -P 32596 root@172.26.14.81:/root/Fast-PGSR/output/A1087-1-R3-0918 $save
scp -r -P 32596 root@172.26.14.81:/root/Fast-PGSR/output/A1087-2-R3-0918 $save
```

下次只部署代码和环境时，复制源码、两组 requirements、Conda YAML、模型资源及本文；重新编译 CUDA 扩展和 COLMAP。视频按实际任务单独传输，旧 `data/`、`output/`、日志和 `.so` 不混入代码迁移包。若要接续既有场景，则额外备份对应数据和完整模型及语义绑定文件，保持目录与引用一致。

当前模型 `cfg_args` 中保留 `/root/Fast-PGSR` 绝对路径。将已训练模型搬到其他目录后渲染，需要通过 `-s /新项目/data/<scene>` 指定实际源路径，并确认语义初始化、掩码引用可用；最简单的复现方式是保持 `/root/Fast-PGSR` 布局。本文训练路径与服务器端口均为本次实例记录，换服务器时应修改 SSH、Conda、CUDA 和项目根目录配置。

## 13. 邻株与离群点：定位、最小修复及当前结果

### 原因与证据

主要来源是分割掩码误把邻株标为目标，而原导出器只按语义类别选择，缺少单株实例的空间约束。第一段末帧 `frame_000142.jpg` 的左侧邻株、第二段中间帧 `frame_000082.jpg` 的右后方邻株均进入绿色植株掩码。将被剔除的三维点投回原图，红点落在这些邻株上；绿点对应保留的目标植株，蓝点对应一个魔方。

![第一段末帧：左为原掩码，右为保留/剔除点的投影](docs/spatial-cleanup-20261017/A1087-1-R3-0918-frame_000142.jpg)

![第二段中间帧：右后方邻株误标](docs/spatial-cleanup-20261017/A1087-2-R3-0918-frame_000082.jpg)

第一段被剔除的4442个植株标签高斯中，3629个源点在语义初始化阶段已经标为植株，813个原为unknown；第二段10831个中，7675个初始化已是植株，3156个原为unknown。因此残留并非导出时打乱索引或损坏NPZ，而是上游误标被训练继承/确认后，语义导出照单保留。初始标签追溯记录为 `docs/spatial-cleanup-20261017/semantic-origin.json`。

两个明显邻株簇与主株的最近距离分别为5.596/9.669和3.444/6.911个标记归一化单位；与主株之间存在清晰空间间隔。投影统计只作辅助证据，不是经过遮挡判定的可见性统计。原渲染还直接读取完整训练模型，因此即使只修改导出，也无法自动清理旧网格。

### 实施的最小改动

1. `scripts/export_target_gaussians.py` 新增可选 `--spatial-radius`。先验证语义与PLY，再按目标类别进行三维密度聚类，保留主实例，剔除远处邻株簇和噪声；不移动点、不优化模型、不改标签。普通直接调用默认0，保留旧API行为。
2. `run_batch_gpu1.sh` 默认 `TARGET_SPATIAL_RADIUS=0.8`，在ArUco边长=1坐标下采用DBSCAN半径0.8、最少5点。植株允许保留距主簇1.6以内的稠密断开叶片，避免误删因遮挡分离的目标叶片；不沿其他簇递归扩张。魔方只保留自身主簇。若最大簇不足次大簇的1.5倍，则报错停止，不猜测目标。
3. 批处理阶段改为 `TRAIN → EXPORT → RENDER`；`render.py` 通过 `--target_ply` 加载同一份清理后的高斯及其语义状态，生成对应的RGB/深度/法线和网格。原训练文件保持原样。
4. 每份导出的 `export_manifest.json` 保存 `spatial_cleanup` 参数、各簇规模、距主簇距离和保留/剔除状态。62项高斯属性、原行序、选择索引和匹配语义校验和均保留。

第一段另保留了距主簇1.071的186个断开叶片高斯；没有为获得“单连通分量”而简单丢掉它们。第二段主株与最近邻株间隔3.444，大于1.6的附着阈值。阈值在这两个已对齐场景中通过投影和三维预览核验；新场景、不同标记尺度或接触重叠的植株不能无检查套用。

本节记录的13:39修复当时只作用于最终交付结果，没有改写上游掩码。随后又实施了第14节的MASK源头修复，并重新生成了训练用掩码和绑定；当前应使用第14节的训练命令。导出空间清理保留为附加保护，不再作为源头修复的替代。

### 当前最终结果

| 场景 | 清理前目标高斯 | 剔除邻株/离群高斯 | 保留植株 | 保留魔方 | 最终目标总数 |
|---|---:|---:|---:|---:|---:|
| A1087-1-R3-0918 | 37437 | 4442 | 32200 | 795 | 32995 |
| A1087-2-R3-0918 | 35887 | 10831 | 24514 | 542 | 25056 |

两份最终目标PLY路径保持不变，内容已更新：

```text
/root/Fast-PGSR/output/A1087-1-R3-0918/target_exports/strict/point_cloud.ply
/root/Fast-PGSR/output/A1087-2-R3-0918/target_exports/strict/point_cloud.ply
```

当前SHA-256：

```text
A1087-1-R3-0918  4cf764cf49d52fed7b3809fd35dbee352d5491b03cfbe85bd1ed35a727e92d0c  8184290 bytes
A1087-2-R3-0918  66a1f9f904ad319a76525ed2a5c1311a299d30f279e543325227b77ab4688d8d  6215418 bytes
```

![第一段清理后：目标大豆和一个魔方，高斯中心预览](docs/spatial-cleanup-20261017/A1087-1-R3-0918-cleaned.png)

![第二段清理后：目标大豆和一个魔方，高斯中心预览](docs/spatial-cleanup-20261017/A1087-2-R3-0918-cleaned.png)

当前输出中的 `plant_only`、`rubik_1_only` 同样经过筛选；不再生成unknown候选目录。RGB渲染分别为142、162帧，使用目标高斯数32995、25056；`render_target.json` 记录所加载的目标模型路径及数量。原始网格也已重建，仅在目标模型空间范围内，没有重新带入已剔除邻株。四份网格全部非空可读。

`mesh/tsdf_fusion_post.ply` 仍使用旧的“保留主要连通分量”后处理，可能只留下部分连通叶片、丢弃独立的魔方；第二段TSDF网格也没有完整恢复魔方表面。因此本次满足“目标大豆+一个魔方”的权威交付文件是上面的完整属性高斯PLY及匹配NPZ，不能把后处理网格当作完整目标模型。网格可作辅助可视化，本次未改变表面重建算法。

本地当前结果副本：

```text
C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\reconstruction-results\3090-32596-20261017-cleaned\
```

包含两个场景的目标模型、完整原训练模型、后处理网格、渲染样本、三维预览、掩码/投影证据及日志。旧本地结果保留在不含 `-cleaned` 后缀的目录，便于对比。服务器清理前的目标导出、网格及渲染备份位于 `/root/fast-pgsr-deployment/before-spatial-cleanup-20261017-133549/`。

### 验收与复现

清理运行退出码：`/root/fast-pgsr-deployment/spatial-cleanup.exit-code` 为0；审计退出码：`spatial-cleanup-audit.exit-code` 为0。完整运行记录：`/root/fast-pgsr-deployment/cleanup_target_outputs_3090.sh.log`。批次业务日志为 `/root/Fast-PGSR/logs/single-rubik-20261017-133549-Se5g1q/`。原始两个训练PLY及两个NPZ的SHA-256与清理前完全一致，没有重训或覆盖检查点。

CPU空间筛选测试覆盖：近处断开叶片保留、远邻株剔除、稀疏噪声剔除、独立魔方保留、尺度变换、无明显主实例拒绝及非法半径拒绝。导出器原51项回归检查全部通过。实际输出逐字段验证62项属性与原模型所选行相同，NPZ顺序和语义校验和通过；类别仅为植株1与魔方2，另核验了保留点的目标空间范围、每类导出与合并导出计数一致。

`test_single_rubik_pipeline.py` 的16项编排回归测试也全部通过。该测试使用Python3.10，需要的测试专用 `plyfile==1.1.3` 仅装入 `/root/fast-pgsr-deployment/test-deps` 并通过 `PYTHONPATH` 引入，没有修改Fast-Sam2生产环境依赖。当前详细结果与原检查点哈希核验记录分别归档在 `environments/exports-3090-32596/reconstruction-results.json` 和 `spatial-final-verification.json`。

仅重新清理已有模型并更新渲染（先备份现存的目标导出、mesh和渲染目录）：

```bash
cd /root/Fast-PGSR
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen
RUN_EXTRACT=0 RUN_COLMAP=0 RUN_MASK=0 RUN_ALIGN=0 RUN_LABEL=0 RUN_TRAIN=0 \
RUN_EXPORT=1 RUN_RENDER=1 FORCE=1 TARGET_SPATIAL_RADIUS=0.8 RENDER_VOXEL_SIZE=0.02 \
bash run_batch_gpu1.sh
```

第14节已更新正式掩码和绑定。复用当前图像/新掩码/对齐结果重新训练，使用**新的输出目录**，不要覆盖历史结果：

```bash
RUN_EXTRACT=0 RUN_COLMAP=0 RUN_MASK=0 RUN_ALIGN=0 RUN_LABEL=0 \
OUTPUT_ROOT=/root/Fast-PGSR/output-retrain-new \
TARGET_SPATIAL_RADIUS=0.8 RENDER_VOXEL_SIZE=0.02 bash run_batch_gpu1.sh
```

保持 `RUN_TRAIN/EXPORT/RENDER` 为默认1。需要重新生成掩码时，应重新执行LABEL、更新绑定并训练；不要直接改PNG后复用旧语义初始化。空间清理无需新增生产依赖，使用训练环境已有的Open3D和SciPy。

## 14. 从训练掩码源头修复邻株误标（当前正式输入）

### 处理位置与输入

修复位于MASK阶段：DINO选择初始化目标、SAM2双向传播后，在保存 `object_mask/*.png` **之前**加入SfM实例约束与逐帧SAM2细化。LABEL和TRAIN直接读取修正后的PNG。仅使用本次RGB图像、当次SAM2粗掩码及上游COLMAP特征轨迹/相机/稀疏点；不读取已训练高斯、旧导出模型或其语义NPZ。适用于当前单植株、单魔方入口，双魔方旧分支保持原行为。

### 最小代码改动

| 文件 | 改动 |
|---|---|
| `scripts/mask_generation/instance_mask_guard.py` | 独立的源头实例约束模块 |
| `scripts/mask_generation/run_grounded_sam2_scene.py` | 在保存训练掩码前调用模块，保存逐帧检查与PNG哈希 |
| `scripts/mask_generation/run_rubik_scene_auto.py` | 与基础入口统一为v4，传递诊断开关 |
| `scripts/validate_semantic_inputs.py` | 拒绝旧掩码、被改写PNG及与当前实现不一致的检查记录 |
| `run_batch_gpu1.sh` | MASK复用和新生成后都强制检查源头实例约束记录 |
| `scripts/test_mask_instance_guard.py`、`scripts/test_single_rubik_pipeline.py` | 新增约束测试，更新模型替身接口回归测试 |

未修改SAM2/DINO权重、训练损失、COLMAP算法或环境依赖。使用Fast-Sam2环境已有的NumPy、SciPy、OpenCV和SAM2。

### 约束如何执行

1. 使用COLMAP可见特征轨迹对粗掩码投票，找出跨视角稳定的主株和单个魔方支持点。尺度由魔方稀疏点的稳健范围估计，因此不依赖尚未执行的ArUco对齐或固定世界尺度。
2. 对稳定植株支持点进行密度连通分析，保留明显主实例和附近断开的目标叶片。多个实例规模接近、初始化与主实例不一致、SfM支持不足时直接报错，不继续保存未经约束的掩码。
3. 邻株可能只在少数帧误入掩码，因此负提示不只取多数投票点，还取至少两次误入掩码、且远离目标三维支持的轨迹。逐帧负提示必须形成至少三个局部特征的支持组，避免单个SfM误匹配把主株细茎当成背景。
4. 每帧按目标SfM范围裁剪RGB，边界外扩12%（至少12像素），用6个分散的目标正提示、最多16个邻株负提示和粗掩码先验重新调用SAM2。最多三轮候选选择/纠正，检查正提示命中率≥80%、负提示误入率≤10%；不满足则停止，不回退到旧错误掩码。
5. 魔方保留唯一跟踪ID2，并用同一物理魔方的三维投影范围限制位置，清除远离魔方的ID2碎片；不会凭空补出被遮挡的魔方。最终PNG只允许 `0背景/1目标大豆/2魔方`。
6. `summary.json` 与 `instance_guard.json` 记录实现版本、实现文件SHA-256、逐帧提示检查和每张PNG的SHA-256。旧掩码、代码版本不匹配或手动改过PNG都会被拒绝复用。两个入口及新模块需要一起部署。

默认启用；`--disable-instance-guard` 仅用于旧流程诊断，批处理单魔方验收不会接受这种输出。

### 两个视频的真实验证

在独立验证目录重新执行了关键帧检测、SAM2正反向传播、源头实例约束及全部PNG写出，随后核验并同步到正式数据目录。第一段关键帧为 `frame_000006.jpg`，第二段为 `frame_000151.jpg`。

| 项目 | A1087-1-R3-0918 | A1087-2-R3-0918 |
|---|---:|---:|
| 新生成并核验掩码 | 142帧 | 162帧 |
| 有魔方标注的帧 | 129帧 | 145帧 |
| 最低正提示命中率 | 83.33% | 83.33% |
| 最高负提示误入率 | 0% | 6.25% |
| 原邻株源点仍被初始化为植株 | 146 → **0** | 255 → **0** |
| 新初始化植株点 | 8955 | 5607 |
| 新初始化魔方点 | 373 | 237 |

魔方被遮挡或跟踪未检出的帧允许ID2暂时为空；整个视频只跟踪一个物理魔方，未为了凑齐每帧对象而生成虚假掩码。新旧语义初始化的POINT3D_ID和顺序保持一致，魔方初始化点数保持373/237不变。

额外人工选定了第一段末帧左侧邻株、第二段中间帧右后方邻株的5处11×11像素检查区域：修复后植株误标占比均为0；5处目标叶片检查区域保持100%植株标签。该抽查不等于对304帧逐像素建立人工真值。所有304帧均自动检查标签、尺寸、哈希、实例约束和绑定，并查看了关键帧对比及每段16个均匀视角的联系图。

此前清理得到的邻株源点ID仅用于这次独立验收，**没有用作生成新掩码的输入**。这些源点从初始化植株标签146/255降为0，说明误标在训练前已被纠正，而非再次依靠导出端删点。

![第一段：左旧掩码，右源头修复后](docs/source-mask-fix-20261017/A1087-1-R3-0918-frame_000142.jpg)

![第二段：左旧掩码，右源头修复后](docs/source-mask-fix-20261017/A1087-2-R3-0918-frame_000082.jpg)

多视角预览另见 `docs/source-mask-fix-20261017/*-contact-sheet.jpg`。CPU源头约束测试、脚本编译检查及16项单/双魔方编排回归测试全部通过；真实视频验证未使用测试替身。

### 当前保存位置与备份

正式训练输入已更新为：

```text
/root/Fast-PGSR/data/A1087-1-R3-0918/object_mask/
/root/Fast-PGSR/data/A1087-2-R3-0918/object_mask/
```

相应的 `color_mask/`、`grounded_sam2_scene/`、`auto_rubik_keyframe.json` 和 `semantic_initialization/` 均已更新。LABEL已经重新执行，新 `input_binding.json` 在两个环境中验证通过。

旧掩码、旧语义初始化和旧绑定已备份到：

```text
/root/fast-pgsr-deployment/before-source-mask-fix-20261017-143808/
```

独立验证工作区：`/root/fast-pgsr-deployment/mask-source-validation/data/`。运行日志包括 `validate_source_masks_3090.sh.log`、`redo_first_source_masks.sh.log`、`promote_source_masks_and_label.sh.log`，均位于 `/root/fast-pgsr-deployment/`。第一段在最终实现冻结后又重跑一次基础分割入口，保证两个场景的实现SHA-256一致。

本地完整304张新掩码、语义初始化、绑定、逐帧报告和对比图保存在：

```text
C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\reconstruction-results\3090-32596-20261017-source-mask-fixed\
```

验收JSON已随项目归档至 `environments/exports-3090-32596/source-mask-validation-results.json`、`source-mask-label-results.json`、`source-mask-backup.json`。

### 后续重训

本次没有重训高斯模型，旧30000次检查点的SHA-256未改变。正式掩码和初始化已更新，因此旧检查点的输入绑定现在会失配，入口应拒绝将它当作新训练结果复用。使用新的输出目录，复用本次已经准备好的新输入：

```bash
cd /root/Fast-PGSR
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen
RUN_EXTRACT=0 RUN_COLMAP=0 RUN_MASK=0 RUN_ALIGN=0 RUN_LABEL=0 \
OUTPUT_ROOT=/root/Fast-PGSR/output-source-mask-retrain \
TARGET_SPATIAL_RADIUS=0.8 RENDER_VOXEL_SIZE=0.02 bash run_batch_gpu1.sh
```

该输出目录必须尚未包含旧训练结果。新视频从完整默认流程运行，源头约束会在MASK阶段自动执行；不要通过关闭校验或只改旧绑定来绕过重新训练。导出端空间筛选继续保留为附加保护。

复杂遮挡、严重SfM误匹配或目标与邻株无法分离时，提示检查和支持不足检查可能报错，需要选择更清晰关键帧或复核输入。当前验证证明了这两个视频的已知邻株误标被修复，不代表任意新场景都已具备逐像素人工标注精度。
