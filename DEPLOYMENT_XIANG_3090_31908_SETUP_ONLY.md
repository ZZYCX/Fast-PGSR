# 项灿老师 RTX 3090 / 31908：仅部署与环境安装

实施时间：2026-10-17至2026-10-18（Asia/Shanghai）。

**本次仅完成项目及环境安装，未执行测试、推理或训练。**

本记录只描述安装及构建结果，不表示运行验证通过。未执行测试套件、模型加载、CUDA张量操作、视频处理、COLMAP重建、分割、对齐、标签生成、训练、导出或渲染。正确入口`run_batch_gpu1.sh`未启动。

## 实际位置及安装前状态

- SSH：`ssh -p 31908 root@172.26.14.81`。
- 主机名：`cuda11-8-0-cudnn8-devel-ub22-web-8bb1b53a756b4c65a9-76c94chss82`。
- 本地部署源码：`C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\deployment\Fast-PGSR`。
- 目标项目：`/root/Fast-PGSR`。
- Conda程序：`/data/miniconda3`；新环境：`/root/.conda/envs/fast-pgsr-old`、`/root/.conda/envs/Fast-Sam2`。
- 安装脚本、原始日志、缓存及退出码：`/root/fast-pgsr-new-setup-20261017`。
- 安装记录归档：`/root/fast-pgsr-new-setup-20261017/installation-records.tar.gz`；本地副本：`C:\Users\serendipity\Documents\ChatGPT\Fast-PGSR三维植株重建\server-audit\new-three-setup-20261017\xiang-3090-installation-records.tar.gz`。归档仅含安装脚本、下载/编译日志与元数据，不含模型权重、视频或训练结果。
- 项目内安装快照：`environments/exports-xiang-3090-setup-only/`。

三台目标均先完成TCP及SSH连接检查。系统Ubuntu22.04.3、x86_64、驱动580.105.08、初始GCC/G++11.4.0、24GiB系统内存、无swap；项目所在容器初始约300GiB可用。GPU由`nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv`检查。本机实际计算能力为8.6。

安装前没有目标项目或上述新环境，不覆盖旧结果。项老师两台服务器的`/data`是同一共享NFS，已有`/data/miniconda3`和其他环境；仅复用其Conda程序，新环境及包缓存均放在各容器的`/root`，不修改共享旧环境。王老师L20没有现有Conda，使用Miniconda3-py310_24.7.1-0-Linux-x86_64.sh安装到`/root/miniconda3`。王老师L20通过已核对哈希的缓存安装器提前完成bootstrap；系统包完成后停止了仍在下载的重复安装器。system-final-installation-metadata.json逐项记录apt包已安装和bootstrap退出码0，未中断dpkg。只读检查中看到的其他已有训练进程没有启动、停止或修改。

## 当前本地源码的完整迁移

依据`DEPLOYMENT_3090_FULL_GUIDE.md`第1～7节及第14节、`MIGRATION_GPU1.md`、`SINGLE_RUBIK.md`及工作区AGENTS.md。原3090文档保留为历史记录，其历史运行结果不属于本次安装证据。

Python/Paramiko通过SSH/SFTP上传本地源码包，Shell文件统一LF。原包3755文件；逐项核对大小和SHA-256后，将31个初次打包带入的历史重建预览/捕获帧/运行记录从项目移除；收尾再次核对哈希后清除这些本次上传的副本及旧传输包，保留3724文件的干净源码包`project-source-final.tar.gz`。没有删除服务器原有项目或结果。最终保留3724个原始源码、配置、依赖及文档文件，包含562个.cpp/.cu/.h/.cuh编译输入；新增部署记录、配置脚本及本机编译产物另计。排除.git、数据、视频、旧输出、运行日志、缓存、build/dist、旧.so/.o/.a/.obj/.pyd/.pyc和旧模型权重。没有以GitHub原项目或旧服务器代码替换本地修改。

以下文件成套部署并与传输清单核对SHA-256：

```text
scripts/mask_generation/instance_mask_guard.py
scripts/mask_generation/run_grounded_sam2_scene.py
scripts/mask_generation/run_rubik_scene_auto.py
scripts/validate_semantic_inputs.py
run_batch_gpu1.sh
```

另记录render.py及目标导出脚本哈希。源头掩码校验、输入绑定、单株空间清理、目标渲染与算法阈值保持原样。完整文件清单见`source-manifest-final.json`，首次逐项核对结果见`source-integrity.txt`。

## 系统依赖与下载处理

实际脚本：`system.sh`，日志：`system-before-mirror.log`、`system.log`；下载预取：`apt-prefetch.py`及其日志。Ubuntu原源下载停滞时，在确认dpkg尚未运行后停止本次apt下载，备份`/etc/apt/sources.list`到`apt-sources-before/`，将archive/security的Ubuntu地址改为清华镜像。NVIDIA软件源保留。对apt提供的准确包URL并发预取，核对大小与包索引提供的摘要，再交给apt正常安装；未使用无签名软件源。

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ffmpeg build-essential cmake ninja-build pkg-config wget curl ca-certificates git libgl1 libglib2.0-0 libeigen3-dev libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libgmp-dev libmpfr-dev
```

## 双Conda环境与Python包

两环境分离，名称大小写不变。先尝试使用两份环境YAML创建；共享Conda26.7.1因默认频道条款提示及巨大索引下载而中断该方式。随后逐项核对项目自带`exports-3090-32596/conda-*.explicit.txt`，确认包含YAML的所有固定包和完全相同版本，再用显式包清单安装，避免重新下载完整频道索引。只把默认频道URL改成镜像，未改版本；证明保存为`conda-yaml-lock-provenance.json`，临时YAML和显式清单均保留。额外包只包含原导出的传递依赖（例如训练环境liblzma/zstd、分割环境tzdata）。

```bash
B=/root/fast-pgsr-new-setup-20261017
export CONDARC="$B/condarc.yml"
export CONDA_ENVS_PATH=/root/.conda/envs CONDA_PKGS_DIRS="$B/conda-pkgs"
source /data/miniconda3/etc/profile.d/conda.sh
conda create -y --prefix /root/.conda/envs/fast-pgsr-old --file "$B/conda-fast-pgsr-old.explicit-local-cache.txt"
conda create -y --prefix /root/.conda/envs/Fast-Sam2 --file "$B/conda-Fast-Sam2.explicit-local-cache.txt"
```

Conda运行库cudatoolkit11.6.2的627909365字节归档如需分块下载，采用Anaconda API提供的SHA-256`5990db3504da9ffc8656c86b3684db9aec25a6462c966e37d0c046fc5745e574`核对，不以另一版本代替。

公网下载因多次超时后，按AGENTS.md及历史部署记录，只读访问172.26.14.81:32624的公共安装包缓存。复制33个官方wheel、63个Conda包归档及Miniconda安装器/COLMAP源码包，共98项；逐项核对源/目标SHA-256，所有wheel与框架清单、cudatoolkit与Anaconda API的官方SHA-256一致。没有复制旧环境目录、项目代码、训练产物或旧项目扩展.so。传输使用SFTP，临时askpass凭据仅在目标/root安装目录以600权限保存，传完删除。安装包进入本次wheelhouse/conda-pkgs，Conda显式清单进一步生成指向已核对归档的file URI版本，再在目标机解包和建立独立环境；构建源码仍为当前本地版本。完整缓存文件及校验见source-cache-transfer-manifest.json和cache-transfer-installed.json。初次SFTP批处理认证未成功，改用标准输入传入传输清单后完成。

六个torch/torchvision/torchaudio CUDA wheel由PyTorch官方索引提供URL及SHA-256，使用download.pytorch.org和download-r2.pytorch.org的相同文件；下载器在超时时保留已下载数据，HTTP Range续传后合并并核对完整哈希。普通包用清华PyPI镜像，27项大型NVIDIA runtime、Open3D、Triton按准确ABI、大小及SHA-256预取到`wheelhouse/`；保留清单要求的cu11/cu12运行库。`/root/.config/pip/pip.conf`仅配置本次wheelhouse的find-links。初次安装脚本按ABI选择框架wheel；分割环境预取后也可能同时安装已缓存的同ABI普通wheel，版本均仍为清单固定值。归档的environments-final.sh及下面复现命令将框架选择进一步限定为torch/torchvision/torchaudio三个文件，避免跨环境ABI3普通包误匹配。普通包按原清单继续安装，源码扩展最后安装。使用仅去掉额外PyTorch索引行的临时普通包清单安装，所有版本约束不变。

```bash
cd /root/Fast-PGSR
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:/usr/local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=3 FORCE_CUDA=1
export SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
conda activate /root/.conda/envs/fast-pgsr-old
python -m pip install --no-deps "$B"/wheelhouse/torch-*-cp37-*.whl "$B"/wheelhouse/torchvision-*-cp37-*.whl "$B"/wheelhouse/torchaudio-*-cp37-*.whl
python -m pip install -r "$B/requirements-fast-pgsr-old-mirror.txt"
python -m pip install -v --no-build-isolation --no-deps -r "$B/requirements-fast-pgsr-old-source-cached.txt"
conda activate /root/.conda/envs/Fast-Sam2
python -m pip install --no-deps "$B"/wheelhouse/torch-*-cp310-*.whl "$B"/wheelhouse/torchvision-*-cp310-*.whl "$B"/wheelhouse/torchaudio-*-cp310-*.whl
python -m pip install --no-index --find-links "$B/wheelhouse" --no-build-isolation --force-reinstall --no-deps -r "$B/requirements-Fast-Sam2-mirror.txt"
python -m pip install -v --no-build-isolation --no-deps -r requirements-Fast-Sam2-source.txt
```

训练环境默认requirements.txt引用requirements-fast-pgsr-old.txt；使用的镜像临时清单来自后者。PyTorch3D采用源码清单指定提交995b60e3b99faa1ee1bcdbe244426d54d98a7242（0.7.1）的本地缓存源归档，31092821字节，SHA-256`0b234e05e35f46a5ff18abb93c7f8321aa88115128267ce07a1170e4ae0a092d`。仅在临时安装清单将原URL替换为此file URI，原项目requirements不变。四个高斯/CUDA扩展、PyTorch3D、GroundingDINO及SAM2均在本机源码构建；没有传输旧服务器编译的扩展二进制。

分割环境普通包预先分4组仅下载到wheelhouse，三台prefetch-sam.exit均为0，不执行分割或模型加载。

分割环境安装时pip仍选择镜像索引候选，重复下载已有的大型包，原日志保存为environments-repeat-download.log。停止本次pip后，使用完整固定清单从已准备好的wheelhouse安装，指定`--no-index --find-links /root/fast-pgsr-new-setup-20261017/wheelhouse --no-build-isolation --force-reinstall --no-deps`；Conda依赖已按YAML对应清单安装，requirements本身为完整pip冻结清单，所有条目均安装，未删减。强制安装同一固定版本用于消除中断安装可能留下的不完整文件，不改变版本。普通包完整安装后再按源码清单编译GroundingDINO和SAM2。

实际版本由Python --version、pip list/freeze及conda list读取安装元数据获得，不导入模型或扩展作运行检查：

| 环境 | Python | torch | torchvision | torchaudio | NumPy | pycolmap |
|---|---|---|---|---|---|---|
| fast-pgsr-old | Python 3.7.13 | 1.12.1+cu116 | 0.13.1+cu116 | 0.12.1+cu116 | 1.21.6 | 0.4.0 |
| Fast-Sam2 | Python 3.10.20 | 2.7.1+cu118 | 0.22.1+cu118 | 2.7.1+cu118 | 2.2.6 | 4.1.0 |

完整版本和本地路径记录在对应的*.pip-list.json、*.freeze.txt和*.conda.json；不是可迁移安装命令。

## CUDA架构及COLMAP

系统编译工具包`/usr/local/cuda-11.8`，nvcc11.8.89；训练框架仍为cu116，分割框架仍为cu118。系统CUDA11.8与训练框架11.6的次版本差异按原指南保留，构建器可能输出警告；未通过GPU运行验证兼容性。

本机TORCH_CUDA_ARCH_LIST=8.6，COLMAP的CMAKE_CUDA_ARCHITECTURES=86。L20使用原生sm_89；PyTorch1.12.1的构建助手只接受到8.6，因此构建时仅临时把8.9加入其白名单，备份到`cpp_extension.py.original`，完成后恢复原文件，记录前后哈希。GroundingDINO/SAM2的setup.py自带70/75/80/86参数，L20通过`NVCC_PREPEND_FLAGS=-gencode=arch=compute_89,code=sm_89`额外生成原生89，不改setup.py、算法或阈值。3090使用86，不做白名单调整。

依据：[PyTorch1.12.1构建助手](https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/utils/cpp_extension.py)、[NVIDIA Ada兼容说明](https://docs.nvidia.com/cuda/ada-compatibility-guide/)、[nvcc环境参数](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-compiler-driver-nvcc/index.html#nvcc-environment-variables)。旧CUDA11.x框架的Ampere cubin/PTX兼容性与本机扩展的原生89编译是不同层面；本次不执行运行测试，不声称已经实测。

COLMAP3.9.1官方源码位于`/root/fast-pgsr-new-setup-20261017/src/colmap-3.9.1`，安装可执行文件`/usr/local/bin/colmap`：

```bash
cmake -S colmap-3.9.1 -B colmap-3.9.1/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build colmap-3.9.1/build -j 3
cmake --install colmap-3.9.1/build
```

安装证据为构建/安装退出码、CMakeCache配置、安装文件及哈希。没有执行COLMAP重建或COLMAP运行验证。扩展ELF架构由cuobjdump静态读取，未加载到GPU执行。

- `fast-pgsr-old`：`/root/.conda/envs/fast-pgsr-old/lib/python3.7/site-packages/fused_ssim_cuda.cpython-37m-x86_64-linux-gnu.so`，5173000字节；ELF架构：sm_86。
- `fast-pgsr-old`：`/root/.conda/envs/fast-pgsr-old/lib/python3.7/site-packages/simple_knn/_C.cpython-37m-x86_64-linux-gnu.so`，5917872字节；ELF架构：sm_86。
- `fast-pgsr-old`：`/root/.conda/envs/fast-pgsr-old/lib/python3.7/site-packages/pytorch3d/_C.cpython-37m-x86_64-linux-gnu.so`，75031824字节；ELF架构：sm_86。
- `fast-pgsr-old`：`/root/.conda/envs/fast-pgsr-old/lib/python3.7/site-packages/diff_plane_rasterization/_C.cpython-37m-x86_64-linux-gnu.so`，6339944字节；ELF架构：sm_86。
- `fast-pgsr-old`：`/root/.conda/envs/fast-pgsr-old/lib/python3.7/site-packages/diff_gaussian_rasterization_fastgs/_C.cpython-37m-x86_64-linux-gnu.so`，6480560字节；ELF架构：sm_86。
- `Fast-Sam2`：`/root/Fast-PGSR/third_party/Grounded-SAM-2/sam2/_C.so`，1936072字节；ELF架构：sm_70, sm_75, sm_80, sm_86。
- `Fast-Sam2`：`/root/Fast-PGSR/third_party/Grounded-SAM-2/grounding_dino/groundingdino/_C.cpython-310-x86_64-linux-gnu.so`，2429184字节；ELF架构：sm_70, sm_75, sm_80, sm_86。

## 必要模型资源

全部7项文件大小和SHA-256与指定清单一致。路径为`third_party/Grounded-SAM-2/`下：

```text
gdino_checkpoints/groundingdino_swint_ogc.pth
checkpoints/sam2.1_hiera_base_plus.pt
bert-base-uncased/config.json
bert-base-uncased/model.safetensors
bert-base-uncased/tokenizer.json
bert-base-uncased/tokenizer_config.json
bert-base-uncased/vocab.txt
```

BERT与GroundingDINO复用本地缓存并重新核对指定哈希后上传。32596源服务器尝试读取所需SAM2资源时SSH握手失败，未修改源服务器；SAM2改从官方`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`分块下载到本机缓存后上传。没有迁移训练产生的模型。完整大小、期望和实际哈希见installation-result.json与models-integrity.json，未通过模型加载或推理验证。

## 实际安装结果与限制

以下为最终阶段状态。环境和COLMAP的0为安装/构建命令退出码；缓存下载准备的0在复制并核对全部归档后写入，中断的公网下载原退出码保留在*-public-download.exit。王老师L20的system状态由已完成的apt安装及缓存Miniconda bootstrap合并记录，详见system-final-installation-metadata.json。均不表示运行验证通过；not finished表示尚未完成。

- `system`：`0`。
- `wheels`：`0`。
- `large-wheels`：`0`。
- `environments`：`0`。
- `colmap`：`0`。
- `models-integrity`：`0`。
- `cache-copy`：`0`。
- `apply-cache`：`0`。

安装后元数据对照：fast-pgsr-old的93项固定pip版本、Fast-Sam2的132项固定pip版本均与清单一致。全部3724个原始部署文件无缺失、无内容变更，562个C++/CUDA/头文件均保留。两个环境中的7个源码扩展已安装，ELF包含本机所需sm_86，COLMAP的ELF亦包含此架构。L20临时构建助手与原文件SHA-256已一致恢复；3090未调整该助手。

未安装成功的必要组件：无。运行兼容性及算法效果未进行实测；这是本次明确的范围限制。详细状态见`installation-result.json`。安装器为构建所需的框架导入属于安装过程；没有另行运行模型加载、CUDA张量或扩展导入检查。所有禁止的业务操作均未执行。后续任何运行行为需由用户手动启动；本次不能提供推理或训练效果、输入质量或完整重建的结论。

## 后续手动启动新视频（本次未执行）

先自行准备新视频及全新数据/输出目录，再登录本机执行。新增activate_setup_only.sh只设置环境与路径，不执行批处理。各机器系统内存均为24GiB，下面沿用指南的RENDER_VOXEL_SIZE=0.02手动配置；源算法默认值未改。

```bash
ssh -p 31908 root@172.26.14.81
cd /root/Fast-PGSR
source ./activate_setup_only.sh
VIDEO_DIR=/root/new-videos \
DATA_ROOT=/root/Fast-PGSR/data-new \
OUTPUT_ROOT=/root/Fast-PGSR/output-new \
LOG_ROOT=/root/Fast-PGSR/logs-new \
RENDER_VOXEL_SIZE=0.02 bash run_batch_gpu1.sh
```
