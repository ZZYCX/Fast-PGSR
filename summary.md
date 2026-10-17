# Fast-PGSR 单视频 → 目标植株 PLY 完整流程

> 本文档总结了从一份原始视频到导出目标植株高斯 PLY 的全流程命令。
> 涉及两个批处理脚本：
> - `run_batch.sh`：抽帧 / COLMAP / ArUco 对齐 / 训练 / 渲染 / 导出
> - `scripts/mask_generation/bash_tasks/batch_run_rubik.sh`：Mask 生成 / COLMAP 点打标签
>
> 两个脚本**分开运行**，mask/label 阶段必须插在训练之前。

---

## 一、整体流程图

```
videos/xxx.mp4
   │  ① 抽帧（run_batch.sh, RUN_EXTRACT）
   ▼
data/xxx/input/*.jpg
   │  ② COLMAP 预处理（run_batch.sh, RUN_COLMAP）
   ▼
data/xxx/sparse/*.bin  +  data/xxx/images/（去畸变图）
   │  ③ ArUco 坐标对齐（run_batch.sh, RUN_ALIGN）
   ▼
data/xxx/sparse/0/*.bin（train 固定读这个目录）
   │  ④ 生成 Mask（batch_run_rubik.sh, mask 阶段）
   ▼
data/xxx/object_mask/  +  color_mask/
   │  ⑤ COLMAP 点打标签（batch_run_rubik.sh, label 阶段）
   ▼
data/xxx/semantic_initialization/point_semantics.npz
   │  ⑥ 训练（run_batch.sh, RUN_TRAIN）
   ▼
output/xxx/point_cloud/iteration_30000/point_cloud.ply
   │  ⑦ 渲染 + 表面提取（run_batch.sh, RUN_RENDER）
   ▼
output/xxx/mesh/mesh_color.ply
   │  ⑧ 导出目标高斯（run_batch.sh, RUN_EXPORT）
   ▼
output/xxx/target_exports/plant_only/point_cloud.ply  ← 目标植株 PLY
   │  ⑨ （可选）LeafFit 植株结构拟合
   ▼
leaf_fit_segments/
```

---

## 二、逐步命令与环境

> 环境分工：
> - 抽帧：系统 `ffmpeg`（无需 conda）
> - COLMAP：系统 `colmap` 命令行（无需 conda）
> - 对齐：conda **FastGS**（pycolmap 4.1.0）
> - mask / label：conda **ObjectGS_GSAM2**（Python 3.10 + torch 2.7）
> - 训练 / 渲染 / 导出 / LeafFit：conda **fast-pgsr**（Python 3.7 + torch 1.12 + CUDA 扩展）

---

### ① 视频抽帧

- 开关：`run_batch.sh` `RUN_EXTRACT=1`
- 环境：系统 `ffmpeg`
- 输入：`videos/xxx.mp4`
- 输出：`data/xxx/input/frame_%06d.jpg`

```bash
ffmpeg -y -i videos/xxx.mp4 -vf "fps=2.5" -q:v 1 -start_number 1 \
  -hide_banner -loglevel error \
  data/xxx/input/frame_%06d.jpg
```

说明：40 秒视频按 2.5 FPS 抽取，约 100 张图片。

---

### ② COLMAP 预处理

- 开关：`run_batch.sh` `RUN_COLMAP=1`
- 环境：任意含 `colmap` 命令行的环境
- 输入：`data/xxx/input/*.jpg`
- 输出：`data/xxx/sparse/*.bin` + `data/xxx/images/`（去畸变图）

```bash
python scripts/preprocess/convert.py --data_path data --camera OPENCV
```

说明：`convert.py` 内部依次调用 COLMAP 命令行：
`colmap feature_extractor` → `colmap exhaustive_matcher` → `colmap mapper` → `colmap image_undistorter`。

---

### ③ ArUco 魔方坐标对齐

- 开关：`run_batch.sh` `RUN_ALIGN=1`
- 环境：conda **FastGS**（pycolmap 4.1.0）
- 输入：`data/xxx/input/` + `data/xxx/sparse/*.bin`
- 输出：`data/xxx/sparse/aligned/*.bin` → 自动复制为 `data/xxx/sparse/0/*.bin`

```bash
conda run --no-capture-output -n FastGS python align_aruco.py data/xxx
```

说明：
- `train.py` 固定读取 `sparse/0/` 目录，对齐结果必须落到这里。
- 需要 **pycolmap 4.1.0**，`fast-pgsr` 环境里的 pycolmap 0.4.0 不满足，故单独用 FastGS 环境。

---

### ④ 生成 Mask

- 阶段：`batch_run_rubik.sh` 的 `mask` 阶段
- 环境：conda **ObjectGS_GSAM2**
- 输入：`data/xxx/images/`
- 输出：`data/xxx/object_mask/` + `data/xxx/color_mask/`

```bash
python scripts/mask_generation/run_rubik_scene_auto.py --scene data/xxx \
  --scan-step 5 --refine-top-k 5 --box-threshold 0.30 --text-threshold 0.25 \
  --nms-threshold 0.72 --direction both --fill-hole-area 0 --overwrite
```

说明：
- 内部调用 `run_grounded_sam2_scene.py`：Grounding DINO 检测关键帧 → SAM2 双向传播。
- `object_mask/` 为单通道 PNG，标签编码：

| 值 | 类别 |
|----|------|
| 0  | 背景 background |
| 1  | 植株 plant |
| 2  | 魔方 rubik_1 |
| 3  | 魔方 rubik_2 |

- `color_mask/` 为可视化用。

---

### ⑤ COLMAP 稀疏点打初始标签

- 阶段：`batch_run_rubik.sh` 的 `label` 阶段
- 环境：conda **ObjectGS_GSAM2**
- 输入：`data/xxx/images/` + `data/xxx/object_mask/` + `data/xxx/sparse/0/`
- 输出：`data/xxx/semantic_initialization/point_semantics.npz`

```bash
python scripts/mask_generation/label_colmap_points_viewer.py --scene data/xxx \
  --patch-size 5 --target-dilation 5 --min-views 3 \
  --target-confidence 0.70 --background-confidence 0.90 \
  --no-visualize --overwrite
```

说明：
- 给每个 COLMAP 稀疏点建立初始类别概率，训练初始化高斯点时读该文件作为语义先验。
- 批处理默认 `--no-visualize`（避免弹三维窗口）；想交互检查可改 `--visualize`。

---

### ⑥ 训练

- 开关：`run_batch.sh` `RUN_TRAIN=1`
- 环境：conda **fast-pgsr**
- 输入：`data/xxx/sparse/0/` + `data/xxx/images/` + `data/xxx/object_mask/` + `data/xxx/semantic_initialization/`
- 输出：`output/xxx/point_cloud/iteration_30000/point_cloud.ply`

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/xxx -m output/xxx \
  --iterations 30000 --resolution 1 --data_device cuda \
  --ncc_scale 0.5 --densification_interval 500 \
  --opacity_cull_threshold 0.05 --exposure_compensation \
  --semantic_pipeline
```

说明：`--semantic_pipeline` 开启语义监管，训练过程中把 mask + 初始点标签同步给每个高斯。

---

### ⑦ 渲染 + 表面提取

- 开关：`run_batch.sh` `RUN_RENDER=1`
- 环境：conda **fast-pgsr**
- 输入：训练输出 `output/xxx/`
- 输出：`output/xxx/mesh/mesh_color.ply`

```bash
CUDA_VISIBLE_DEVICES=0 python render.py -m output/xxx \
  --max_depth 10.0 --voxel_size 0.01
```

---

### ⑧ 导出目标高斯

- 开关：`run_batch.sh` `RUN_EXPORT=1`（**当前唯一默认开启的开关**）
- 环境：conda **fast-pgsr**
- 输入：训练好的高斯 `output/xxx/point_cloud/iteration_30000/`
- 输出：
  - `output/xxx/target_exports/plant_only/point_cloud.ply` ← **目标植株 PLY**
  - `output/xxx/target_exports/rubik_1_only/`
  - `output/xxx/target_exports/rubik_2_only/`
  - `output/xxx/target_exports/selection_debug.ply`（彩色 debug 点云）

```bash
python scripts/export_target_gaussians.py \
  --iteration_dir output/xxx/point_cloud/iteration_30000 \
  --output_dir output/xxx/target_exports \
  --mode both --unknown_target_threshold 0.90 \
  --export_per_class --export_debug_cloud --overwrite
```

说明：按每个高斯的语义概率筛选出目标类别；`--mode both` 表示同时导出严格确定目标与高置信 unknown。

---

### ⑨ （可选）LeafFit 植株结构拟合

- 脚本：`scripts/leaf_fit/leafFit_batch.sh`
- 环境：conda **fast-pgsr**
- 输入：`output/xxx/plant_only/point_cloud.ply`（或 `target_exports/plant_only/`）
- 输出：`leaf_fit_segments/`

```bash
python scripts/leaf_fit/leafFit_segment_root_path.py \
  --input output/xxx/plant_only/point_cloud.ply \
  --leaf-fit-root third_party/leaf_fit \
  --disable-apex-cluster --tip-k 32 --path-k 64
```

---

## 三、运行要点

### 1. 两次运行批处理脚本

`run_batch.sh` 不会自动执行 ④⑤（mask/label）。完整处理一个新视频的次序：

```
第 1 次 run_batch.sh        # 开 RUN_EXTRACT / RUN_COLMAP / RUN_ALIGN
        ↓
batch_run_rubik.sh          # 开 mask + label
        ↓
第 2 次 run_batch.sh        # 开 RUN_TRAIN / RUN_RENDER / RUN_EXPORT
```

两个脚本均有**幂等跳过**逻辑（对应输出已存在即跳过），可反复运行；`FORCE=1` 可忽略已完成判断强制重跑。

### 2. 前置依赖链

| 阶段 | 依赖的上一阶段产物 |
|------|-------------------|
| ④ mask | ② 的 `images/` |
| ⑤ label | ③ 的 `sparse/0/` + ④ 的 `object_mask/` |
| ⑥ 训练 | `sparse/0/` + `images/` + `object_mask/` + `semantic_initialization/` |

### 3. 三个 conda 环境分工

| 环境 | Python / Torch | 负责阶段 |
|------|----------------|----------|
| **FastGS** | Python 3.10，pycolmap 4.1.0 | ③ ArUco 对齐 |
| **ObjectGS_GSAM2** | Python 3.10，torch 2.7.1+cu118 | ④ mask、⑤ label |
| **fast-pgsr** | Python 3.7，torch 1.12.1+cu116 | ⑥ 训练、⑦ 渲染、⑧ 导出、⑨ LeafFit |

> 提示：FastGS 可以合并进 ObjectGS_GSAM2（都是 Python 3.10 现代栈），只需 `pip install pycolmap==4.1.0`，并把 `run_batch.sh` 里的 `ALIGN_ENV` 改为 `ObjectGS_GSAM2`，即可从 3 个环境减到 2 个。
