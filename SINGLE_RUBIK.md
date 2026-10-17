# 单魔方大豆重建

本地适配入口为 `run_batch_gpu1.sh`，文件名保留，默认 GPU 0、一个魔方、30000 次训练、`strict` 导出。按株串行执行：当前视频抽帧 → COLMAP → MASK → ALIGN → LABEL → TRAIN → EXPORT → RENDER → 下一株。2026-10-17 的单株空间清理修复已同步本地和3090；4090源服务器保持原样。`run_batch.sh` 是旧入口，不包含完整的 MASK/LABEL 自动流程；使用下述入口运行。

## 启动

将本地项目部署到已配置双环境的 Linux 机器后，在项目根目录执行：

```bash
VIDEO_DIR="/实际大豆视频目录" \
DATA_ROOT="/新的大豆中间数据目录" \
OUTPUT_ROOT="/新的大豆输出目录" \
bash run_batch_gpu1.sh
```

已有数据但无对应视频的场景也会进入队列；`RUN_EXTRACT=0` 可只处理已有场景。重复视频场景名（例如同名 MP4/MOV）会明确报错，避免互相覆盖。

省略路径时使用项目的 `videos/`、`data/`、`output/`、`logs/`。`PROJECT_ROOT` 可覆盖项目位置。不要将数据根和输出根设成同一目录、互相包含的目录或包含源码的目录。

```bash
# 只处理一个场景；用视频文件名去掉扩展名作为 SCENE。
SCENE=A907-1-V12-0918 VIDEO_DIR="/视频目录" bash run_batch_gpu1.sh
# 已有中间数据，关闭抽帧和 COLMAP（保留其他阶段默认开关）。
RUN_EXTRACT=0 RUN_COLMAP=0 DATA_ROOT="/数据目录" bash run_batch_gpu1.sh
# 显式选择其他 GPU，或处理历史双魔方场景。
GPU_ID=1 RUBIK_COUNT=2 bash run_batch_gpu1.sh
```

如果已有 `CUDA_VISIBLE_DEVICES`，入口尊重该设置。COLMAP 和视频 CUDA 解码在可见设备中使用索引 0，分割/训练/渲染/导出继承相同可见设备。默认 `COLMAP_USE_GPU=1`、`EXTRACT_USE_GPU=1`，训练图像预加载到 CUDA；COLMAP CUDA 构建无法确认时直接报错，视频硬件解码失败则隔离清理失败帧后回退 CPU。需要时可显式设置 `COLMAP_USE_GPU=0 EXTRACT_USE_GPU=0`，或设置 `TRAIN_DATA_DEVICE=cpu` 减少图像驻留显存。详见 [GPU_AUDIT.md](GPU_AUDIT.md)。环境默认为 COLMAP/训练 `fast-pgsr-old`，分割/对齐/标签 `Fast-Sam2`，可由 `COLMAP_ENV`、`TRAIN_ENV`、`MASK_ENV`、`ARUCO_ENV` 覆盖。`RUN_EXTRACT/COLMAP/MASK/ALIGN/LABEL/TRAIN/RENDER/EXPORT` 均支持环境变量 0/1。

## 单魔方行为

2026-10-17 已进一步从MASK源头约束实例：默认使用COLMAP跨视角支持构造主株正提示/邻株负提示，裁剪后逐帧细化SAM2，再写出训练用PNG；魔方ID2限制在同一物理魔方的投影区域。检查不通过直接停止。`summary.json` 记录实现版本/哈希及逐张PNG哈希，批处理拒绝旧掩码或被修改的掩码。生产默认不使用 `--disable-instance-guard` 诊断开关。两个入口、新模块及校验器须成套部署。算法和两个视频的304帧验证见 [完整文档第14节](DEPLOYMENT_3090_FULL_GUIDE.md#14-从训练掩码源头修复邻株误标当前正式输入)。

- 两个分割 Python 入口 `--rubik-count` 默认 1，双魔方明确传 2。
- `object_mask` 仅使用 `0=背景、1=目标植株、2=魔方`。遮挡帧可暂时缺少某目标，但整个场景必须包含植株和魔方。
- 自动关键帧评分不再索引第二魔方的框/Mask；评分后的目标框直接通过 `--selection-json` 交给传播，避免重新检测换目标。
- `--plant-index` / 批处理 `PLANT_INDEX` 选择 NMS 后植株检测索引，默认 0。多株场景中索引不保证跨帧身份，需要查看 `auto_rubik_keyframe.json`、初始化预览和传播预览。必要时直接用指定关键帧脚本选择确认过的目标后关闭批处理 MASK 阶段复用。
- 四类 `[N,4]` 概率结构和 NPZ 格式保留；类别3为未使用槽位。单模式不再报告 `rubik_2_missing`，CSV 保留 `rubik_2_area=0` 便于旧工具读取。
- ArUco 对齐继续使用已有单标记分支；默认字典 `DICT_6X6_250`、ID 0/1、至少两个注册视角及几何质量检查。一个物理魔方可包含多个标记，不能混为同一概念。`MARKER_LENGTH=1.0` 是归一化标记边长，不代表默认单位是米。

## 输出与复用

最终合并高斯文件是：

```text
<OUTPUT_ROOT>/<scene>/target_exports/strict/point_cloud.ply
```

它包含目标植株与一个魔方，保留全部原高斯属性（坐标、SH、不透明度、尺度、旋转）。原 `point_cloud/iteration_30000/point_cloud.ply` 是训练完整模型，不等于筛选结果。`plant_only`、`rubik_1_only` 和 `selection_debug.ply` 用于单独检查。无需网格时可设置 `RUN_RENDER=0`，仍可导出高斯。

`strict` 排除 background 和 unknown；单模式发现 rubik_2 标签或目标类别为空会报错。严格筛选可能遗漏证据不足的细茎/叶缘，必须视觉检查目标完整性和邻株残留，标签计数不能单独证明正确分割。

批处理现默认 `TARGET_SPATIAL_RADIUS=0.8`：对每个目标类别保留有足够密度支持的主要三维实例；植株额外保留距主簇不超过1.6的稠密断开叶片，拒绝邻株簇及稀疏噪声。半径采用ArUco标记边长=1的归一化坐标，不能直接用于未对齐模型。若两个最大实例规模接近（不足1.5倍），停止导出并要求复核，避免猜测目标。原模型与掩码不修改，62项高斯属性保留；阈值及各簇计数记录于 `export_manifest.json` 的 `spatial_cleanup`。

先导出，再通过 `render.py --target_ply .../target_exports/strict/point_cloud.ply` 生成相同目标的渲染和网格，避免清理后的高斯与旧完整场景网格混用。24GiB系统内存的3090使用 `RENDER_VOXEL_SIZE=0.02`。只有确认要复现旧语义导出时才设置 `TARGET_SPATIAL_RADIUS=0`；保守导出模式也需显式关闭空间清理。若 `RUN_EXPORT=0 RUN_RENDER=1`，必须已有经过核验的strict导出模型。

默认不删除已有产物：匹配的中间产物可以复用，不完整或配置冲突的产物会报错；已有 render/export 输出也不会自动清除。明确需要覆盖时设 `FORCE=1`，它仅重做已开启阶段，会替换这些阶段的已有产物。优先使用新的独立数据/输出目录验证。

旧双魔方 Mask 或未带绑定记录的旧语义初始化不能直接当作新单魔方输入。重新生成 MASK、LABEL 后会生成 `semantic_initialization/input_binding.json`，绑定当前图像、掩膜和 `sparse/0` 的 SHA256；训练前和复用结果时验证一致性。训练结果同时保存 `semantic_input_binding.json`。对齐复用使用 `sparse/aligned/input.sha256`。某场景上游阶段失败，本轮后续阶段跳过该场景，最终返回非零退出码，日志按运行保存。

独立 COLMAP 工具新增 `--scene_path`、`--gpu_index`；默认不覆盖已有中间文件，重做明确传 `--overwrite`。命令采用 argv 列表，支持包含空格的路径。

## 验证

```bash
bash -n run_batch_gpu1.sh
# 在具备 numpy、Pillow、opencv-python、plyfile 的 Python 3.10+ 环境：
python scripts/test_single_rubik_pipeline.py
python scripts/test_gpu_optimizations.py
# 在训练环境验证单株清理与完整属性导出：
python scripts/test_spatial_target_selection.py
python scripts/test_target_gaussian_exporter.py
# 在Fast-Sam2环境检查源头实例约束与旧掩码拒绝：
python scripts/test_mask_instance_guard.py
```

合成测试涵盖单/双魔方评分与传播编排、全分辨率细化路径、参数传递、旧输入绑定拒绝、单魔方空类别、严格导出的全部62个PLY字段和值/行序/校验和，以及批处理 GPU 0、上游失败阻断和旧结果保护。
模型与 Conda 调用使用测试替身，不能替代真实 GroundingDINO/SAM2 识别、ArUco 拟合及30000次训练质量验证。本地未启动真实视频训练。

正式3090的两个场景已经替换新掩码并重建语义初始化。旧训练模型保持原样，输入绑定将与新掩码失配；后续应使用新的 `OUTPUT_ROOT` 训练。第14节提供可直接执行的命令和原掩码备份路径。
