# Fast-PGSR 面向植株目标重建的语义增强改造总结

## 1. 项目目标

本项目以 Fast-PGSR 为基础，将二维目标分割结果引入三维高斯训练过程，使模型逐步具备以下能力：

1. 将二维图像中的植株、两个魔方和背景标签可靠地传递到 COLMAP 三维稀疏点；
2. 将三维稀疏点语义精确绑定到初始 Gaussian；
3. 在 Gaussian clone、split、prune、保存和恢复过程中始终保持语义状态与几何状态逐行一致；
4. 利用训练视角中的二维 Mask，为未知 Gaussian 累积多视图语义证据；
5. 周期性消解部分高置信 unknown；
6. 通过语义门控限制背景 Gaussian 继续 densify，将模型容量集中到植株和参考魔方区域；
7. 在不修改 Fast-PGSR CUDA 光栅化核心、不引入语义梯度分支的前提下，形成可解释、可调试、可复现的目标感知训练流程。

当前已经完成到 **TADC：目标感知密度控制**。背景渐进裁剪、目标区域加权损失等后续功能尚未实现。

---

## 2. 当前语义定义

### 2.1 硬标签

| 标签值 | 类别 |
|---:|---|
| `-1` | unknown |
| `0` | background |
| `1` | plant |
| `2` | rubik_1 |
| `3` | rubik_2 |

### 2.2 每个 Gaussian 携带的语义状态

#### 长期语义状态

```python
_object_id                  # [N]，当前硬标签
_object_prob                # [N, 4]，四类软概率
_semantic_confidence        # [N]，当前最高类别概率
_source_point3D_id          # [N]，来源 COLMAP POINT3D_ID
```

#### 动态语义状态

```python
_semantic_evidence_sum      # [N, 4]，当前证据窗口累计值
_semantic_obs_count         # [N]，当前窗口有效观测次数
_semantic_stable_count      # [N]，连续稳定窗口计数
```

这些语义张量均为普通 Tensor：

- 不是 `nn.Parameter`；
- `requires_grad=False`；
- 不进入 optimizer；
- 不通过梯度更新；
- 由多视图 Mask 证据和显式规则更新。

---

## 3. 完整数据流

```text
二维图像
   ↓
Grounded-SAM2 / 已生成 object_mask
   ↓
COLMAP 二维特征点与 POINT3D_ID 对应
   ↓
UPSI：三维稀疏点语义初始化
   ↓
point_semantics.npz
   ↓
POINT3D_ID 精确绑定
   ↓
初始 Gaussian 获得语义状态
   ↓
SLS：clone / split / prune 语义同步
   ↓
SSP：checkpoint / PLY sidecar 保存恢复
   ↓
VMA：训练 Camera 与 object_mask 对齐
   ↓
VEA：可见 unknown 多视图证据累计
   ↓
VGER：周期性更新概率并消解高置信 unknown
   ↓
TADC：语义门控 clone / split
   ↓
更紧凑的目标感知 Gaussian 模型
```

---

# 4. 各阶段代码修改与解决的问题

## 4.1 UPSI：不确定性保留的语义初始化

### 主要解决的问题

原始 COLMAP 三维稀疏点只有位置、颜色、重投影误差和观测关系，没有植株、魔方或背景标签。UPSI 负责将二维 `object_mask` 中的语义通过 COLMAP 的二维—三维观测关系融合到三维点上。

### 核心实现

主要脚本：

```text
label_colmap_points_viewer.py
```

输入：

```text
sparse/0/images.bin
sparse/0/points3D.bin
object_mask/*.png
```

输出：

```text
semantic_initialization/point_semantics.npz
semantic_initialization/points3D_semantic.ply
```

### `point_semantics.npz` 主要字段

```text
point3D_ids
xyz
original_rgb
object_probs
argmax_label
object_id
confidence
margin
view_count
fine_view_count
target_hit_count
reprojection_error
class_names
unknown_label
```

### 设计特点

- 不将所有点强制分类；
- 对证据不足、视图冲突、边界区域点保留 `unknown=-1`；
- `unknown` 不是第五类概率通道；
- 四类软概率顺序固定为：

```text
[background, plant, rubik_1, rubik_2]
```

### 解决效果

二维 Mask 已能在 COLMAP 稀疏点云上形成植株和两个魔方的大体三维轮廓，同时保留不确定点供后续训练消解。

---

## 4.2 初始语义绑定：POINT3D_ID → Gaussian

### 主要解决的问题

Fast-PGSR 会在读取 `points3D.bin` 时过滤高重投影误差或观测次数不足的点。因此，不能假设：

```text
NPZ 行号 = Fast-PGSR 过滤后点云行号
```

也不能依靠 XYZ 最近邻匹配，否则可能发生语义错位。

### 修改文件

```text
arguments/__init__.py
scene/colmap_loader.py
scene/dataset_readers.py
scene/__init__.py
scene/gaussian_model.py
train.py
```

### 核心修改

1. 在 COLMAP 点读取流程中继续保留过滤后的 `POINT3D_ID`；
2. 保证：

```text
point3D_id[i]
xyz[i]
rgb[i]
```

始终表示同一个三维点；
3. 新增参数：

```text
--semantic_init_path
```

4. 根据 `POINT3D_ID` 从 `point_semantics.npz` 中精确查找对应行；
5. 在不修改原 Fast-PGSR 初始化规则的情况下，为初始 Gaussian 附加语义状态。

### 保留的原 Fast-PGSR 初始化

```text
COLMAP XYZ       → Gaussian 中心
COLMAP RGB       → SH 零阶颜色
局部点间距       → scaling
单位四元数       → rotation
0.1              → opacity
随机初始化       → knn_f
scaling+rotation → 协方差
```

### 验收结果

- 所有过滤后 `POINT3D_ID` 均成功匹配；
- XYZ 对齐误差仅来自 float64→float32 舍入；
- 语义张量不在 optimizer 中；
- 未传语义参数时原路径完全不变；
- 10 轮端到端训练正常。

---

## 4.3 SLS：Semantic Lifecycle Synchronization

### 中文名称

**Gaussian 语义生命周期同步**

### 主要解决的问题

训练过程中 Gaussian 会发生：

```text
clone
split
prune
```

如果语义张量不跟随这些拓扑变化同步更新，就会出现：

```text
Gaussian 第 i 行
≠
语义张量第 i 行
```

这是后续所有语义控制失效的根本风险。

### 主要修改文件

```text
scene/gaussian_model.py
scripts/test_sls_sync.py
scripts/audit_sls_correctness.py
```

### 核心修改

增加集中辅助函数，对所有语义状态进行统一管理：

```text
_append_semantics_from_parents(...)
_prune_semantics(...)
_assert_semantic_alignment(...)
_log_semantic_op(...)
```

### Clone 规则

新子 Gaussian 继承父 Gaussian 的长期状态：

```text
object_id
object_prob
semantic_confidence
source_point3D_id
```

动态状态清零：

```text
semantic_evidence_sum = 0
semantic_obs_count = 0
semantic_stable_count = 0
```

### Split 规则

所有子 Gaussian 按几何生成顺序继承父语义。当前 Fast-PGSR 的 split 为块重复：

```text
父索引 [a,b,c]，N=2
子顺序 [a,b,c,a,b,c]
```

语义侧使用相同顺序。

### Prune 规则

语义张量使用与 `_xyz` 完全相同的保留 Mask：

```python
after_semantic = before_semantic[keep_mask]
```

### 覆盖路径

```text
densify_and_clone
densify_and_split
densify_and_clone_fastgs
densify_and_split_fastgs
densify_and_prune_fastgs
final_prune_fastgs
prune_points
densification_postfix
```

### 验收结果

- 1051 项独立审计断言通过；
- 52 项 SLS 验收断言通过；
- 几何指纹 `_knn_f` 与语义指纹 `_source_point3D_id` 顺序一致；
- clone、split、prune 连续与随机操作均通过；
- 无父子 Tensor 存储别名；
- 无语义路径保持原行为；
- A/B 训练高斯数量轨迹一致。

---

## 4.4 SSP：Semantic State Persistence

### 中文名称

**语义状态持久化**

### 主要解决的问题

如果训练中断后恢复，动态语义状态必须连续保留，否则会丢失：

```text
object_prob
evidence_sum
obs_count
stable_count
```

从而导致续训结果与连续训练不一致。

### 主要修改文件

```text
scene/gaussian_model.py
scene/__init__.py
scripts/test_semantic_persistence.py
```

### Checkpoint 路径

原有：

```text
capture()
restore()
```

扩展保存：

```text
_semantics_attached
semantic_state_version
7 个语义 Tensor
```

新格式兼容旧 17 字段 checkpoint。

### PLY 路径

保持：

```text
point_cloud.ply
```

格式不变，同时在同目录写入：

```text
semantic_state.npz
```

### Sidecar 保存内容

```text
object_id
object_prob
semantic_confidence
source_point3D_id
semantic_evidence_sum
semantic_obs_count
semantic_stable_count
semantics_attached
semantic_state_version
gaussian_count
class_names
xyz_order_checksum
semantic_order_checksum
```

### 双重顺序校验

- `xyz_order_checksum`：检查 PLY 中 Gaussian 顺序；
- `semantic_order_checksum`：检查语义数组内部顺序和内容。

### 验收结果

- 58 项测试全部通过；
- checkpoint round-trip 七个 Tensor 精确相等；
- clone/split/prune 后保存恢复精确相等；
- PLY + sidecar 恢复正确；
- 点顺序打乱、语义行打乱、数量错误、非法概率均能报错；
- 连续训练与中断续训的语义状态一致；
- 老 checkpoint 和老 PLY 路径保持兼容。

---

## 4.5 VMA：View–Mask Alignment

### 中文名称

**训练视图—语义 Mask 对齐**

### 主要解决的问题

后续 Gaussian 投影到像素 `(u,v)` 后，需要读取当前训练 Camera 对应的二维语义标签。如果图像和 Mask 文件错配、分辨率不一致或插值方式错误，后续证据全部不可信。

### 主要修改文件

```text
arguments/__init__.py
scene/dataset_readers.py
scene/cameras.py
utils/camera_utils.py
scene/__init__.py
scripts/test_vma_alignment.py
```

### 核心修改

新增参数：

```text
--semantic_mask_dir
--export_mask_alignment_debug
```

匹配规则：

```text
images/<stem>.<ext>
↔
object_mask/<stem>.png
```

不按文件排序位置匹配。

### Camera 新增属性

```python
camera.object_mask
```

特性：

- 单通道；
- `torch.uint8`；
- 标签仅为 `0/1/2/3`；
- 尺寸与当前训练图像完全一致；
- 图像缩放使用原 Fast-PGSR 方式；
- Mask 缩放使用最近邻插值。

### 调试输出

```text
原训练图像
彩色 Mask
半透明叠加图
mask_alignment_report.json
```

### 验收结果

- 101 个 Camera 全部匹配；
- 缺失数为 0；
- 多分辨率下最近邻缩放与参考逐像素一致；
- Mask 标签未产生非法中间值；
- 加载 Mask 不改变训练图像 Tensor；
- 无 Mask 模式原行为不变；
- 32 项测试全部通过。

---

## 4.6 VEA：Visibility-filtered Evidence Accumulation

### 中文名称

**可见性过滤的语义证据累计**

### 主要解决的问题

初始 `unknown` Gaussian 需要在训练过程中利用多个视角重新获得语义证据。但不能直接读取投影像素标签，还必须排除：

- 视锥外点；
- 相机后方点；
- 图像外点；
- 无效深度点；
- 被前景遮挡的后方 Gaussian。

### 主要修改文件

```text
arguments/__init__.py
gaussian_renderer/__init__.py
utils/semantic_evidence_utils.py
scene/gaussian_model.py
train.py
scripts/test_semantic_evidence_accumulation.py
```

### Renderer 深度审计结论

Fast-PGSR 使用：

```text
plane_depth
```

它是沿像素射线到 alpha 合成切平面的射线参数，属于 surface depth，不是普通期望深度。

Gaussian 中心投影后，相机空间深度 `Z` 与该射线参数处于同一深度定义，可进行一致性比较：

```text
|Z - plane_depth|
<= max(abs_tol, rel_tol × |plane_depth|)
```

### 证据累计流程

```text
unknown Gaussian
    ↓
visibility_filter
    ↓
投影到当前训练 Camera
    ↓
前方与图像范围检查
    ↓
有效 plane_depth 与 alpha 检查
    ↓
深度一致性遮挡过滤
    ↓
读取 5×5 Mask 局部类别比例
    ↓
evidence_sum += patch_prob
obs_count += 1
```

### 预热机制

默认：

```text
semantic_warmup_iterations = 1000
```

预热期间只进行原 Fast-PGSR 几何与外观优化，不累计语义证据。

### 本阶段不修改

```text
object_id
object_prob
semantic_confidence
semantic_stable_count
```

只修改：

```text
semantic_evidence_sum
semantic_obs_count
```

### 验收结果

- 58 项测试全部通过；
- 5×5 patch 中心、边缘和角落归一化正确；
- 投影与独立数学参考一致；
- 遮挡点能够被深度一致性拒绝；
- 只更新 unknown；
- SLS 和 SSP 集成正确；
- 2 万 Gaussian 单轮约 2.3 ms；
- 无计算图和持续显存增长；
- 真实训练中观测次数随视角增加；
- VEA 不改变几何训练轨迹。

---

## 4.7 VGER：Visibility-Guided Evidence Resolution

### 中文名称

**可见性引导的 unknown 语义消解**

### 主要解决的问题

VEA 只收集证据，不能改变标签。VGER 负责每隔固定窗口，将累计证据与历史概率融合，并只消解证据充分的 unknown。

### 主要修改文件

```text
arguments/__init__.py
scene/gaussian_model.py
train.py
scripts/test_semantic_resolution.py
```

### 刷新时机

默认：

```text
warmup = 1000
refresh_interval = 500
首次刷新 = 1500
```

执行位置：

```text
当前 iteration 的 VEA 累计之后
clone/split/prune 之前
```

### 概率融合

窗口概率：

```text
window_prob = evidence_sum / obs_count
```

历史融合：

```text
new_prob =
(prior_strength × old_prob + evidence_sum)
/
(prior_strength + obs_count)
```

默认：

```text
prior_strength = 5
```

### Unknown 转目标

满足：

```text
obs_count >= 5
top1_class ∈ {1,2,3}
top1_prob >= 0.75
margin >= 0.20
```

则转换为 plant、rubik_1 或 rubik_2。

### Unknown 转背景

满足：

```text
obs_count >= 8
p_background >= 0.95
p_target <= 0.05
连续两个窗口满足
```

才转换为 background。

背景判定比目标更保守。

### 窗口刷新后

清零：

```text
semantic_evidence_sum
semantic_obs_count
```

保留：

```text
object_prob
semantic_confidence
semantic_stable_count
object_id
```

### 验收结果

- 40 项测试全部通过；
- 概率融合与独立 NumPy 参考一致；
- 目标阈值和 margin 边界正确；
- 背景连续稳定窗口逻辑正确；
- 证据不足的 unknown 保持不变；
- 已确定类别不被修改；
- checkpoint 续训后继续正常刷新；
- 真实短训练中约 8% 初始 unknown 被高置信消解；
- VGER 不改变可学习参数和原几何轨迹。

---

## 4.8 TADC：Target-Aware Density Control

### 中文名称

**目标感知的 Gaussian 密度控制**

### 主要解决的问题

原 Fast-PGSR densify 不区分类别。背景占初始 Gaussian 的大多数，如果继续参与 clone/split，会大量消耗模型容量、显存和计算时间。

TADC 第一次让语义真正影响训练资源分配。

### 主要修改文件

```text
arguments/__init__.py
scene/gaussian_model.py
train.py
scripts/test_target_aware_density_control.py
```

### 统一生长资格

定义：

```text
known_target =
    object_id ∈ {1,2,3}

p_target =
    p_plant + p_rubik_1 + p_rubik_2

unknown_grow =
    object_id == -1
    且 p_target >= threshold

background =
    object_id == 0
```

最终：

```text
semantic_grow_mask =
known_target OR unknown_grow
```

### 接入方式

保留 Fast-PGSR 原始候选条件：

```text
梯度
绝对梯度
scale
importance
radii
FastGS metric
```

只在最终候选 Mask 后增加：

```text
final_mask =
original_mask & semantic_grow_mask
```

覆盖：

```text
densify_and_clone
densify_and_split
densify_and_clone_fastgs
densify_and_split_fastgs
```

### 当前背景行为

Background：

- 禁止 clone；
- 禁止 split；
- 仍参与渲染；
- 仍参与原有损失；
- 仍更新 XYZ、SH、scale、rotation、opacity；
- 仍可被 Fast-PGSR 原有 prune 条件删除；
- 未执行语义强制删除。

### 验收结果

48 项 TADC 测试全部通过。

短训练对比：

| 运行 | 最终 Gaussian 数 | PSNR | L1 |
|---|---:|---:|---:|
| 无 TADC | 166729 | 14.413 | 0.150 |
| TADC，unknown 阈值 0.40 | 66062 | 16.367 | 0.118 |
| TADC，unknown 阈值 1.00 | 65972 | 16.377 | 0.118 |

主要现象：

- 背景 Gaussian 从初始 `60977` 保持不增长；
- 无 TADC 时背景增长到 `117383`；
- plant 和两个 rubik 类别仍能正常 densify；
- 阈值 0.40 时，仅部分高目标概率 unknown 允许生长；
- Gaussian 总数短训练中减少约 60%；
- 未出现 NaN 或 Inf；
- SLS、SSP、VEA、VGER 回归测试继续通过。

注意：以上 PSNR 和 L1 来自短训练，只能说明当前机制运行有效，不能直接替代完整训练和正式论文实验结论。

---

# 5. 当前新增或扩展的主要命令行参数

## 初始语义与 Mask

```text
--semantic_init_path
--semantic_mask_dir
--export_semantic_binding_debug
--export_mask_alignment_debug
```

## SLS / SSP 调试

```text
--sls_debug
```

## VEA

```text
--enable_semantic_evidence
--semantic_warmup_iterations
--semantic_patch_size
--semantic_depth_abs_tolerance
--semantic_depth_rel_tolerance
--semantic_min_render_alpha
--semantic_evidence_debug
--semantic_evidence_debug_interval
```

## VGER

```text
--semantic_refresh_interval
--semantic_prior_strength
--semantic_target_min_observations
--semantic_target_confidence
--semantic_target_margin
--semantic_background_min_observations
--semantic_background_confidence
--semantic_background_max_target_prob
--semantic_background_stable_windows
--semantic_resolution_debug
```

## TADC

```text
--enable_semantic_density_control
--semantic_unknown_grow_threshold
--semantic_density_debug
--semantic_density_debug_interval
```

---

# 6. 当前测试体系

当前已经形成独立的自动化测试和回归测试体系：

```text
scripts/test_sls_sync.py
scripts/audit_sls_correctness.py
scripts/test_semantic_persistence.py
scripts/test_vma_alignment.py
scripts/test_semantic_evidence_accumulation.py
scripts/test_semantic_resolution.py
scripts/test_target_aware_density_control.py
```

累计测试结果：

```text
1339 项检查全部通过
```

测试覆盖：

- POINT3D_ID 对齐；
- 语义 Tensor 长度、顺序和内容；
- clone/split/prune 生命周期；
- checkpoint 和 PLY sidecar 持久化；
- 图像—Mask 对齐；
- Gaussian 投影；
- 深度与遮挡判断；
- 5×5 Mask 局部证据；
- unknown 概率更新；
- 背景稳定窗口；
- densify 语义门控；
- optimizer 隔离；
- 无语义兼容路径；
- 真实训练 A/B 回归。

---

# 7. 当前训练流程

```text
训练开始
   ↓
读取 COLMAP 点云与 Camera
   ↓
根据 POINT3D_ID 加载初始语义
   ↓
按原 Fast-PGSR 规则初始化 Gaussian
   ↓
加载每个 Camera 对应 object_mask
   ↓
0～warmup：原 Fast-PGSR 几何和外观预热
   ↓
warmup 后：
   VEA 对可见 unknown 累积 Mask 证据
   ↓
每 refresh_interval：
   VGER 更新概率并消解高置信 unknown
   ↓
每次 densify：
   TADC 计算语义生长资格
   ↓
原 Fast-PGSR 候选 AND 语义资格
   ↓
clone / split
   ↓
SLS 同步语义状态
   ↓
checkpoint / PLY 保存时由 SSP 持久化语义
```

---

# 8. 当前已经解决的核心问题

## 已解决

1. 二维 Mask 到 COLMAP 三维点的语义传递；
2. 不确定点不被强制错误分类；
3. COLMAP 过滤后语义与 Gaussian 的精确 ID 对齐；
4. clone/split/prune 后语义不再错位；
5. 中断训练和模型加载后语义状态不丢失；
6. Camera 图像与 Mask 在训练分辨率下严格对齐；
7. 被遮挡 unknown 不会错误读取前景 Mask；
8. unknown 可通过多视图证据逐步消解；
9. 背景不再大量占用 densify 预算；
10. 无语义模式仍保持原 Fast-PGSR 行为；
11. 所有语义机制均不进入梯度优化器。

---

# 9. 当前尚未实现的内容

## 9.1 稳定背景渐进裁剪

尚未实现：

```text
object_id == 0 → 强制删除
```

后续应采用联合条件：

```text
稳定高置信背景
AND
Fast-PGSR 原有低 opacity / 低 importance / prune 条件
```

逐步删除背景，而不是一次性全部删除。

## 9.2 目标区域加权损失

当前背景仍完整参与原 Fast-PGSR 的图像与几何损失。

后续可考虑：

```text
目标区域高权重
目标边缘低权重
远处背景低权重或不参与
```

但需要独立验证是否会造成渲染空洞或几何不稳定。

## 9.3 正式长训练与完整评价

目前大部分结果来自短训练和功能验收，仍需进行：

- 完整迭代训练；
- 多场景复现；
- PSNR / SSIM / LPIPS；
- 目标区域指标；
- 几何完整性评价；
- 枝干断裂与叶缘保持情况；
- Gaussian 数量；
- 峰值显存；
- 每轮耗时；
- 总训练时间；
- 模型文件大小。

## 9.4 阈值消融

需要系统比较：

```text
semantic_min_render_alpha
depth_abs_tolerance
depth_rel_tolerance
target_confidence
target_margin
background_confidence
unknown_grow_threshold
```

当前默认值已通过功能测试，但尚不能视为最终最优值。

---

# 10. 下一阶段建议

下一阶段建议实现：

## Stable Background Progressive Pruning

中文：

**稳定背景渐进式裁剪**

目标：

```text
只删除长期稳定为背景
并且同时满足 Fast-PGSR 原有低重要性条件的 Gaussian
```

建议原则：

1. 不在训练早期删除背景；
2. 从较晚 iteration 开始；
3. 每隔固定周期检查一次；
4. 只处理 `object_id == 0`；
5. 叠加高背景概率与稳定条件；
6. 叠加原有低 opacity、低 importance 或 prune 条件；
7. unknown 永不因语义直接删除；
8. 目标类别永不进入背景语义裁剪；
9. 删除后继续复用 SLS 和 SSP；
10. 先做独立测试和短训练，再进入正式长训练。

---

# 11. 阶段性结论

截至当前，Fast-PGSR 已从原始的无类别重建框架扩展为一个具备以下能力的目标感知训练系统：

```text
二维语义初始化
+
三维不确定性保留
+
Gaussian 语义生命周期
+
语义状态持久化
+
视图—Mask 对齐
+
可见性与遮挡过滤
+
多视图证据累计
+
unknown 周期消解
+
目标感知密度控制
```

当前系统已经能够在不改动 CUDA 光栅化核心、不增加语义梯度分支的情况下，将训练资源从大面积背景逐步转移到植株和参考魔方区域，同时保留 unknown 的保守处理机制，并通过完整测试体系保证几何与语义状态始终一致。




# 执行命令：
python train.py \
    -s data_test/soybean10_0730 \
    -m output/soybean10_0730 \
    --iterations 30000 \
    --resolution 1 \
    --data_device cuda \
    --ncc_scale 0.5 \
    --densification_interval 500 \
    --opacity_cull_threshold 0.05 \
    --exposure_compensation \
    --semantic_pipeline \
    2>&1 | tee logs/mungbean13_semantic_train.log



# 分割命令
得到mask命令：
python scripts/mask_generation/run_rubik_scene_auto.py     --scene mungbean12_0730     --fill-hole-area 0        --overwrite
在进行colmap和align后，可能需要进行批处理的得到mask的命令如下：
./scripts/mask_generation/bash_tasks/batch_run_rubik.sh 

