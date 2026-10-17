# GPU 使用检查与修改

只修改本地项目，未连接或修改服务器。默认设备为 GPU 0，并尊重已设置的 CUDA_VISIBLE_DEVICES。未进行加速幅度测量。

## 调度修改

`run_batch_gpu1.sh` 将场景作为外层循环，每株依次执行抽帧、COLMAP、MASK、ALIGN、LABEL、TRAIN、RENDER、EXPORT，完成后才开始下一视频。阶段函数保留；单株失败跳过其余阶段并继续下一株，最后返回非零状态。保留关闭阶段、SCENE 筛选、已有数据场景和 FORCE 约定。

## 已启用或新增的 GPU 路径

| 部分 | 实现 |
|---|---|
| 视频抽帧 | 默认请求 FFmpeg CUDA 解码（可见设备 0），fps 及 JPEG 参数保持；不支持时自动重试 CPU，两次使用不同临时目录，不混入失败帧 |
| COLMAP | 默认 CUDA SIFT 提取、全量特征匹配；显式传可见设备 0，检查 `colmap -h` 是否报告 `with CUDA`，拒绝无 CUDA 构建悄悄回退 |
| GroundingDINO/SAM2 | 原代码已在 CUDA 上进行检测、分割和传播；继承批处理显卡可见性 |
| 训练/渲染/语义证据 | 原代码已有 CUDA 张量、CUDA 光栅化和训练优化；保留默认图像预加载到 CUDA |
| 图像灰度 | PIL 解码与缩放保持原处理，RGB 搬到指定设备后计算灰度，默认 CUDA |
| 多视图/NCC | 网格直接在 GPU 创建并按分辨率复用；有效像素无放回采样改用 GPU randperm，去掉 NumPy 采样及计数回传 |
| 相机几何 | 内参/逆内参按尺度缓存；R/T 张量驻留 GPU，更新插值相机时同步更新，不在每轮重建 CPU 张量 |
| 邻近视角 | 在 GPU 上按距离主键、夹角次键稳定排序，与原 np.lexsort 顺序对照；最终仅将小量相机索引传回用于 Python 相机对象/JSON |
| 法向/patch | offset 分支网格直接建立在点坐标设备，patch offsets 使用有界缓存 |

共享像素网格缓存最多4种尺寸/设备，避免给每一台相机保存全尺寸网格；每株使用独立进程，退出后释放训练显存。内参和 patch 缓存的返回张量只读使用。

## 保留 CPU 的部分及原因

- FFmpeg fps 筛选及 JPEG 编码、PIL/OpenCV 图像读写：保持像素处理和输出格式，不替换成另一种滤镜/编码流程。
- 当前 COLMAP mapper、去畸变：保留安装版本的实现；没有把传一个 GPU 参数当成这些步骤已全部使用 CUDA。
- ArUco OpenCV 检测、NumPy 三角化/几何对齐：保留已有单标记几何流程及拟合约束。
- COLMAP 稀疏点初始标签：按观测轨迹匹配 ID、读取掩膜并用 NumPy/OpenCV 累积；未重写成新的 CUDA 算法。
- 当前 Open3D 传统 TSDF、网格后处理：API 使用 CPU geometry 实现，替换为其他后端会改变重建流程，保持原逻辑。
- PLY/NPZ、哈希校验、日志/JSON/调试图：文件与调试输出需要 CPU，保留必要的数据回传。

## 配置

```bash
# 默认：GPU 0；CUDA 解码及 COLMAP GPU 开启
bash run_batch_gpu1.sh
# 指定显卡
GPU_ID=1 bash run_batch_gpu1.sh
# 显式 CPU 预处理（训练仍使用 GPU）
COLMAP_USE_GPU=0 EXTRACT_USE_GPU=0 bash run_batch_gpu1.sh
# 图像放系统内存、训练张量仍使用 CUDA；显存紧张时使用
TRAIN_DATA_DEVICE=cpu bash run_batch_gpu1.sh
```

CUDA 解码依赖 FFmpeg 的 CUDA 支持和视频编码支持；不满足时输出回退日志。COLMAP GPU 模式依赖 CUDA 构建，现有检查依赖其帮助信息 `with CUDA` 标识。保留原项目双 Conda 环境要求。

## 正确性验证范围

- 16 项原单魔方及调度测试通过，包含第二株抽帧必须晚于第一株导出、已有场景串行、失败阻断和输出保护。
- 9 项 GPU/解码测试通过；6 项真实 CUDA 张量测试在本机 RTX 3060 Laptop GPU、PyTorch 2.5.1+cu121 上执行，涵盖 RGB/RGBA 灰度对照、射线/内参、插值相机缓存、邻近排序、无重复采样；另外3项模拟 FFmpeg/COLMAP 用于检查回退隔离与 CUDA 构建检查。
- Python 语法检查与 Bash 语法检查通过。
- 未执行完整训练、真实视频硬件解码、实际 COLMAP 重建或服务器环境验证，也未测量速度。

GPU 浮点计算可能与 CPU 有很小舍入差异；采样仍为均匀无放回，但随机源由 NumPy 改为 PyTorch，会改变相同 seed 下的后续训练轨迹。未开启 TF32/混合精度或修改损失、迭代数、学习率来换取速度。
