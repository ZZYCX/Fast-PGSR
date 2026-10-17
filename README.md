> 本项目大豆三维重建的正确批处理入口为 **`run_batch_gpu1.sh`**（保留历史文件名，默认 **GPU 0、一个魔方、逐株完成全部阶段、严格目标导出**）。单魔方操作与测试说明见 [SINGLE_RUBIK.md](SINGLE_RUBIK.md)，GPU 加速范围见 [GPU_AUDIT.md](GPU_AUDIT.md)。迁移训练需要分别重建 `fast-pgsr-old` 和 `Fast-Sam2` 两个 Conda 环境；安装步骤见 [MIGRATION_GPU1.md](MIGRATION_GPU1.md)。根目录 `requirements.txt` 默认对应训练环境，分割/对齐环境使用 `requirements-Fast-Sam2.txt`。

> 3090 实际安装、完整迁移步骤、实跑修复及两个视频的结果路径见 [DEPLOYMENT_3090_FULL_GUIDE.md](DEPLOYMENT_3090_FULL_GUIDE.md)。

> 当前批处理默认在训练后执行单株空间清理（`TARGET_SPATIAL_RADIUS=0.8`），再用清理后的目标高斯渲染，保留目标大豆与一个魔方。24GiB内存的3090运行设置 `RENDER_VOXEL_SIZE=0.02`；诊断证据、参数及当前结果见完整文档第13节。

> 已进一步完成MASK源头修复：在训练PNG写出前使用SfM实例约束和SAM2正负提示，两个视频304帧新掩码及语义初始化已更新。旧检查点仍对应旧输入，后续重训使用新的输出目录。当前输入、验证结果和命令见 [完整文档第14节](DEPLOYMENT_3090_FULL_GUIDE.md#14-从训练掩码源头修复邻株误标当前正式输入)。

<div align="center">
<h1>FastGS: Training 3D Gaussian Splatting in 100 Seconds</h1> 

[🌐 Homepage](https://fastgs.github.io/) | [📄 Paper](https://arxiv.org/abs/2511.04283) ｜[🤗 Pre-trained model](https://huggingface.co/Goodsleepeverday/fastgs)

</div>

<p align="center">
    <img src="assets/teaser_fastgs.png" width="800px"/>
</p>

<div align="center">
<h1>Fast-PGSR: Accelerating Surface Reconstruction</h1>
</div>


## 🏗️ Training Framework

Our training pipeline leverages **PyTorch** and optimized **CUDA extensions** to efficiently produce high-quality trained models in record time.

### 💻 Hardware Requirements

- **GPU**: CUDA-ready GPU with Compute Capability 7.0+
- **Memory**: 24 GB VRAM (for paper-quality results; we recommend NVIDIA RTX4090)

### 📦 Software Requirements

- **Conda** (recommended for streamlined setup)
- **C++ Compiler** compatible with PyTorch extensions
- **CUDA SDK 11** (or compatible version)
- **⚠️ Important**: Ensure C++ Compiler and CUDA SDK versions are compatible

### ⚠️ CUDA Version Reference

Our testing environment uses the following CUDA configuration:

| Component                             | Version          |
|---------------------------------------|------------------|
| Conda environment CUDA version        | 11.6             |
| Ubuntu system `nvidia-smi` CUDA       | 12.2             |
| `nvcc -V` compiler version            | 11.8 (v11.8.89)  |

> **Note**: The Conda CUDA and system CUDA versions may differ. The compiler version (`nvcc`) is what matters for PyTorch extensions compilation (diff-gaussian-rasterization_fastgs).


## 🚀 Quick Start

### 📥 Clone the Repository

```bash
git clone --branch fast-pgsr https://github.com/fastgs/FastGS.git
cd FastGS
```

### ⚙️ Environment Setup

We provide a streamlined setup using Conda:

```shell
# Windows only
SET DISTUTILS_USE_SDK=1

# Create and activate environment
conda env create --file environment.yaml
conda activate fast-pgsr
```

### 🗂️ Data Preparation

Please download the Tanks & Temples Dataset from the [official website](https://www.tanksandtemples.org/download/).

The data folder should like this:
```shell
data
├── tnt_dataset
│   ├── tnt
│   │   ├── Ignatius
│   │   │   ├── images_raw
│   │   │   ├── Ignatius_COLMAP_SfM.log
│   │   │   ├── Ignatius_trans.txt
│   │   │   ├── Ignatius.json
│   │   │   ├── Ignatius_mapping_reference.txt
│   │   │   └── Ignatius.ply
│   │   └── ...
```

Then run the scripts to preprocess Tanks and Temples dataset:
```shell
# Install COLMAP
Refer to https://colmap.github.io/install.html

# Tanks and Temples dataset
python scripts/preprocess/convert_tnt.py --tnt_path your_tnt_path
```

## 🎯 Training & Evaluation

### ⚡ Fast-PGSR

```bash
# Tanks and Temples dataset
python scripts/run_tnt.py
```
<details>
<summary><span style="font-weight: bold;">📋 Advanced: Command Line Arguments for train.py</span></summary>

  #### --loss_thresh
  Threshold of the loss map; a lower value generally results in more Gaussians being retained.
  #### --grad_abs_thresh 
  Absolute gradient (same as Abs-GS) threshold for split.
  #### --grad_thresh
  Gradient(same as vanilla 3DGS) threshold for clone.
  #### --highfeature_lr
  Learning rate for high-order SH coefficients (features_rest).
  #### --lowfeature_lr
  Learning rate for low-order SH coefficients (features_dc).
  #### --dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified.
  #### --mult 
  Multiplier for the compact box to control the tile number of each splat
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

## 🙏 Acknowledgements

This project is built upon [PGSR](https://github.com/zju3dv/PGSR), [Taming-3DGS](https://github.com/humansensinglab/taming-3dgs), [Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat), and [Abs-GS](https://github.com/TY424/AbsGS). We extend our gratitude to all the authors for their outstanding contributions and excellent repositories!

**License**: Please adhere to the licenses of Deformable-3D-Gaussians, 4DGaussians, Taming-3DGS, Speedy-Splat, and Abs-GS.

Special thanks to the authors of [DashGaussian](https://github.com/YouyuChen0207/DashGaussian) for their generous support!


## Citation
If you find this repo useful, please cite:
```
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}

```

---

<div align="center">

**⭐ If FastGS helps your research, please consider starring this repository!**

*FastGS: Training 3D Gaussian Splatting in 100 Seconds*

</div>

---
