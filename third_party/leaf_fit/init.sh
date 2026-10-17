#!/bin/bash

# Initialize conda for bash shell
conda deactivate

conda remove -n leaf_fit --all -y

conda create -n leaf_fit python=3.10.11 pip -y
conda activate leaf_fit
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
CC=gcc-10 CXX=g++-10 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')" && echo "✓ CUDA验证成功！" || echo "✗ CUDA验证失败"
cd ./libs/diff_gaussian_rasterization && CC=gcc-10 CXX=g++-10 pip install --no-build-isolation -e .
