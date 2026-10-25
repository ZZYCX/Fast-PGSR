#!/bin/bash
# Configuration only. This file does not start any processing.
_fast_pgsr_conda_base=$(cat /root/fast-pgsr-new-setup-20261017/conda-base.txt)
export CONDA_ENVS_PATH=/root/.conda/envs
source "$_fast_pgsr_conda_base/etc/profile.d/conda.sh"
export PATH="$_fast_pgsr_conda_base/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen
unset _fast_pgsr_conda_base
