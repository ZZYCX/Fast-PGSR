#!/bin/bash
# Configuration only. This file does not start any processing.
_fast_pgsr_conda_base=""
if [[ -r /root/fast-pgsr-new-setup-20261017/conda-base.txt ]]; then
    _fast_pgsr_conda_base=$(cat /root/fast-pgsr-new-setup-20261017/conda-base.txt)
fi
if [[ ! -f "$_fast_pgsr_conda_base/etc/profile.d/conda.sh" ]]; then
    for _fast_pgsr_candidate in "${CONDA_EXE:-}" "$(type -P conda || true)" /root/miniconda3/bin/conda /root/anaconda3/bin/conda /opt/conda/bin/conda; do
        [[ -x "$_fast_pgsr_candidate" ]] || continue
        _fast_pgsr_candidate=$(readlink -f -- "$_fast_pgsr_candidate")
        _fast_pgsr_conda_base=$(dirname -- "$(dirname -- "$_fast_pgsr_candidate")")
        [[ -f "$_fast_pgsr_conda_base/etc/profile.d/conda.sh" ]] && break
    done
fi
if [[ ! -f "$_fast_pgsr_conda_base/etc/profile.d/conda.sh" ]]; then
    echo "Conda setup not found; set CONDA_EXE to an installed conda executable" >&2
    return 1 2>/dev/null || exit 1
fi
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/.conda/envs}"
source "$_fast_pgsr_conda_base/etc/profile.d/conda.sh"
export PATH="$_fast_pgsr_conda_base/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
unset _fast_pgsr_conda_base _fast_pgsr_candidate
