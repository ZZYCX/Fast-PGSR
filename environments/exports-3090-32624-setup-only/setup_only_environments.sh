set -euo pipefail
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=3 FORCE_CUDA=1
export SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export PIP_DEFAULT_TIMEOUT=120
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/Fast-PGSR
conda env create -f environments/environment-fast-pgsr-old.conda.yml
conda env create -f environments/environment-Fast-Sam2.conda.yml
while [ ! -f /root/fast-pgsr-setup-only/wheels.exit ]; do sleep 5; done
test "$(cat /root/fast-pgsr-setup-only/wheels.exit)" = 0
python3 - <<'PY'
from pathlib import Path
for name in ('fast-pgsr-old','Fast-Sam2'):
    p=Path('requirements-'+name+'.txt')
    dest=Path('/root/fast-pgsr-setup-only/requirements-'+name+'-mirror.txt')
    dest.write_text('\n'.join(x for x in p.read_text().splitlines() if not x.startswith('--extra-index-url'))+'\n')
PY
conda activate fast-pgsr-old
python -m pip install --no-deps /root/fast-pgsr-setup-only/wheelhouse/*cp37*.whl
python -m pip install --retries 3 -r /root/fast-pgsr-setup-only/requirements-fast-pgsr-old-mirror.txt
python -m pip install -v --no-build-isolation --no-deps -r requirements-fast-pgsr-old-source.txt
python -m pip freeze --all > /root/fast-pgsr-setup-only/fast-pgsr-old.freeze.txt
conda list --json > /root/fast-pgsr-setup-only/fast-pgsr-old.conda.json
conda activate Fast-Sam2
python -m pip install --no-deps /root/fast-pgsr-setup-only/wheelhouse/*cp310*.whl
python -m pip install --retries 3 -r /root/fast-pgsr-setup-only/requirements-Fast-Sam2-mirror.txt
python -m pip install -v --no-build-isolation --no-deps -r requirements-Fast-Sam2-source.txt
python -m pip freeze --all > /root/fast-pgsr-setup-only/Fast-Sam2.freeze.txt
conda list --json > /root/fast-pgsr-setup-only/Fast-Sam2.conda.json
