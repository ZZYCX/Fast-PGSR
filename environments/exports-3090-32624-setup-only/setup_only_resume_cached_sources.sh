set -euo pipefail
export PATH=/root/miniconda3/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=3 FORCE_CUDA=1
export SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple PIP_DEFAULT_TIMEOUT=120
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/Fast-PGSR
python3 - <<'PY'
import hashlib,json,pathlib
base=pathlib.Path('/root/fast-pgsr-setup-only')
expected=json.loads((base/'pytorch3d-source.tar.sha256.json').read_text())
archive=base/'pytorch3d-source.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected['sha256']
assert expected['commit']=='995b60e3b99faa1ee1bcdbe244426d54d98a7242'
text=pathlib.Path('requirements-fast-pgsr-old-source.txt').read_text()
text=text.replace('https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/995b60e3b99faa1ee1bcdbe244426d54d98a7242',archive.as_uri())
(base/'requirements-fast-pgsr-old-source-cached.txt').write_text(text)
print('Fixed-commit PyTorch3D source archive SHA-256 matched:',expected['sha256'])
PY
conda activate fast-pgsr-old
python -m pip install -v --no-build-isolation --no-deps -r /root/fast-pgsr-setup-only/requirements-fast-pgsr-old-source-cached.txt
python -m pip freeze --all > /root/fast-pgsr-setup-only/fast-pgsr-old.freeze.txt
conda list --json > /root/fast-pgsr-setup-only/fast-pgsr-old.conda.json
conda activate Fast-Sam2
python -m pip install --no-deps /root/fast-pgsr-setup-only/wheelhouse/torch-2.7.1+cu118-cp310*.whl /root/fast-pgsr-setup-only/wheelhouse/torchvision-0.22.1+cu118-cp310*.whl /root/fast-pgsr-setup-only/wheelhouse/torchaudio-2.7.1+cu118-cp310*.whl
python -m pip install --retries 3 -r /root/fast-pgsr-setup-only/requirements-Fast-Sam2-mirror.txt
python -m pip install -v --no-build-isolation --no-deps -r requirements-Fast-Sam2-source.txt
python -m pip freeze --all > /root/fast-pgsr-setup-only/Fast-Sam2.freeze.txt
conda list --json > /root/fast-pgsr-setup-only/Fast-Sam2.conda.json
