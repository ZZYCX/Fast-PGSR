set -euo pipefail
B=/root/fast-pgsr-new-setup-20261017
if [ -x /data/miniconda3/bin/conda ]; then
 BASE=/data/miniconda3
 printf "%s\n" "$BASE" > "$B/conda-base.txt"
elif [ -x /root/miniconda3/bin/conda ]; then
 BASE=/root/miniconda3
 printf "%s\n" "$BASE" > "$B/conda-base.txt"
else
 while [ ! -f "$B/system.exit" ]; do sleep 5; done
 [ "$(cat "$B/system.exit")" = 0 ]
 BASE=$(cat "$B/conda-base.txt")
fi
export CONDARC="$B/condarc.yml"
export CONDA_ENVS_PATH=/root/.conda/envs CONDA_PKGS_DIRS="$B/conda-pkgs"
source "$BASE/etc/profile.d/conda.sh"
export PATH=/usr/local/cuda-11.8/bin:/usr/local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="$(cat "$B/gpu-architecture.txt")" MAX_JOBS=3 FORCE_CUDA=1
if [ "$TORCH_CUDA_ARCH_LIST" = 8.9 ]; then export NVCC_PREPEND_FLAGS="-gencode=arch=compute_89,code=sm_89"; fi
export SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple PIP_DEFAULT_TIMEOUT=120
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
cd /root/Fast-PGSR
python3 - <<'PY'
from pathlib import Path
b=Path('/root/fast-pgsr-new-setup-20261017')
for n in ('fast-pgsr-old','Fast-Sam2'):
 p=Path('environments/environment-'+n+'.conda.yml')
 s=p.read_text().replace('- defaults','- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n- nodefaults')
 (b/p.name).write_text(s)
PY
for NAME in fast-pgsr-old Fast-Sam2; do
  if [ ! -f "$B/$NAME.created" ]; then
    conda create -y --prefix "/root/.conda/envs/$NAME" --file "$B/conda-$NAME.explicit-local-cache.txt"
    touch "$B/$NAME.created"
  fi
done
while [ ! -f "$B/system.exit" ]; do sleep 5; done
[ "$(cat "$B/system.exit")" = 0 ]
python3 - <<'PY'
import hashlib,json
from pathlib import Path
b=Path('/root/fast-pgsr-new-setup-20261017')
for name in ('fast-pgsr-old','Fast-Sam2'):
 p=Path('requirements-'+name+'.txt')
 (b/(p.stem+'-mirror.txt')).write_text('\n'.join(x for x in p.read_text().splitlines() if not x.startswith('--extra-index-url'))+'\n')
a=b/'pytorch3d-source.tar.gz'; expected=json.loads((b/'pytorch3d-source.tar.sha256.json').read_text())
assert hashlib.sha256(a.read_bytes()).hexdigest()==expected['sha256']
assert expected['commit']=='995b60e3b99faa1ee1bcdbe244426d54d98a7242'
p=Path('requirements-fast-pgsr-old-source.txt')
(b/'requirements-fast-pgsr-old-source-cached.txt').write_text(p.read_text().replace('https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/'+expected['commit'],a.as_uri()))
PY
while [ ! -f "$B/wheels.exit" ]; do sleep 5; done
[ "$(cat "$B/wheels.exit")" = 0 ]
for NAME in fast-pgsr-old Fast-Sam2; do
 conda activate "/root/.conda/envs/$NAME"
 if [ ! -f "$B/$NAME.ordinary" ]; then
   if [ "$NAME" = fast-pgsr-old ]; then ABI=cp37; else ABI=cp310; fi
   python -m pip install --force-reinstall --no-deps "$B"/wheelhouse/torch-*-$ABI-*.whl "$B"/wheelhouse/torchvision-*-$ABI-*.whl "$B"/wheelhouse/torchaudio-*-$ABI-*.whl
   python -m pip install --no-index --find-links "$B/wheelhouse" --no-build-isolation --force-reinstall --no-deps -r "$B/requirements-$NAME-mirror.txt"
   touch "$B/$NAME.ordinary"
 fi
 if [ "$NAME" = fast-pgsr-old ] && [ "$TORCH_CUDA_ARCH_LIST" = 8.9 ]; then
   python - <<'PY'
from pathlib import Path
import sys,shutil,json,hashlib
p=Path(sys.prefix)/'lib/python3.7/site-packages/torch/utils/cpp_extension.py'
b=Path('/root/fast-pgsr-new-setup-20261017')
backup=b/'cpp_extension.py.original'
if not backup.exists():shutil.copy2(p,backup)
s=backup.read_text();old="'7.0', '7.2', '7.5', '8.0', '8.6']"
assert old in s
p.write_text(s.replace(old,"'7.0', '7.2', '7.5', '8.0', '8.6', '8.9']"))
(b/'torch-build-architecture-adjustment.json').write_text(json.dumps({'file':str(p),'change':'temporarily allow 8.9 in build helper; nvcc 11.8 generates native sm_89; restore after build','before_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'during_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
PY
 fi
 if [ ! -f "$B/$NAME.source" ]; then
   if [ "$NAME" = fast-pgsr-old ]; then REQ="$B/requirements-fast-pgsr-old-source-cached.txt"; else REQ=requirements-Fast-Sam2-source.txt; fi
   python -m pip install -v --no-build-isolation --no-deps -r "$REQ"
   touch "$B/$NAME.source"
 fi
 if [ "$NAME" = fast-pgsr-old ] && [ -f "$B/cpp_extension.py.original" ]; then
    cp "$B/cpp_extension.py.original" "$CONDA_PREFIX/lib/python3.7/site-packages/torch/utils/cpp_extension.py"
    sha256sum "$CONDA_PREFIX/lib/python3.7/site-packages/torch/utils/cpp_extension.py" > "$B/cpp-extension-restored.sha256"
 fi
 python -m pip freeze --all > "$B/$NAME.freeze.txt"
 conda list --json --prefix "$CONDA_PREFIX" > "$B/$NAME.conda.json"
 python --version > "$B/$NAME.python-version.txt"
 conda list --explicit --prefix "$CONDA_PREFIX" > "$B/$NAME.conda-explicit.txt"
done
