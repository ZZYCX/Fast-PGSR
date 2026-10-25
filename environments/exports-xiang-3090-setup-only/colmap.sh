set -euo pipefail
GPU_ARCH_DOT=$(cat /root/fast-pgsr-new-setup-20261017/gpu-architecture.txt)
export PATH=/usr/local/cuda-11.8/bin:$PATH
cd /root/fast-pgsr-new-setup-20261017
mkdir -p src
cd src
if [ -f /root/fast-pgsr-new-setup-20261017/colmap-3.9.1.tar.gz ]; then
 cp /root/fast-pgsr-new-setup-20261017/colmap-3.9.1.tar.gz .
else
 curl -fL --retry 3 https://codeload.github.com/colmap/colmap/tar.gz/refs/tags/3.9.1 -o colmap-3.9.1.tar.gz
fi
tar -xzf colmap-3.9.1.tar.gz
cmake -S colmap-3.9.1 -B colmap-3.9.1/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH_DOT/./}" -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build colmap-3.9.1/build -j 3
cmake --install colmap-3.9.1/build
