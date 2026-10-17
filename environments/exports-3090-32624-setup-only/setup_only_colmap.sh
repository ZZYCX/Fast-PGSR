set -euo pipefail
export PATH=/usr/local/cuda-11.8/bin:$PATH
cd /root/fast-pgsr-setup-only
mkdir -p src
cd src
curl -fL --retry 3 https://codeload.github.com/colmap/colmap/tar.gz/refs/tags/3.9.1 -o colmap-3.9.1.tar.gz
tar -xzf colmap-3.9.1.tar.gz
cmake -S colmap-3.9.1 -B colmap-3.9.1/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build colmap-3.9.1/build -j 3
cmake --install colmap-3.9.1/build
