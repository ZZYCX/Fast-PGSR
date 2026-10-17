set -euo pipefail
mkdir -p /root/fast-pgsr-setup-only
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ffmpeg build-essential cmake ninja-build pkg-config wget curl ca-certificates git libgl1 libglib2.0-0 libeigen3-dev libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libgmp-dev libmpfr-dev
if [ ! -e /root/miniconda3 ]; then
  curl -fL --retry 3 https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -o /root/fast-pgsr-setup-only/miniconda-installer.sh
  bash /root/fast-pgsr-setup-only/miniconda-installer.sh -b -p /root/miniconda3
  /root/miniconda3/bin/conda config --set auto_activate_base false
fi
