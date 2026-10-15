#!/usr/bin/env bash
# One-time install of the tools venv (calibration, camera, point clouds, grasp selection) and FoundationStereo.
#   bash install/install_tools.sh
# Creates (no sudo):
#   $BASE/venv              python3 venv with system site-packages (sees ROS rclpy) + OpenCV, RealSense, scipy, matplotlib
#   $BASE/fs_venv           PyTorch for the RTX 5090 (CUDA 12.8 wheels) + FoundationStereo dependencies
#   $BASE/FoundationStereo  NVIDIA FoundationStereo code + official weights (Google Drive), sha256-checked
set -eo pipefail
BASE=${BASE:-$HOME/bodex_sharpa_deploy}
log() { printf '\n==== %s\n' "$*"; }

log "1/4 tools venv -> $BASE/venv"
[ -x "$BASE/venv/bin/python" ] || python3 -m venv --system-site-packages "$BASE/venv"
"$BASE/venv/bin/pip" install -q "numpy<2" "opencv-contrib-python==4.10.0.84" pyrealsense2 pillow scipy matplotlib pyyaml

log "2/4 FoundationStereo code -> $BASE/FoundationStereo"
[ -d "$BASE/FoundationStereo/.git" ] || git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git "$BASE/FoundationStereo"

log "3/4 fs_venv (PyTorch cu128 for RTX 50xx)"
[ -x "$BASE/fs_venv/bin/python" ] || python3 -m venv "$BASE/fs_venv"
"$BASE/fs_venv/bin/python" -c "import torch" 2>/dev/null || \
  "$BASE/fs_venv/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$BASE/fs_venv/bin/pip" install -q omegaconf timm einops scipy opencv-python-headless imageio pillow gdown "numpy<2"
"$BASE/fs_venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

log "4/4 FoundationStereo weights"
W="$BASE/FoundationStereo/pretrained_models"
mkdir -p "$W"
if [ ! -f "$W/23-51-11/model_best_bp2.pth" ] || [ ! -f "$W/11-33-40/model_best_bp2.pth" ]; then
  (cd "$W" && "$BASE/fs_venv/bin/gdown" --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf) || true
  # gdown keeps the Drive folder name; flatten if needed
  [ -d "$W/pretrained_models" ] && cp -rn "$W/pretrained_models/"* "$W/" && rm -rf "$W/pretrained_models"
fi
ok=1
check() { [ -f "$1" ] && [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] && echo "  ok   $1" || { echo "  MISSING/BAD $1"; ok=0; }; }
check "$W/23-51-11/model_best_bp2.pth" 60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1
check "$W/11-33-40/model_best_bp2.pth" 8d7850b9dc68d1366722a02a39745704b1db41471211be6abd93ef463727e6be
[ $ok = 1 ] || echo "Download the folders 23-51-11 and 11-33-40 by browser from
  https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf  into  $W/  and run this script again."
echo DONE
