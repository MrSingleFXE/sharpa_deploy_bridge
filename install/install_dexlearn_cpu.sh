#!/usr/bin/env bash
# One-time install of a CPU-only DexLearn inference environment (no sudo, everything under $BASE).
#   - Miniforge (if not present) in $BASE/miniforge3
#   - conda env dexlearn_cpu: Python 3.10, PyTorch 2.0.1 (CPU), its own gcc 11 + OpenBLAS (same versions as training)
#   - DexLearn code (public fork, same commit as the trained model) with pinned submodules
#   - MinkowskiEngine 0.5.4 (CPU build), pytorch3d 0.7.5 (CPU build), nflows + diffusers from the pinned submodules
# Takes about 20-40 min (mostly compiling). Re-running is safe: finished steps are skipped.
#   bash install/install_dexlearn_cpu.sh
set -eo pipefail
BASE=${BASE:-$HOME/bodex_sharpa_deploy}
CONDA_BASE=${CONDA_BASE:-$BASE/miniforge3}
ENV=${ENV_PREFIX:-$CONDA_BASE/envs/dexlearn_cpu}
DEXLEARN=${DEXLEARN_ROOT:-$BASE/DexLearn}
JOBS=${MAX_JOBS:-8}
log() { printf '\n==== %s\n' "$*"; }

if [ ! -x "$CONDA_BASE/bin/conda" ]; then
  log "1/7 Miniforge -> $CONDA_BASE"
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_BASE"
else
  log "1/7 Miniforge already at $CONDA_BASE"
fi
source "$CONDA_BASE/bin/activate"

if [ ! -x "$ENV/bin/python" ]; then
  log "2/7 conda env $ENV (python 3.10, gcc 11, openblas)"
  conda create -y -q -p "$ENV" -c conda-forge python=3.10 pip "gxx_linux-64=11" "gcc_linux-64=11" openblas \
    "libblas=*=*openblas" ninja git
else
  log "2/7 conda env exists: $ENV"
fi
conda activate "$ENV"
# the env's own C++ runtime first (MinkowskiEngine is compiled with the env's gcc 11)
if [ "$(conda env config vars list 2>/dev/null | grep -c LD_LIBRARY_PATH)" = 0 ]; then
  conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib" > /dev/null
  conda deactivate; conda activate "$ENV"
fi
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ MAX_JOBS=$JOBS

log "3/7 PyTorch 2.0.1 CPU + python packages"
python -c "import torch; assert torch.__version__.startswith('2.0.1')" 2>/dev/null || \
  pip install -q torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q "setuptools<70" wheel "numpy<2" omegaconf hydra-core einops scipy "huggingface_hub<0.26" fvcore iopath

log "4/7 DexLearn code -> $DEXLEARN"
[ -d "$DEXLEARN/.git" ] || git clone --recurse-submodules https://github.com/MrSingleFXE/DexLearn.git "$DEXLEARN"
git -C "$DEXLEARN" submodule update --init --recursive
echo "DexLearn $(git -C "$DEXLEARN" rev-parse --short HEAD), nflows $(git -C "$DEXLEARN/third_party/nflows" rev-parse --short HEAD), MinkowskiEngine $(git -C "$DEXLEARN/third_party/MinkowskiEngine" rev-parse --short HEAD)"
python -c "import nflows" 2>/dev/null || pip install -q -e "$DEXLEARN/third_party/nflows"
python -c "import diffusers" 2>/dev/null || pip install -q -e "$DEXLEARN/third_party/diffusers"

log "5/7 MinkowskiEngine (CPU build, a few minutes)"
if ! python -c "import MinkowskiEngine" 2>/dev/null; then
  (cd "$DEXLEARN/third_party/MinkowskiEngine" && python setup.py install --cpu_only --blas=openblas \
     --blas_include_dirs="$CONDA_PREFIX/include" --blas_library_dirs="$CONDA_PREFIX/lib" > "$BASE/minkowski_build.log" 2>&1) \
    || { tail -30 "$BASE/minkowski_build.log"; echo "MinkowskiEngine build failed, full log: $BASE/minkowski_build.log"; exit 1; }
fi

log "6/7 pytorch3d 0.7.5 (CPU build, 10-20 min)"
if ! python -c "import pytorch3d.transforms" 2>/dev/null; then
  FORCE_CUDA=0 pip install --no-build-isolation -q "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5" \
    > "$BASE/pytorch3d_build.log" 2>&1 || { tail -30 "$BASE/pytorch3d_build.log"; echo "pytorch3d build failed, log: $BASE/pytorch3d_build.log"; exit 1; }
fi

log "7/7 check"
cd "$DEXLEARN" && python - <<'PY'
import torch, MinkowskiEngine as ME, pytorch3d.transforms, nflows, diffusers
import dexlearn.network.models as m
print('OK torch', torch.__version__, '| MinkowskiEngine', ME.__version__, '(CPU)' if not ME.is_cuda_available() else '(CUDA)',
      '| model class', m.BaseModel.__name__)
PY
echo "DONE. Python for DexLearn: $ENV/bin/python"
