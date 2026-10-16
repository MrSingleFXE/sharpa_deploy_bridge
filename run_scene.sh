#!/usr/bin/env bash
# One real object, one command: capture -> FoundationStereo -> object cloud -> DexLearn -> select (-> world).
# Never moves the robot. Each stage uses its own environment (see README "终端类型").
#
#   bash run_scene.sh scene01                       # new capture, object = largest thing on the table
#   bash run_scene.sh scene01 --pixel 640 400       # object = the cluster at this color-image pixel
#   bash run_scene.sh scene01 --skip-capture        # reuse captures/scene01 (e.g. after changing --pixel)
#   bash run_scene.sh scene01 --fs-model vits       # faster FoundationStereo (ViT-S); default vitl (best)
#
# World-frame output (step 6) is produced only when calib/camera_extrinsics.json exists.
set -eo pipefail
BASE=${BASE:-$HOME/bodex_sharpa_deploy}
cd "$(dirname "$0")"
SCENE=${1:?usage: bash run_scene.sh <scene_name> [--pixel U V] [--skip-capture] [--depth fs|raw]}; shift
CAP=captures/$SCENE
PIXEL=(); SKIP=0; DEPTH=fs; FSM=vitl
while [ $# -gt 0 ]; do
  case "$1" in
    --pixel) PIXEL=(--pixel "$2" "$3"); shift 3 ;;
    --skip-capture) SKIP=1; shift ;;
    --depth) DEPTH=$2; shift 2 ;;
    --fs-model) FSM=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
VENV_PY=$BASE/venv/bin/python
FS_PY=$BASE/fs_venv/bin/python
step() { printf '\n======== %s\n' "$*"; }

if [ $SKIP = 0 ]; then
  step "1/6 capture -> $CAP  (D415: color, IR pair, RealSense depth)"
  $VENV_PY perception/capture.py --out "$CAP"
fi

if [ "$DEPTH" = fs ]; then
  step "2/6 FoundationStereo depth  (look at $CAP/fs_vis.png)"
  FSR=${FS_ROOT:-$BASE/FoundationStereo}
  CK=$FSR/pretrained_models/23-51-11/model_best_bp2.pth
  [ "$FSM" = vits ] && CK=$FSR/pretrained_models/11-33-40/model_best_bp2.pth
  FS_ROOT=$FSR $FS_PY perception/fs_depth.py "$CAP" --ckpt "$CK"
fi

EXT=()
[ -f calib/camera_extrinsics.json ] && EXT=(--extrinsics calib/camera_extrinsics.json)
step "3/6 object point cloud  (look at $CAP/object_$DEPTH/overlay.png: only the object should be red)"
$VENV_PY perception/object_cloud.py "$CAP" --depth "$DEPTH" "${PIXEL[@]}" "${EXT[@]}"

step "4/6 DexLearn grasps (CPU)"
(
  source "$BASE/miniforge3/bin/activate"
  conda activate "$BASE/miniforge3/envs/dexlearn_cpu"
  python grasping/dexlearn_infer.py "$CAP/object_$DEPTH/object_table.npy" --out "$CAP/grasps" \
    --dexlearn-root "$BASE/DexLearn" --model "$BASE/models/dexlearn_sharpa_v3_s34_50k"
)

step "5/6 filter + rank  (look at $CAP/grasps/selected_views.png)"
$VENV_PY grasping/select_grasps.py "$CAP/grasps/grasps_table.npy" "$CAP/object_$DEPTH/object_table.npy"

if [ ${#EXT[@]} -gt 0 ]; then
  step "6/6 to world frame"
  $VENV_PY grasping/to_world.py "$CAP/grasps/selected.npy" "$CAP/object_$DEPTH/object.json"
else
  step "6/6 skipped: no calib/camera_extrinsics.json yet (do README part B first)"
fi
echo; echo "done: $CAP"
