#!/usr/bin/env bash
# Server-only stand-in for `bimanual_system.launch.py crx_mock:=true sharpa_backend:=mock`
# (the server has no FANUC ros2_control plugin). Uses only unmodified nodes from the deployment repos:
# dual_mock_robot.py (ideal arm mock), interpolation_node (quintic, 100 Hz in), dual_sharpa_wave mock hands,
# bimanual_description.launch.py (/bimanual/robot_description). Ctrl+C stops everything.
set -eo pipefail
WS=${WS:-$HOME/ws_fanuc_mock}              # workspace with the deployment packages built
OUT=${OUT:-$(cd "$(dirname "$0")/.." && pwd)/runs/mock_stack}
mkdir -p "$OUT"
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${ROS_ENV:-rosjazzy}"   # conda env with ROS 2 Jazzy (RoboStack)
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-198} ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python - "$OUT/mock.yaml" <<'PY'
import sys, xacro, yaml
from ament_index_python.packages import get_package_share_directory as share
xml = xacro.process_file(share('dual_crx_control') + '/urdf/dual_crx.urdf.xacro').toxml()
open(sys.argv[1], 'w').write(yaml.safe_dump({'/**': {'ros__parameters': {'robot_description': xml}}}))
PY
pids=()
trap 'kill ${pids[@]} 2>/dev/null; wait' EXIT INT TERM
python "$WS/src/dual_crx_control/tools/dual_mock_robot.py" --ros-args --params-file "$OUT/mock.yaml" > "$OUT/arm_mock.log" 2>&1 & pids+=($!)
ros2 run dual_crx_control interpolation_node --ros-args -r __ns:=/crx5ia -p input_rate_hz:=100.0 -p method:=quintic > "$OUT/interp.log" 2>&1 & pids+=($!)
ros2 launch dual_sharpa_wave dual_sharpa.launch.py backend:=mock > "$OUT/hands.log" 2>&1 & pids+=($!)
ros2 launch bimanual_manipulation bimanual_description.launch.py sides:=both rviz:=false > "$OUT/description.log" 2>&1 & pids+=($!)
echo "mock stack up (ROS_DOMAIN_ID=$ROS_DOMAIN_ID); logs in $OUT"
wait
