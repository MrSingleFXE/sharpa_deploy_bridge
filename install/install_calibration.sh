#!/usr/bin/env bash
# One-time build of ROS-Industrial industrial_calibration (hand-eye calibration) for the deployment host.
#   bash install/install_calibration.sh
# Builds in its OWN colcon workspace $BASE/ws_calib (never touches ~/ws_fanuc):
#   industrial_calibration_ros2 1.2.0  (ROS 2 data collection nodes + RViz panel; package industrial_calibration_ros)
#   industrial_calibration 1.2.0       (core library + extrinsic hand-eye calibration GUI app)
#   ros_industrial_cmake_boilerplate 0.5.4, boost_plugin_loader 0.2.1 (pinned by industrial_calibration_ros2)
#   sharpa_ic_tools (this repo)        (headless runner ic_extrinsic_cli, same API calls as the official example)
# Also installs the RealSense ROS 2 driver (realsense2_camera) used to publish the colour image + camera_info.
# Needs sudo for apt / rosdep (system packages: Ceres, Qt5, OpenCV 4.6 dev, realsense2_camera, ...).
set -eo pipefail
BASE=${BASE:-$HOME/bodex_sharpa_deploy}
WS=$BASE/ws_calib
BRIDGE=$(cd "$(dirname "$0")/.." && pwd)
log() { printf '\n==== %s\n' "$*"; }
if [ -n "$VIRTUAL_ENV" ]; then echo "deactivate the Python venv first (run: deactivate)"; exit 1; fi

log "1/5 system packages (sudo)"
sudo apt-get update -qq
sudo apt-get install -y python3-vcstool python3-colcon-common-extensions python3-rosdep ros-jazzy-realsense2-camera
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update --rosdistro jazzy > /dev/null

log "2/5 sources -> $WS/src"
mkdir -p "$WS/src"
cd "$WS/src"
[ -d industrial_calibration_ros2 ] || git clone -b 1.2.0 https://github.com/ros-industrial/industrial_calibration_ros2.git
vcs import --skip-existing < industrial_calibration_ros2/dependencies.repos
ln -sfn "$BRIDGE/ic_tools" sharpa_ic_tools
for d in industrial_calibration industrial_calibration_ros2 ros_industrial_cmake_boilerplate boost_plugin_loader; do
  echo "  $d $(git -C $d describe --tags --always 2>/dev/null)"
done

log "3/5 dependencies (rosdep, sudo)"
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths . --ignore-src --rosdistro jazzy -y -r || true

log "4/5 colcon build (5-15 min)"
cd "$WS"
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DINDUSTRIAL_CALIBRATION_BUILD_EXAMPLES=OFF \
  2>&1 | tail -15

log "5/5 check"
source "$WS/install/setup.bash"
ros2 pkg prefix industrial_calibration_ros
ls "$WS/install/industrial_calibration/bin/" | grep -i hand_eye
ls "$WS/install/sharpa_ic_tools/bin/ic_extrinsic_cli"
ros2 pkg prefix realsense2_camera
echo "DONE. In calibration terminals: source /opt/ros/jazzy/setup.bash && source $WS/install/setup.bash"
