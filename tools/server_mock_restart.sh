#!/usr/bin/env bash
# Stop and restart the server mock stack (server only).
for p in $(pgrep -u "$USER" -f "dual_mock_robot.py|lib/dual_crx_control/interpolation_node|hand_node|bimanual_description.launch|dual_sharpa.launch|joint_state_merger|bimanual/robot_state|server_mock_stack.sh"); do
  [ "$p" != "$$" ] && kill "$p" 2>/dev/null
done
sleep 3
cd "$(dirname "$0")/.." && setsid nohup tools/server_mock_stack.sh > runs/mock_stack.out 2>&1 < /dev/null &
sleep 25
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${ROS_ENV:-rosjazzy}"; source "${WS:-$HOME/ws_fanuc_mock}/install/setup.bash"
export ROS_DOMAIN_ID=198 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
timeout 10 ros2 topic list | grep -E "right|robot_description"
