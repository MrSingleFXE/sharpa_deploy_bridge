#!/usr/bin/env bash
# Stop the server mock stack started by tools/server_mock_stack.sh (server only).
for p in $(pgrep -u "$USER" -f "dual_mock_robot.py|lib/dual_crx_control/interpolation_node|lib/dual_sharpa_wave/hand_node|bimanual_description.launch|dual_sharpa.launch|joint_state_merger|bimanual/robot_state|server_mock_stack.sh"); do
  [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null
done
sleep 2; echo "remaining: $(pgrep -u "$USER" -fc 'dual_mock_robot.py|lib/dual_sharpa_wave/hand_node')"
