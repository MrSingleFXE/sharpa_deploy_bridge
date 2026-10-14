# Optional shortcut for the terminal setups written out in README.md ("terminal types").
#   source env.sh mock launch   ROS + ws_fanuc, domain 198 localhost          (robot launch terminal, NO venv)
#   source env.sh real launch   ROS + ws_fanuc + Sharpa SDK, domain 87        (robot launch terminal, NO venv)
#   source env.sh mock          same as "mock launch" + tools venv           (bridge / calibration tools)
#   source env.sh real          same as "real launch" + tools venv
# The launch terminal must not activate the venv: some deployment scripts use `#!/usr/bin/env python3`.
case "$1" in
  mock) _dom=198 ;;
  real) _dom=87 ;;
  *) echo "usage: source env.sh mock|real [launch]"; return 1 2>/dev/null || exit 1 ;;
esac
source /opt/ros/jazzy/setup.bash
source "${WS_FANUC:-$HOME/ws_fanuc}/install/setup.bash"
if [ "$1" = real ]; then
  export PYTHONPATH="/opt/sharpa-wave-sdk/python${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="/opt/sharpa-wave-sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export ROS_DOMAIN_ID=$_dom
if [ "$1" = mock ]; then export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; else unset ROS_AUTOMATIC_DISCOVERY_RANGE; fi
if [ "$2" != launch ] && [ -f "$HOME/bodex_sharpa_deploy/venv/bin/activate" ]; then
  source "$HOME/bodex_sharpa_deploy/venv/bin/activate"
fi
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "env: $1 ${2:-tools} | ROS_DOMAIN_ID=$ROS_DOMAIN_ID | bimanual_manipulation=$(ros2 pkg prefix bimanual_manipulation 2>/dev/null || echo NOT FOUND) | python=$(command -v python3)"
unset _dom
