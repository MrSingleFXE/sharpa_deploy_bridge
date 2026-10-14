#!/usr/bin/env python3
"""Record right arm/hand feedback and the interpolator output to an .npz (for mock verification)."""
import sys, time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

out, duration = sys.argv[1], float(sys.argv[2])
rclpy.init()
node = rclpy.create_node('bridge_state_recorder')
data = {'arm': [], 'hand': [], 'cmd': [], 'hand_cmd': []}

def rec(key, names_prefix):
    def cb(m):
        if m.name and m.name[0].startswith(names_prefix):
            data[key].append([time.monotonic()] + list(m.position))
    return cb

node.create_subscription(JointState, '/crx5ia/right/joint_states', rec('arm', 'right_J'), qos_profile_sensor_data)
node.create_subscription(JointState, '/sharpa/right_hand/joint_states', rec('hand', 'right_'), 10)
node.create_subscription(JointState, '/sharpa/right_hand/joint_command', rec('hand_cmd', 'right_'), 10)
node.create_subscription(JointState, '/crx5ia/interpolated_joint_commands', rec('cmd', ''), 10)
end = time.monotonic() + duration
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)
np.savez(out, **{k: np.array(v) for k, v in data.items()})
print({k: len(v) for k, v in data.items()})
