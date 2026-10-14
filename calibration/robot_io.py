"""Single-threaded ROS I/O for calibration: right-arm feedback/FK and guarded slow right-arm motion.

Same topics and guards as execute_grasp.py. The hand is never commanded here.
"""

import math
import select
import sys
import time

import numpy as np

ARM = [f'right_J{i}' for i in range(1, 7)]
LEFT = [f'left_J{i}' for i in range(1, 7)]
ARM_TARGETS = '/crx5ia/joint_targets'
HAND_COMMAND = '/sharpa/right_hand/joint_command'
FLANGE = 'right_ee_mount'


class Abort(RuntimeError):
    pass


def _ordered(msg, names):
    if len(msg.name) != len(msg.position):
        return None
    v = dict(zip(msg.name, msg.position))
    if not all(n in v for n in names):
        return None
    q = np.array([v[n] for n in names], float)
    return q if np.isfinite(q).all() else None


class RobotIO:
    def __init__(self, node_name='bodex_calibration', urdf_path=None):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                               qos_profile_sensor_data)
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        self.rclpy, self.JointState = rclpy, JointState
        rclpy.init()
        self.node = rclpy.create_node(node_name)
        self.log = self.node.get_logger()
        self.hand_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                   reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.st = {'arm': None, 'arm_t': -math.inf, 'hand': None, 'hand_names': None, 'hand_t': -math.inf,
                   'left': None, 'urdf': open(urdf_path).read() if urdf_path else None, 'limiter': None}

        def on_arm(m):
            q = _ordered(m, ARM)
            if q is not None:
                self.st['arm'], self.st['arm_t'] = q, time.monotonic()

        def on_hand(m):
            names = [n for n in m.name if n.startswith('right_')]
            if len(names) == 22 and len(m.position) == len(m.name):
                self.st['hand_names'] = names
                self.st['hand'], self.st['hand_t'] = _ordered(m, names), time.monotonic()

        n = self.node
        n.create_subscription(JointState, '/crx5ia/right/joint_states', on_arm, qos_profile_sensor_data)
        n.create_subscription(JointState, '/crx5ia/left/joint_states',
                              lambda m: self.st.__setitem__('left', _ordered(m, LEFT)), qos_profile_sensor_data)
        n.create_subscription(JointState, '/sharpa/right_hand/joint_states', on_hand, self.hand_qos)
        n.create_subscription(String, '/crx5ia/collision_force_limiter/state',
                              lambda m: self.st.__setitem__('limiter', m.data), 10)
        if not urdf_path:
            n.create_subscription(String, '/bimanual/robot_description',
                                  lambda m: self.st.__setitem__('urdf', m.data), latched)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(n)
        self.arm_pub = None

    # ------------------------------------------------------------ basics
    def pump(self, seconds=0.0):
        end = time.monotonic() + seconds
        while True:
            for _ in range(30):
                self.ex.spin_once(timeout_sec=0.0)
            if time.monotonic() >= end:
                return
            time.sleep(0.002)

    def wait_ready(self, timeout=15.0, need_hand=True):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(0.05)
            if self.fresh(need_hand) and self.st['urdf']:
                break
        if not (self.fresh(need_hand) and self.st['urdf']):
            raise Abort(f'missing inputs: arm={self.st["arm"] is not None} hand={self.st["hand"] is not None} '
                        f'urdf={self.st["urdf"] is not None}; is bimanual_system.launch.py running (same ROS_DOMAIN_ID)?')
        end = time.monotonic() + 2.0
        while self.st['left'] is None and time.monotonic() < end:
            self.pump(0.05)

    def fresh(self, need_hand=True):
        now = time.monotonic()
        return now - self.st['arm_t'] < 0.5 and (not need_hand or now - self.st['hand_t'] < 0.5)

    def stationary(self, window=0.4, tol_deg=0.05):
        a = self.st['arm'].copy()
        self.pump(window)
        return np.abs(self.st['arm'] - a).max() < math.radians(tol_deg)

    # ------------------------------------------------------------ motion
    def enable_motion(self):
        self.pump(0.2)
        if self.node.count_publishers(ARM_TARGETS):
            raise Abort('another sender is publishing /crx5ia/joint_targets (keyboard_control?); stop it first')
        self.arm_pub = self.node.create_publisher(self.JointState, ARM_TARGETS, 1)
        end = time.monotonic() + 5
        while time.monotonic() < end and not self.arm_pub.get_subscription_count():
            self.pump(0.05)
        if not self.arm_pub.get_subscription_count():
            raise Abort('interpolation node not subscribed (read_only launch?)')
        if self.st['limiter'] is not None and self.st['limiter'] not in ('ARMED', 'DEGRADED'):
            raise Abort(f'collision limiter state {self.st["limiter"]}; wait for ARMED')
        if self.st['limiter'] is None:
            self.log.warning('no collision limiter state (OK for mock with collision_force_limit:=false)')

    def run_segment(self, seg, rate=100.0, tol_deg=6.0):
        """Stream seg.arm with its quintic timing; hand untouched. Raises Abort on any guard."""
        if self.arm_pub is None:
            raise Abort('motion not enabled')
        if np.abs(self.st['arm'] - seg.arm[0]).max() > math.radians(1):
            raise Abort('arm moved since planning')
        period, tol = 1.0 / rate, math.radians(tol_deg)
        t0 = nxt = time.monotonic()
        try:
            while True:
                self.pump()
                now = time.monotonic()
                if now - self.st['arm_t'] > 0.5:
                    raise Abort(f'arm feedback stale ({now - self.st["arm_t"]:.2f} s)')
                if self.st['limiter'] == 'TRIPPED':
                    raise Abort('collision force limiter TRIPPED')
                q, _ = seg.sample(now - t0)
                m = self.JointState(name=list(ARM), position=[float(v) for v in q])
                m.header.stamp = self.node.get_clock().now().to_msg()
                self.arm_pub.publish(m)
                if np.abs(self.st['arm'] - q).max() > tol:
                    raise Abort(f'arm tracking error {math.degrees(np.abs(self.st["arm"] - q).max()):.1f} deg')
                if now - t0 >= seg.duration + 0.2:
                    return
                nxt += period
                d = nxt - time.monotonic()
                if d > 0:
                    time.sleep(d)
                elif d < -0.1:
                    nxt = time.monotonic()
        except KeyboardInterrupt:
            raise Abort('Ctrl+C')

    def close(self):
        self.ex.shutdown(timeout_sec=1.0)
        self.rclpy.try_shutdown()


def ask(prompt, auto=False):
    """Blocking line input (Enter / s / q). With auto=True returns '' immediately."""
    if auto:
        print(prompt.strip(), flush=True)
        return ''
    print(prompt, end='', flush=True)
    line = sys.stdin.readline()
    if line == '':
        raise Abort('stdin closed')
    return line.strip().lower()


def stdin_ready():
    return bool(select.select([sys.stdin], [], [], 0.0)[0])
