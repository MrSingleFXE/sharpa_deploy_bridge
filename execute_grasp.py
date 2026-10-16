#!/usr/bin/env python3
"""Execute one BODex Sharpa grasp (pregrasp -> grasp -> squeeze) on the right CRX arm + right Sharpa hand.

Decoupled from the deployment workspace: this script only uses the public ROS topics
of the already running `bimanual_manipulation` launch (mock or real):

  read : /bimanual/robot_description, /crx5ia/right/joint_states, /sharpa/right_hand/joint_states,
         /crx5ia/collision_force_limiter/state (optional)
  write: /crx5ia/joint_targets (right_J1..J6), /sharpa/right_hand/joint_command (22 joints)

Default is a DRY RUN (plan + safety report only). Add --execute to move; every stage asks for Enter.
Ctrl+C stops sending new targets. It is NOT an emergency stop; keep the pendant E-stop at hand.
"""

import argparse
import json
import math
import sys
import select
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge.planner import (ARM, Limits, PlanError, Planner, load_grasp, load_hand_mount,  # noqa: E402
                            override_hand_mount)

ARM_TARGETS = '/crx5ia/joint_targets'
ARM_STATES = '/crx5ia/right/joint_states'
HAND_COMMAND = '/sharpa/right_hand/joint_command'
HAND_STATES = '/sharpa/right_hand/joint_states'
DESCRIPTION = '/bimanual/robot_description'
LIMITER = '/crx5ia/collision_force_limiter/state'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grasp', required=True, help='BODex succgrasp *_grasp.npy or equivalent .json')
    p.add_argument('--index', type=int, default=0, help='row if the file stores a batch')
    p.add_argument('--stop-after', choices=['pregrasp', 'grasp', 'squeeze'], default='squeeze')
    p.add_argument('--release', action='store_true', help='afterwards open to pregrasp and lift')
    p.add_argument('--wrist-frame', choices=['scene', 'world'], default='scene',
                   help='scene: place the BODex scene so the grasp wrist lands on --anchor (in-air test); '
                        'world: wrist poses are already in the deployment world frame')
    p.add_argument('--anchor', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   help='world point for the grasp wrist (default: current right_hand_C_MC position)')
    p.add_argument('--anchor-offset', type=float, nargs=3, default=[0, 0, 0.15], metavar=('DX', 'DY', 'DZ'),
                   help='added to the anchor (default: 15 cm up, so the in-air test stays high)')
    p.add_argument('--yaw-deg', type=float, nargs='*', help='scene yaw(s) to try (default 0..345 step 15)')
    p.add_argument('--table-z', type=float, help='table top height in world (m); default: table_z_m of '
                                                 'calib/verify/verify.json (README B.7) if it exists, else 0')
    p.add_argument('--clearance', type=float, default=0.10, help='min height of any moving link above table (m)')
    p.add_argument('--approach-lift', type=float, default=0.08)
    p.add_argument('--arm-speed-deg', type=float, default=5.0)
    p.add_argument('--arm-accel-deg', type=float, default=10.0)
    p.add_argument('--hand-speed-deg', type=float, default=10.0)
    p.add_argument('--hand-accel-deg', type=float, default=20.0)
    p.add_argument('--min-duration', type=float, default=3.0)
    p.add_argument('--rate', type=float, default=100.0, help='must match the launch input_rate_hz')
    p.add_argument('--hold', type=float, default=3.0, help='seconds to keep streaming the final hand target')
    p.add_argument('--urdf', help='use this URDF file instead of /bimanual/robot_description')
    p.add_argument('--hand-mount', help='calibrated hand mount (.json/.yaml, hand_mounts.yaml format); replaces the '
                                        'arm-flange -> hand-flange transform in the PLANNING model only; default: '
                                        'calib/hand_mount.json (README C) if it exists; "none" = lab placeholder')
    p.add_argument('--obstacles', help='object points in world (.npy, e.g. object_world.npy): the hand must keep '
                                       '--obstacle-clearance from them until it reaches pregrasp')
    p.add_argument('--obstacle-clearance', type=float, default=0.02)
    p.add_argument('--plan-out', default='grasp_plan.json')
    p.add_argument('--execute', action='store_true', help='actually publish motion')
    p.add_argument('--yes', action='store_true', help='do not pause before each stage')
    p.add_argument('--arm-track-tol-deg', type=float, default=6.0)
    p.add_argument('--hand-track-tol-deg', type=float, default=20.0)
    a = p.parse_args(argv)
    calib = Path(__file__).resolve().parent / 'calib'
    if a.table_z is None:
        f = calib / 'verify' / 'verify.json'
        tz = json.loads(f.read_text()).get('table', {}).get('table_z_m') if f.exists() else None
        a.table_z = float(tz) if tz is not None else 0.0
        print(f'table z = {a.table_z} m (from {f})' if tz is not None else 'table z = 0 m (no calibration result; assumption)')
    if a.hand_mount is None and (calib / 'hand_mount.json').exists():
        a.hand_mount = str(calib / 'hand_mount.json')
        print(f'hand mount = {a.hand_mount} (default; --hand-mount none to use the lab placeholder)')
    elif a.hand_mount == 'none':
        a.hand_mount = None
    return a


def limits_from(args):
    return Limits(rate_hz=args.rate, arm_vel=math.radians(args.arm_speed_deg),
                  arm_acc=math.radians(args.arm_accel_deg), hand_vel=math.radians(args.hand_speed_deg),
                  hand_acc=math.radians(args.hand_accel_deg), min_duration=args.min_duration,
                  table_z=args.table_z, clearance=args.clearance, approach_lift=args.approach_lift)


def ordered(msg, names):
    if len(msg.name) != len(msg.position):
        return None
    values = dict(zip(msg.name, msg.position))
    if not all(n in values for n in names):
        return None
    q = np.array([values[n] for n in names], dtype=float)
    return q if np.isfinite(q).all() else None


def main(argv=None):
    args = parse_args(argv)
    limits = limits_from(args)
    limits.validate()
    grasp = load_grasp(args.grasp, args.index)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                           qos_profile_sensor_data)
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    hand_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                          reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)

    rclpy.init()
    node = Node('bodex_grasp_executor')
    log = node.get_logger()
    LEFT = [f'left_J{i}' for i in range(1, 7)]
    st = {'left': None, 'arm': None, 'arm_t': -math.inf, 'hand': None, 'hand_t': -math.inf,
          'urdf': None, 'limiter': None}

    def on_arm(msg):
        q = ordered(msg, ARM)
        if q is not None:
            st['arm'], st['arm_t'] = q, time.monotonic()

    def on_hand(msg):
        q = ordered(msg, grasp.joint_names)
        if q is not None:
            st['hand'], st['hand_t'] = q, time.monotonic()

    def on_left(msg):
        q = ordered(msg, LEFT)
        if q is not None:
            st['left'] = q

    node.create_subscription(JointState, ARM_STATES, on_arm, qos_profile_sensor_data)
    node.create_subscription(JointState, '/crx5ia/left/joint_states', on_left, qos_profile_sensor_data)
    node.create_subscription(JointState, HAND_STATES, on_hand, hand_qos)
    node.create_subscription(String, LIMITER, lambda m: st.__setitem__('limiter', m.data), 10)
    if args.urdf:
        st['urdf'] = Path(args.urdf).read_text()
    else:
        node.create_subscription(String, DESCRIPTION, lambda m: st.__setitem__('urdf', m.data), latched)

    # Single-threaded on purpose: callbacks, publishing and checks all run in one loop.
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def pump(seconds=0.0):
        """Run all ready callbacks; if seconds > 0 keep spinning that long."""
        end = time.monotonic() + seconds
        while True:
            executor.spin_once(timeout_sec=0.0)
            for _ in range(30):                      # drain a burst without blocking
                executor.spin_once(timeout_sec=0.0)
            if time.monotonic() >= end:
                return
            time.sleep(0.002)

    def fresh():
        now = time.monotonic()
        return now - st['arm_t'] < 0.5 and now - st['hand_t'] < 0.5

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (fresh() and st['urdf']):
        pump(0.05)
    if not (fresh() and st['urdf']):
        log.error(f'missing inputs: arm={st["arm"] is not None} hand={st["hand"] is not None} '
                  f'urdf={st["urdf"] is not None}. Is bimanual_system.launch.py running (same ROS_DOMAIN_ID)?')
        return shutdown(rclpy, executor, 2)
    left_deadline = time.monotonic() + 2.0
    while st['left'] is None and time.monotonic() < left_deadline:
        pump(0.05)

    arm_now, hand_now = st['arm'].copy(), st['hand'].copy()
    try:
        urdf = st['urdf']
        mount = None
        if args.hand_mount:
            mount = load_hand_mount(args.hand_mount)
            urdf = override_hand_mount(urdf, *mount)
            log.info(f'hand mount for planning: xyz {mount[0]} rpy {mount[1]} ({args.hand_mount})')
        planner = Planner(urdf, grasp.joint_names, limits)
        if args.obstacles:
            planner.set_obstacles(np.load(args.obstacles), args.obstacle_clearance)
        if st['left'] is not None:
            planner.set_left_arm(st['left'])
        elif any(n.startswith('left_J') for n in planner.m.joints):
            log.warning('no left arm feedback; left-arm distance check assumes left joints = 0')
        plan = planner.plan(grasp, arm_now, hand_now, anchor=args.anchor, anchor_offset=args.anchor_offset,
                            wrist_frame=args.wrist_frame, stop_after=args.stop_after, release=args.release,
                            yaw_list=args.yaw_deg)
    except (PlanError, ValueError) as e:
        log.error(f'PLAN REJECTED: {e}')
        return shutdown(rclpy, executor, 3)

    report = dict(plan.report, grasp=grasp.source, limiter_state=st['limiter'],
                  hand_mount=None if not args.hand_mount else {'file': args.hand_mount, 'xyz': mount[0], 'rpy': mount[1]},
                  obstacles=args.obstacles,
                  limits={k: getattr(limits, k) for k in ('rate_hz', 'table_z', 'clearance', 'approach_lift')},
                  speeds_deg={'arm_vel': args.arm_speed_deg, 'arm_acc': args.arm_accel_deg,
                              'hand_vel': args.hand_speed_deg, 'hand_acc': args.hand_accel_deg},
                  start={'arm_deg': np.round(np.degrees(arm_now), 2).tolist(),
                         'hand_deg': np.round(np.degrees(hand_now), 2).tolist(),
                         'left_arm_deg': None if st['left'] is None else np.round(np.degrees(st['left']), 2).tolist()})
    Path(args.plan_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not args.execute:
        print(f'\nDRY RUN only. Plan saved to {args.plan_out}. Add --execute to move.')
        return shutdown(rclpy, executor, 0)

    # ---------------------------------------------------------- execution guards
    pump(0.2)
    if node.count_publishers(ARM_TARGETS) or node.count_publishers(HAND_COMMAND):
        log.error('another sender is already publishing arm targets or right-hand commands; stop it first')
        return shutdown(rclpy, executor, 4)
    arm_pub = node.create_publisher(JointState, ARM_TARGETS, 1)
    hand_pub = node.create_publisher(JointState, HAND_COMMAND, hand_qos)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not (arm_pub.get_subscription_count() and hand_pub.get_subscription_count()):
        pump(0.05)
    if not (arm_pub.get_subscription_count() and hand_pub.get_subscription_count()):
        log.error('interpolation node or right hand_node not subscribed (read_only launch?)')
        return shutdown(rclpy, executor, 4)
    if st['limiter'] is not None and st['limiter'] not in ('ARMED', 'DEGRADED'):
        log.error(f'collision limiter state is {st["limiter"]}; wait for ARMED')
        return shutdown(rclpy, executor, 4)
    if st['limiter'] is None:
        log.warning('no collision limiter state received (OK for mock with collision_force_limit:=false)')
    if (np.abs(st['arm'] - arm_now).max() > math.radians(1) or
            np.abs(st['hand'] - hand_now).max() > math.radians(3)):
        log.error('robot moved since planning; rerun')
        return shutdown(rclpy, executor, 5)

    tol_arm, tol_hand = math.radians(args.arm_track_tol_deg), math.radians(args.hand_track_tol_deg)
    period = 1.0 / args.rate

    def publish(arm_q, hand_q):
        stamp = node.get_clock().now().to_msg()
        if arm_q is not None:
            m = JointState(name=list(ARM), position=[float(v) for v in arm_q])
            m.header.stamp = stamp
            arm_pub.publish(m)
        m = JointState(name=list(grasp.joint_names), position=[float(v) for v in hand_q])
        m.header.stamp = stamp
        hand_pub.publish(m)

    def enter_pressed():
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return False
        if sys.stdin.readline() == '':
            raise EOFError('stdin closed')
        return True

    segs = list(plan.segments)
    k, moving, seg_t0, hold_until = 0, False, 0.0, None
    arm_cmd, hand_cmd = None, segs[0].hand[0].copy()   # measured pose: satisfies the MIT 0.1 rad takeover check
    abort, nxt, max_gap = None, time.monotonic(), 0.0
    last_loop = time.monotonic()

    def announce():
        seg = segs[k]
        if args.yes:
            print(f'[{seg.name}] {seg.duration:.1f} s', flush=True)
        else:
            print(f'[{seg.name}] {seg.duration:.1f} s -- press Enter to run, Ctrl+C to stop ', end='', flush=True)

    announce()
    try:
        while True:
            pump()
            now = time.monotonic()
            max_gap, last_loop = max(max_gap, now - last_loop), now
            if now - st['arm_t'] > 0.5 or now - st['hand_t'] > 0.5:
                abort = f'feedback stale: arm age {now - st["arm_t"]:.2f} s, hand age {now - st["hand_t"]:.2f} s'
                break
            if st['limiter'] == 'TRIPPED':
                abort = 'collision force limiter TRIPPED'
                break
            if hold_until is not None:
                if now >= hold_until:
                    break
            elif not moving:
                if args.yes or enter_pressed():
                    moving, seg_t0 = True, now
            if moving:
                seg = segs[k]
                arm_cmd, hand_cmd = seg.sample(now - seg_t0)
                publish(arm_cmd, hand_cmd)
                if now - seg_t0 >= seg.duration + 0.2:
                    arm_cmd, hand_cmd = seg.arm[-1].copy(), seg.hand[-1].copy()
                    moving, k = False, k + 1
                    if k == len(segs):
                        hold_until = now + args.hold
                    else:
                        announce()
            else:
                publish(None, hand_cmd)              # keep the hand stream alive between stages
            if arm_cmd is not None and np.abs(st['arm'] - arm_cmd).max() > tol_arm:
                abort = f'arm tracking error {math.degrees(np.abs(st["arm"] - arm_cmd).max()):.1f} deg'
                break
            if np.abs(st['hand'] - hand_cmd).max() > tol_hand:
                abort = f'hand tracking error {math.degrees(np.abs(st["hand"] - hand_cmd).max()):.1f} deg'
                break
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:
                nxt = time.monotonic()               # do not burst after a stall
    except KeyboardInterrupt:
        abort = 'Ctrl+C'
    except EOFError as e:
        abort = str(e)
    print()
    if abort:
        log.error(f'STOPPED during {segs[min(k, len(segs) - 1)].name}: {abort}. No more targets are sent; '
                  'the arm holds its last accepted target and the hand its last MIT target.')
        return shutdown(rclpy, executor, 6)
    print('final arm (deg):', np.round(np.degrees(st['arm']), 2).tolist())
    print('final hand err vs target (deg max):', round(math.degrees(np.abs(st['hand'] - segs[-1].hand[-1]).max()), 2))
    print(f'longest control-loop gap: {max_gap * 1000:.0f} ms')
    print('Done. The hand keeps its last MIT target. To go home: '
          'ros2 run bimanual_manipulation move_to_default_pose.py --execute')
    return shutdown(rclpy, executor, 0)


def shutdown(rclpy, executor, code):
    sys.stdout.flush()
    executor.shutdown(timeout_sec=1.0)
    rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
