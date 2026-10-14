#!/usr/bin/env python3
"""Plan a BODex grasp offline (no ROS): same planner and safety checks as execute_grasp.py.

Start pose defaults to the deployment default pose (right arm -90,-30,210,0,-60,0 deg; hand zero).
Writes the plan report and, with --samples, the full timed joint trajectory at --rate.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge.planner import Planner, load_grasp  # noqa: E402
from execute_grasp import limits_from, parse_args as exec_args  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', required=True)
    p.add_argument('--arm-start-deg', type=float, nargs=6, default=[-90, -30, 210, 0, -60, 0])
    p.add_argument('--left-start-deg', type=float, nargs=6, default=[0, 30, -30, 0, 60, 0])
    p.add_argument('--samples', help='write timed trajectory .npz here')
    own, rest = p.parse_known_args()
    args = exec_args(rest + ['--urdf', own.urdf])
    grasp = load_grasp(args.grasp, args.index)
    urdf = Path(own.urdf).read_text()
    if args.hand_mount:
        from bridge.planner import load_hand_mount, override_hand_mount
        urdf = override_hand_mount(urdf, *load_hand_mount(args.hand_mount))
    planner = Planner(urdf, grasp.joint_names, limits_from(args))
    if args.obstacles:
        planner.set_obstacles(np.load(args.obstacles), args.obstacle_clearance)
    planner.set_left_arm(np.radians(own.left_start_deg))
    plan = planner.plan(grasp, np.radians(own.arm_start_deg), np.zeros(22), anchor=args.anchor,
                        anchor_offset=args.anchor_offset, wrist_frame=args.wrist_frame,
                        stop_after=args.stop_after, release=args.release, yaw_list=args.yaw_deg)
    print(json.dumps(plan.report, indent=2))
    Path(args.plan_out).write_text(json.dumps(plan.report, indent=2))
    if own.samples:
        t, arm, hand, seg = [], [], [], []
        offset = 0.0
        for k, s in enumerate(plan.segments):
            for tt in np.arange(0, s.duration + 1e-9, 1 / args.rate):
                a, h = s.sample(tt)
                t.append(offset + tt), arm.append(a), hand.append(h), seg.append(k)
            offset += s.duration
        arm, hand = np.array(arm), np.array(hand)
        dt = 1 / args.rate
        print('peak arm speed deg/s', round(math.degrees(np.abs(np.diff(arm, axis=0)).max() / dt), 2),
              '| peak hand speed deg/s', round(math.degrees(np.abs(np.diff(hand, axis=0)).max() / dt), 2),
              '| samples', len(t))
        np.savez(own.samples, t=np.array(t), arm=arm, hand=hand, segment=np.array(seg),
                 names=np.array([s.name for s in plan.segments]))


if __name__ == '__main__':
    main()
