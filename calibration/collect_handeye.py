#!/usr/bin/env python3
"""Collect eye-to-hand samples: right-flange FK (A_i) + ChArUco board pose in the camera (B_i).

Modes
  manual (default): this script never moves the robot. Move the right arm yourself (keyboard_control.py,
                    or pendant hand-guiding), stop, then press Enter here to capture. 's' skip, 'q' finish.
  auto            : plans small slow moves around the CURRENT pose (+-6 cm, 15-30 deg about varied axes),
                    each one confirmed with Enter, captures after settling, returns to the start pose.
                    Needs --execute; without it only the planned poses are checked and printed.

Camera: --camera realsense (pyrealsense2) | ros (realsense2_camera topics) | sim (testing, needs --sim-truth).
Output: <out>/samples.jsonl, <out>/img_XXX.png, <out>/intrinsics.json, <out>/board.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.planner import Limits, PlanError, Planner  # noqa: E402
from bridge.urdf_kin import URDFModel, axis_angle  # noqa: E402
from calibration.board import BoardSpec, detect_pose  # noqa: E402
from calibration.robot_io import ARM, FLANGE, Abort, RobotIO, ask  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='output folder for this collection session')
    p.add_argument('--board', required=True, help='board.json written by make_board.py (with MEASURED sizes)')
    p.add_argument('--camera', choices=['realsense', 'ros', 'sim'], default='realsense')
    p.add_argument('--color-topic', default='/camera/camera/color/image_raw')
    p.add_argument('--info-topic', default='/camera/camera/color/camera_info')
    p.add_argument('--model', default='D435I', help='RealSense model to use when --serial is not given')
    p.add_argument('--serial', help='RealSense serial number (if several cameras)')
    p.add_argument('--mode', choices=['manual', 'auto'], default='manual')
    p.add_argument('--n', type=int, default=15, help='auto: number of poses to try')
    p.add_argument('--pos-range', type=float, default=0.06, help='auto: max position offset per axis (m)')
    p.add_argument('--rot-min-deg', type=float, default=15)
    p.add_argument('--rot-max-deg', type=float, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--frames', type=int, default=5, help='frames averaged per capture')
    p.add_argument('--table-z', type=float, default=0.0)
    p.add_argument('--clearance', type=float, default=0.10)
    p.add_argument('--arm-speed-deg', type=float, default=5.0)
    p.add_argument('--arm-accel-deg', type=float, default=10.0)
    p.add_argument('--execute', action='store_true', help='auto: actually move the arm')
    p.add_argument('--yes', action='store_true', help='auto: do not pause before each move (testing)')
    p.add_argument('--urdf', help='URDF file instead of /bimanual/robot_description')
    p.add_argument('--sim-truth', help='sim camera: json with T_world_cam and T_flange_board (4x4 lists)')
    return p.parse_args(argv)


def auto_targets(T0, n, pos_range, rot_min, rot_max, seed):
    """Targets for right_hand_C_MC around T0: offsets in a box, rotations about cycled frame axes + random tilt."""
    rng = np.random.default_rng(seed)
    base_axes = [np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1])]
    out = []
    for k in range(n):
        ax = base_axes[k % len(base_axes)] + rng.normal(0, 0.2, 3)
        ax /= np.linalg.norm(ax)
        ang = math.radians(rng.uniform(rot_min, rot_max)) * (1 if k % 2 == 0 else -1)
        T = T0.copy()
        T[:3, :3] = T0[:3, :3] @ axis_angle(ax, ang)       # rotate about the hand's own axes
        T[:3, 3] = T0[:3, 3] + rng.uniform(-pos_range, pos_range, 3)
        out.append(T)
    return out


def average_pose(poses):
    from calibration.handeye import mean_transform
    return mean_transform([p.T_cam_board for p in poses])


def capture(cam, spec, frames, out, idx):
    import cv2
    poses, last = [], None
    for _ in range(frames):
        color, _ = cam.grab()
        last = color
        bp = detect_pose(color, cam.intr.K, cam.intr.dist, spec)
        if bp is not None:
            poses.append(bp)
    cv2.imwrite(str(out / f'img_{idx:03d}.png'), last)
    if len(poses) < max(1, frames // 2 + 1):
        return None
    T = average_pose(poses)
    spread_mm = 1000 * max(np.linalg.norm(p.T_cam_board[:3, 3] - T[:3, 3]) for p in poses)
    return {'T_cam_board': T.tolist(), 'corners': int(np.median([p.corners for p in poses])),
            'reproj_px': float(np.median([p.reproj_px for p in poses])), 'frame_spread_mm': spread_mm,
            'frames_detected': len(poses)}


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = BoardSpec.load(args.board)
    spec.save(out / 'board.json')
    io = RobotIO(urdf_path=args.urdf)
    try:
        io.wait_ready(need_hand=True)
        urdf = io.st['urdf']
        model = URDFModel(urdf)
        if FLANGE not in model.links:
            raise Abort(f'{FLANGE} not in robot description')

        def flange_pose():
            io.pump()
            return model.fk(dict(zip(ARM, io.st['arm'])), FLANGE)

        if args.camera == 'realsense':
            from calibration.camera import RealSenseCamera
            cam = RealSenseCamera(serial=args.serial, depth=False, model=args.model)
        elif args.camera == 'ros':
            from calibration.camera import RosCamera
            cam = RosCamera(io.node, io.pump, args.color_topic, args.info_topic, depth_topic='')
        else:
            from calibration.camera import SimCamera
            truth = json.loads(Path(args.sim_truth).read_text())
            cam = SimCamera(spec, np.array(truth['T_world_cam']), np.array(truth['T_flange_board']), flange_pose)
        cam.intr.save(out / 'intrinsics.json')
        print(f'camera: {cam.intr.source}, size {cam.intr.size}, fx={cam.intr.K[0, 0]:.1f}')

        samples_path = out / 'samples.jsonl'
        idx = sum(1 for _ in open(samples_path)) if samples_path.exists() else 0

        def record(tag):
            nonlocal idx
            io.pump(0.1)
            q = io.st['arm'].copy()
            A = model.fk(dict(zip(ARM, q)), FLANGE)
            det = capture(cam, spec, args.frames, out, idx)
            if det is None:
                print(f'  [{tag}] board NOT detected (img_{idx:03d}.png) -- sample not used')
                idx += 1
                return False
            if np.abs(io.st['arm'] - q).max() > math.radians(0.05):
                print(f'  [{tag}] arm moved during capture -- sample not used')
                return False
            rec = dict(det, index=idx, tag=tag, arm_rad=q.tolist(), T_world_flange=A.tolist(),
                       left_arm_rad=None if io.st['left'] is None else io.st['left'].tolist(), time=time.time())
            with open(samples_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f'  [{tag}] sample {idx}: {det["corners"]} corners, reproj {det["reproj_px"]:.2f}px, '
                  f'frame spread {det["frame_spread_mm"]:.2f}mm, board {np.round(det["T_cam_board"][2][3], 3)} m away')
            idx += 1
            return True

        if args.mode == 'manual':
            print('MANUAL mode: this script sends no motion commands. Move the arm, let it stop, then:\n'
                  '  Enter = capture, s = skip, q = finish. Aim for 15-25 poses with varied tilt (see README).')
            while True:
                c = ask(f'[{idx} samples] Enter=capture  q=finish > ')
                if c == 'q':
                    break
                if c == 's':
                    continue
                if not io.stationary():
                    print('  arm still moving; wait and retry')
                    continue
                record('manual')
        else:
            lim = Limits(arm_vel=math.radians(args.arm_speed_deg), arm_acc=math.radians(args.arm_accel_deg),
                         table_z=args.table_z, clearance=args.clearance)
            planner = Planner(urdf, io.st['hand_names'], lim)
            if io.st['left'] is not None:
                planner.set_left_arm(io.st['left'])
            start_arm, hand = io.st['arm'].copy(), io.st['hand'].copy()
            T0 = planner.wrist_pose(start_arm)
            start_c = planner.clearance(start_arm, hand)[0]
            floor = min(lim.clearance, start_c - 0.005)
            targets = auto_targets(T0, args.n, args.pos_range, args.rot_min_deg, args.rot_max_deg, args.seed)
            # Dry check all moves as a chain from the start pose.
            plan, q = [], start_arm
            for k, T in enumerate(targets):
                try:
                    seg = planner.cartesian_segment(f'pose_{k}', q, T, hand, hand)
                    planner.check(seg, floor=floor)
                    plan.append(seg)
                    q = seg.arm[-1]
                except PlanError as e:
                    print(f'  pose {k}: skipped ({e})')
            back = planner.cartesian_segment('return', q, T0, hand, hand)
            planner.check(back, floor=floor)
            print(f'auto plan: {len(plan)}/{len(targets)} poses feasible, '
                  f'total {sum(s.duration for s in plan) + back.duration:.0f} s at {args.arm_speed_deg} deg/s; '
                  f'start clearance {start_c:.3f} m')
            record('start')
            if not args.execute:
                print('DRY RUN: no motion. Add --execute to move (each move asks for Enter).')
                return 0
            io.enable_motion()
            for seg in plan + [back]:
                io.pump(0.2)
                c = ask(f'[{seg.name}] {seg.duration:.1f} s -- Enter=move, s=skip rest and return, q=stop > ',
                        args.yes)
                if c == 'q':
                    print('stopped by user (arm holds its last target)')
                    break
                if c == 's' and seg is not back:
                    seg = planner.cartesian_segment('return', io.st['arm'].copy(), T0, hand, hand)
                    planner.check(seg, floor=floor)
                    io.run_segment(seg)
                    break
                io.run_segment(seg)
                io.pump(1.0)                              # settle
                if seg is not back:
                    record(seg.name)
        print(f'done: {idx} images, samples in {samples_path}')
        return 0
    except (Abort, PlanError) as e:
        print(f'STOPPED: {e}. No more targets are sent; the arm holds its last accepted target.')
        return 2
    finally:
        io.close()


if __name__ == '__main__':
    sys.exit(main())
