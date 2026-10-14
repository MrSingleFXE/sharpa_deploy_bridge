#!/usr/bin/env python3
"""Read-only check of the camera extrinsics (never moves the robot).

1. Overlay: projects right arm / hand link origins (robot FK + extrinsics) onto the live color image.
   Arm dots should sit on the arm. Hand dots that are consistently off by a few cm point at the
   uncalibrated hand mount (hand_mounts.yaml translation = 0), not at the camera calibration.
2. Board check (if the board is still mounted): observed board pose vs prediction A*Y -> mm / deg.
3. Table height: RANSAC plane on the depth cloud (world frame), near-horizontal planes only ->
   suggested --table-z for execute_grasp.py.
Writes <out>/overlay.png and <out>/verify.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.urdf_kin import URDFModel  # noqa: E402
from calibration.board import BoardSpec, detect_pose  # noqa: E402
from calibration.handeye import inv, log_err  # noqa: E402
from calibration.robot_io import ARM, FLANGE, Abort, RobotIO  # noqa: E402


def table_plane(depth, K, X, rng, max_range=2.0, iters=400, thr=0.008, roi=None):
    """Largest near-horizontal plane in world coordinates; returns (z, inlier_ratio, n_points)."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:4, 0:w:4]
    d = depth[vs, us]
    ok = (d > 0.15) & (d < max_range)
    if roi:
        x0, y0, x1, y1 = roi
        ok &= (us >= x0) & (us < x1) & (vs >= y0) & (vs < y1)
    u, v, z = us[ok].astype(float), vs[ok].astype(float), d[ok].astype(float)
    if len(z) < 500:
        return None
    pc = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
    pw = (X[:3, :3] @ pc.T).T + X[:3, 3]
    best = (0, None)
    for _ in range(iters):
        s = pw[rng.choice(len(pw), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        if abs(nrm[2]) < math.cos(math.radians(10)):       # only near-horizontal planes
            continue
        inl = np.abs((pw - s[0]) @ nrm) < thr
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best[1] is None:
        return None
    P = pw[best[1]]
    c = P.mean(0)
    nrm = np.linalg.svd(P - c)[2][2]
    tilt = math.degrees(math.acos(min(1.0, abs(nrm[2]))))
    return {'table_z_m': round(float(np.median(P[:, 2])), 4), 'tilt_deg': round(tilt, 2),
            'inlier_ratio': round(float(best[0] / len(pw)), 3), 'points': int(best[0])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--extrinsics', default='calib/camera_extrinsics.json')
    p.add_argument('--board', help='board.json; enables the board check')
    p.add_argument('--camera', choices=['realsense', 'ros'], default='realsense')
    p.add_argument('--color-topic', default='/camera/camera/color/image_raw')
    p.add_argument('--info-topic', default='/camera/camera/color/camera_info')
    p.add_argument('--depth-topic', default='/camera/camera/aligned_depth_to_color/image_raw')
    p.add_argument('--serial')
    p.add_argument('--table-roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                   help='pixel box containing mostly table (default: whole image)')
    p.add_argument('--out', default='calib/verify')
    p.add_argument('--urdf')
    args = p.parse_args(argv)
    import cv2

    ext = json.loads(Path(args.extrinsics).read_text())
    X, Y = np.array(ext['T_world_cam']), np.array(ext['T_flange_board'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    io = RobotIO('bodex_calib_verify', urdf_path=args.urdf)
    try:
        io.wait_ready(need_hand=False)
        model = URDFModel(io.st['urdf'])
        if args.camera == 'realsense':
            from calibration.camera import RealSenseCamera
            cam = RealSenseCamera(serial=args.serial, depth=True)
        else:
            from calibration.camera import RosCamera
            cam = RosCamera(io.node, io.pump, args.color_topic, args.info_topic, args.depth_topic)
        K, dist = cam.intr.K, cam.intr.dist
        if np.abs(K - np.array(ext['intrinsics']['K'])).max() > 1.0:
            print('WARNING: camera intrinsics differ from calibration (different resolution?)')
        color, depth = cam.grab()
        io.pump(0.1)
        q = dict(zip(ARM, io.st['arm']))
        if io.st['hand'] is not None:
            q.update(zip(io.st['hand_names'], io.st['hand']))
        poses = model.link_poses(q)
        report = {'camera_position_world_m': np.round(X[:3, 3], 4).tolist()}

        # 1. overlay
        Xi = inv(X)
        img = color.copy()
        groups = {'arm': [f'right_{n}' for n in ('J1_link', 'J2_link', 'J3_link', 'J4_link', 'J5_link', 'J6_link')]
                  + [FLANGE], 'hand': [l for l in poses if l.startswith('right_') and
                                       ('fingertip' in l or l in ('right_hand_C_MC', 'right_hand_flange'))]}
        colors = {'arm': (0, 255, 255), 'hand': (255, 0, 255)}
        drawn = {}
        for g, links in groups.items():
            pts = [poses[l][:3, 3] for l in links if l in poses]
            if not pts:
                continue
            pc = (Xi[:3, :3] @ np.array(pts).T).T + Xi[:3, 3]
            front = pc[:, 2] > 0.05
            uv, _ = cv2.projectPoints(pc[front], np.zeros(3), np.zeros(3), K, dist)
            for (u, v) in uv.reshape(-1, 2):
                cv2.circle(img, (int(u), int(v)), 6, colors[g], 2)
            drawn[g] = int(front.sum())
        cv2.putText(img, 'yellow: arm FK  magenta: hand FK (mount uncalibrated)', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / 'overlay.png'), img)
        report['overlay_points'] = drawn

        # 2. board
        if args.board:
            bp = detect_pose(color, K, dist, BoardSpec.load(args.board))
            if bp is None:
                report['board_check'] = 'board not detected'
            else:
                A = model.fk(dict(zip(ARM, io.st['arm'])), FLANGE)
                r, t = log_err(inv(A @ Y) @ X @ bp.T_cam_board)
                report['board_check'] = {'mm': round(1000 * float(np.linalg.norm(t)), 2),
                                         'deg': round(math.degrees(float(np.linalg.norm(r))), 3),
                                         'corners': bp.corners}

        # 3. table
        if depth is not None:
            report['table'] = table_plane(depth, K, X, np.random.default_rng(0), roi=args.table_roi)
        else:
            report['table'] = 'no depth stream'
        (out / 'verify.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f'overlay image: {out / "overlay.png"}')
        cam.close()
        return 0
    except Abort as e:
        print(f'STOPPED: {e}')
        return 2
    finally:
        io.close()


if __name__ == '__main__':
    sys.exit(main())
