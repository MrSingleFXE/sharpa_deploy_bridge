#!/usr/bin/env python3
"""Synthetic test of calibration/hand_mount_icp.py (no camera, no robot): render the Sharpa mesh with a known
"true" mount into depth images from several arm poses (+ noise, + arm clutter behind the flange), solve from the
placeholder mount, compare.        python3 calibration/test_hand_mount.py [--out runs/hand_mount_test]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.hand_model import HandModel, apply  # noqa: E402
from bridge.urdf_kin import axis_angle, rpy_matrix  # noqa: E402
from calibration.hand_mount_icp import diff, main as solve_main, to_T  # noqa: E402

PLACEHOLDER = {'right': {'xyz': [0.0, 0.0, 0.0], 'rpy': [3.14159265359, -1.57079665359, 0.0]}}


def look_at(pos, target):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return to_T(np.c_[x, np.cross(z, x), z], pos)


def render(points_cam, K, size):
    w, h = size
    z = points_cam[:, 2]
    u = np.round(K[0, 0] * points_cam[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * points_cam[:, 1] / z + K[1, 2]).astype(int)
    ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.full(h * w, np.inf, np.float32)
    np.minimum.at(depth, v[ok] * w + u[ok], z[ok].astype(np.float32))
    depth[~np.isfinite(depth)] = 0
    return depth.reshape(h, w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/hand_mount_test')
    p.add_argument('--views', type=int, default=6)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    dense = HandModel(points_per_m2=1_500_000, min_per_link=400, seed=5)          # "real" surface for rendering
    names = dense.joint_names
    joints = np.zeros(len(names))
    M_ph = to_T(rpy_matrix(*PLACEHOLDER['right']['rpy']), PLACEHOLDER['right']['xyz'])
    M_true = to_T(axis_angle(np.array([1.0, 0, 0]), math.radians(40)) @ M_ph[:3, :3], [0.012, 0.004, -0.003])
    (out / 'init_mount.json').write_text(json.dumps(PLACEHOLDER))
    K = np.array([[910.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
    X = look_at(np.array([-0.05, -0.45, 0.95]), [0.55, -0.7, 0.35])
    ext = {'T_world_cam': X.tolist(), 'intrinsics': {'K': K.tolist()}}
    (out / 'camera_extrinsics.json').write_text(json.dumps(ext))
    hand_pts = dense.points_in_root(joints, names)
    # clutter: the robot wrist behind the flange face (x < 0 in the flange frame)
    t = rng.uniform(0, 2 * math.pi, 20000)
    arm = np.c_[rng.uniform(-0.15, 0.0, 20000), 0.045 * np.cos(t), 0.045 * np.sin(t)]
    folders = []
    for i in range(a.views):
        R = axis_angle(rng.normal(size=3), math.radians(rng.uniform(15, 40))) @ look_at(
            np.zeros(3), X[:3, 3] - np.array([0.55, -0.7, 0.35]))[:3, :3] @ rpy_matrix(0, math.pi / 2, 0)
        A = to_T(R, [0.55 + rng.uniform(-0.08, 0.08), -0.7 + rng.uniform(-0.08, 0.08), 0.35 + rng.uniform(-0.05, 0.05)])
        W = np.concatenate([apply(hand_pts, A @ M_true), apply(arm, A)])
        C = apply(W, np.linalg.inv(X))
        depth = render(C, K, (1280, 720))
        depth[depth > 0] += rng.normal(0, 0.0015, (depth > 0).sum()).astype(np.float32)   # 1.5 mm depth noise
        f = out / f'view_{i:02d}'
        f.mkdir(exist_ok=True)
        np.save(f / 'depth_fs_color.npy', depth)
        (f / 'meta.json').write_text(json.dumps({'K_color': K.tolist(), 'size': [1280, 720]}))
        (f / 'robot.json').write_text(json.dumps({'T_world_flange': A.tolist(), 'hand_rad': joints.tolist(),
                                                  'hand_names': names, 'table_z_world_m': None}))
        import cv2
        cv2.imwrite(str(f / 'color.png'), np.full((720, 1280, 3), 90, np.uint8))
        folders.append(str(f))
    solve_main(folders + ['--extrinsics', str(out / 'camera_extrinsics.json'), '--init', str(out / 'init_mount.json'),
                          '--out', str(out / 'hand_mount.json')])
    res = json.loads((out / 'hand_mount.json').read_text())
    M = to_T(rpy_matrix(*res['right']['rpy']), res['right']['xyz'])
    mm, deg = diff(M_true, M)
    print(f'TRUE vs SOLVED: {mm:.2f} mm, {deg:.2f} deg  (placeholder was {diff(M_true, M_ph)[0]:.1f} mm, '
          f'{diff(M_true, M_ph)[1]:.1f} deg off)')
    assert mm < 3 and deg < 1.5
    print('ALL OK')


if __name__ == '__main__':
    main()
