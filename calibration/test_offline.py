#!/usr/bin/env python3
"""Offline self-test (no ROS, no camera): ChArUco detection on rendered images, the industrial_calibration
target-finder config, and the depth table-plane fit used by verify_extrinsics.py.

    python3 calibration/test_offline.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, detect_pose, synth_view  # noqa: E402
from calibration.ic_config import inv  # noqa: E402


def to_T(rv, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rv, float))[0]
    T[:3, 3] = t
    return T


def test_detection():
    spec = BoardSpec()
    K = np.array([[910.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
    rng = np.random.default_rng(1)
    worst = (0, 0)
    for k in range(10):
        rv = rng.normal(0, 0.35, 3)
        rv *= min(1.0, math.radians(45) / np.linalg.norm(rv))
        T = to_T(rv, [rng.uniform(-0.15, 0.1), rng.uniform(-0.1, 0.05), rng.uniform(0.5, 1.0)])
        bp = detect_pose(synth_view(spec, T, K), K, np.zeros(5), spec)
        assert bp is not None, f'view {k} not detected'
        D = inv(T) @ bp.T_cam_board
        e = (1000 * np.linalg.norm(D[:3, 3]), math.degrees(np.linalg.norm(cv2.Rodrigues(D[:3, :3])[0])))
        worst = (max(worst[0], e[0]), max(worst[1], e[1]))
    print(f'ChArUco detection: 10/10 views, worst error {worst[0]:.2f} mm / {worst[1]:.3f} deg')
    assert worst[0] < 3 and worst[1] < 0.5


def test_ic_target_config():
    tf = BoardSpec(square_m=0.0299, marker_m=0.0219).ic_target_finder()
    assert tf == {'type': 'CharucoGridTargetFinder', 'rows': 7, 'cols': 5, 'chessboard_dim': 0.0299,
                  'aruco_marker_dim': 0.0219, 'dictionary': cv2.aruco.DICT_5X5_100}, tf
    print(f'industrial_calibration target finder config: {tf}')


def test_table_plane():
    from calibration.verify_extrinsics import table_plane
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    pos = np.array([-0.05, -0.45, 0.95])
    z = np.array([0.6, -0.3, -0.75])
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    X = np.eye(4)
    X[:3, :3] = np.c_[x, np.cross(z, x), z]
    X[:3, 3] = pos
    table_z = 0.021
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 600.0, (v - 240) / 600.0, np.ones_like(u, float)], -1) @ X[:3, :3].T
    depth = ((table_z - pos[2]) / rays[..., 2]).astype(np.float32)
    depth[(depth < 0) | (depth > 3)] = 0
    depth += np.random.default_rng(0).normal(0, 0.002, depth.shape).astype(np.float32)
    depth[100:200, 300:400] = 0.6
    r = table_plane(depth, K, X, np.random.default_rng(1))
    print(f'table plane: z {r["table_z_m"]} m (true {table_z}), tilt {r["tilt_deg"]} deg')
    assert abs(r['table_z_m'] - table_z) < 0.003


if __name__ == '__main__':
    print('OpenCV', cv2.__version__)
    test_detection()
    test_ic_target_config()
    test_table_plane()
    print('ALL OK')
