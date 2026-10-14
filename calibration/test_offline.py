#!/usr/bin/env python3
"""Offline self-test (no ROS, no camera): board detection on rendered images + hand-eye solver on noisy data.

    python3 calibration/test_offline.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, detect_pose, synth_view  # noqa: E402
from calibration.handeye import inv, log_err, solve, to_T  # noqa: E402


def err(Ta, Tb):
    r, t = log_err(inv(Ta) @ Tb)
    return 1000 * np.linalg.norm(t), math.degrees(np.linalg.norm(r))


def test_detection():
    spec = BoardSpec()
    K = np.array([[910.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
    rng = np.random.default_rng(1)
    worst = (0, 0)
    for k in range(10):
        rv = rng.normal(0, 0.35, 3)
        rv *= min(1.0, math.radians(45) / np.linalg.norm(rv))        # recommended tilt range (<= 45 deg)
        T = to_T(rv, [rng.uniform(-0.15, 0.1), rng.uniform(-0.1, 0.05),
                                                             rng.uniform(0.5, 1.0)])
        img = synth_view(spec, T, K)
        bp = detect_pose(img, K, np.zeros(5), spec)
        assert bp is not None, f'view {k} not detected'
        e = err(T, bp.T_cam_board)
        worst = (max(worst[0], e[0]), max(worst[1], e[1]))
    print(f'detection: 10/10 views, worst error {worst[0]:.2f} mm / {worst[1]:.3f} deg')
    assert worst[0] < 3 and worst[1] < 0.5


def test_solver():
    rng = np.random.default_rng(2)
    X = to_T([2.2, 0.1, -0.4], [0.3, -0.45, 0.9])          # camera in world
    Y = to_T([0.1, -0.2, 0.3], [0.02, 0.05, 0.12])          # board on flange
    A, B = [], []
    base = to_T([0, 2.5, 0], [0.55, -0.7, 0.35])
    for _ in range(15):
        Ai = base @ to_T(rng.normal(0, 0.35, 3), rng.uniform(-0.06, 0.06, 3))
        Bi = inv(X) @ Ai @ Y
        Bi = Bi @ to_T(rng.normal(0, math.radians(0.1), 3), rng.normal(0, 0.0007, 3))   # PnP noise
        A.append(Ai), B.append(Bi)
    B[3] = B[3] @ to_T([0, 0, 0], [0.03, 0, 0])             # one gross outlier
    res = solve(A, B)
    ex, ey = err(X, res['X']), err(Y, res['Y'])
    print(f'solver: X error {ex[0]:.2f} mm / {ex[1]:.3f} deg, Y error {ey[0]:.2f} mm / {ey[1]:.3f} deg, '
          f'kept {len(res["keep"])}/15 (outlier 3 dropped: {3 not in res["keep"]}), init {res["init_method"]}')
    assert ex[0] < 3 and ex[1] < 0.3 and 3 not in res['keep']


def test_table_plane():
    from calibration.verify_extrinsics import table_plane
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    pos = np.array([-0.05, -0.45, 0.95])
    z = np.array([0.6, -0.3, -0.75]); z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
    X = np.eye(4); X[:3, :3] = np.c_[x, np.cross(z, x), z]; X[:3, 3] = pos
    table_z = 0.021
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - 320) / 600.0, (v - 240) / 600.0, np.ones_like(u, float)], -1) @ X[:3, :3].T
    depth = ((table_z - pos[2]) / rays[..., 2]).astype(np.float32)         # ray parameter = camera z depth
    depth[(depth < 0) | (depth > 3)] = 0
    depth += np.random.default_rng(0).normal(0, 0.002, depth.shape).astype(np.float32)
    depth[100:200, 300:400] = 0.6                                           # an object above the table
    r = table_plane(depth, K, X, np.random.default_rng(1))
    print(f'table plane: z {r["table_z_m"]} m (true {table_z}), tilt {r["tilt_deg"]} deg')
    assert abs(r['table_z_m'] - table_z) < 0.003


if __name__ == '__main__':
    import cv2
    print('OpenCV', cv2.__version__)
    test_detection()
    test_solver()
    test_table_plane()
    print('ALL OK')
