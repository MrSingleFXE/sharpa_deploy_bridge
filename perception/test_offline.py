#!/usr/bin/env python3
"""Offline self-test of the perception geometry (no camera, no GPU):
disparity -> depth, IR-left -> color registration, world transform, table + object extraction.

    python3 perception/test_offline.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.geometry import disparity_to_depth, register_depth  # noqa: E402
from perception.object_cloud import extract  # noqa: E402


def look_at(pos, target):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(z, x), z]
    T[:3, 3] = pos
    return T


def render(T_world_cam, K, size, table_z, box):
    """Ray-cast depth (camera z) of a table plane plus an axis-aligned box (xmin,xmax,ymin,ymax,height)."""
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)   # z = 1 rays
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
    dw = d @ R.T
    best = np.full((h, w), np.inf)
    t = (table_z - o[2]) / dw[..., 2]
    best = np.where(t > 0, t, best)
    x0, x1, y0, y1, bh = box
    lo, hi = np.array([x0, y0, table_z]), np.array([x1, y1, table_z + bh])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - o) / dw, (hi - o) / dw
    tmin = np.nanmax(np.minimum(t1, t2), -1)
    tmax = np.nanmin(np.maximum(t1, t2), -1)
    hit = (tmax >= tmin) & (tmin > 0)
    best = np.where(hit & (tmin < best), tmin, best)
    best[~np.isfinite(best)] = 0
    return best.astype(np.float32)          # ray parameter with z=1 rays == camera depth


def test_registration():
    K_ir = np.array([[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]])
    K_c = np.array([[910.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
    X_ir = look_at(np.array([-0.05, -0.45, 0.95]), [0.62, -0.7, 0.05])
    T_c_ir = np.eye(4)
    T_c_ir[:3, 3] = [0.015, 0.0, 0.0]                                     # color 15 mm right of IR-left
    X_c = X_ir @ np.linalg.inv(T_c_ir)
    box = (0.55, 0.65, -0.75, -0.65, 0.12)
    d_ir = render(X_ir, K_ir, (1280, 720), 0.02, box)
    baseline = 0.050
    disp = np.where(d_ir > 0, K_ir[0, 0] * baseline / np.maximum(d_ir, 1e-6), 0).astype(np.float32)
    d_ir2 = disparity_to_depth(disp, K_ir[0, 0], baseline)
    d_c = register_depth(d_ir2, K_ir, T_c_ir, K_c, (1280, 720))
    truth = render(X_c, K_c, (1280, 720), 0.02, box)
    both = (d_c > 0) & (truth > 0)
    err = np.abs(d_c[both] - truth[both])
    edge_free = np.percentile(err, 95)
    print(f'registration: {both.mean() * 100:.1f}% pixels compared, median err {1000 * np.median(err):.2f} mm, '
          f'p95 {1000 * edge_free:.2f} mm')
    assert np.median(err) < 0.002 and edge_free < 0.01
    # uncalibrated: table frame from the depth alone
    r = extract(d_c, K_c, np.random.default_rng(0))
    obj = r['object_table']
    print(f'object (no calibration): {len(obj)} pts, height {obj[:, 2].max():.3f} m (true 0.12), '
          f'table fit rms {r["table_rms_mm"]:.2f} mm, camera {r["camera_height_m"]:.3f} m above table')
    assert abs(obj[:, 2].max() - 0.12) < 0.006 and np.abs(obj[:, :2].mean(0)).max() < 1e-6
    # calibrated: world frame
    r = extract(d_c, K_c, np.random.default_rng(0), X=X_c)
    Tw = X_c @ r['T_cam_table']
    W = r['object_table'] @ Tw[:3, :3].T + Tw[:3, 3]
    tilt = math.degrees(math.acos(min(1.0, Tw[2, 2])))
    print(f'object (calibrated): table_z {Tw[2, 3]:.4f} (true 0.02), tilt {tilt:.2f} deg, '
          f'top {W[:, 2].max():.3f} (true 0.14), centroid xy {np.round(W.mean(0)[:2], 3)} (box centre 0.60,-0.70)')
    assert abs(Tw[2, 3] - 0.02) < 0.003 and tilt < 0.5 and abs(W[:, 2].max() - 0.14) < 0.006
    assert np.allclose(W.mean(0)[:2], [0.60, -0.70], atol=0.03)

if __name__ == '__main__':
    test_registration()
    print('ALL OK')
