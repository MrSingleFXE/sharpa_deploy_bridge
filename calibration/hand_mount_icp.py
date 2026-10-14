#!/usr/bin/env python3
"""Hand mount (arm flange right_ee_mount -> hand flange right_hand_flange) from depth views -- no ruler needed.

Needs the camera calibration (calib/camera_extrinsics.json) and views collected with
calibration/collect_hand_views.py (+ perception/fs_depth.py for FoundationStereo depth).

For every view i:  observed hand points (depth -> camera -> world with T_world_cam -> flange frame with the arm FK)
                   model hand points  (Sharpa collision meshes at the recorded hand joint angles, hand-flange frame)
One unknown transform M (flange -> hand flange) is shared by all views:  observed ~= M * model.
Solved by trimmed point-to-point ICP over all views together, started from several rotations about the flange axis
(the clocking of the hand is the least known part); the start with the best fit wins.
Quality: per-view RMS / inlier fraction and leave-one-view-out stability (mm / deg).

Output: calib/hand_mount.json ({"right": {"xyz": [...], "rpy": [...]}}, the hand_mounts.yaml format) + report,
        and <view>/hand_mount_overlay.png (fitted hand projected on the color image).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.hand_model import HandModel, apply  # noqa: E402
from bridge.urdf_kin import rpy_matrix  # noqa: E402

TOOL_AXIS = np.array([1.0, 0.0, 0.0])        # right_ee_mount x = flange normal (the URDF TCP offset is along +x)


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rpy_of(R):
    return [math.atan2(R[2, 1], R[2, 2]), math.asin(-max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[1, 0], R[0, 0])]


def axis_rot(axis, ang):
    from bridge.urdf_kin import axis_angle
    return axis_angle(axis / np.linalg.norm(axis), ang)


def nearest(A, B, chunk=2000):
    """index and distance of the nearest B point for every A point (KD-tree if scipy is available)."""
    try:
        from scipy.spatial import cKDTree
        d, idx = cKDTree(B).query(A)
        return idx, d
    except ImportError:
        pass
    idx, dist = np.empty(len(A), int), np.empty(len(A))
    for i in range(0, len(A), chunk):
        d = ((A[i:i + chunk, None] - B[None]) ** 2).sum(-1)
        j = d.argmin(1)
        idx[i:i + chunk], dist[i:i + chunk] = j, np.sqrt(d[np.arange(len(j)), j])
    return idx, dist


def kabsch(P, Q):
    """R, t minimising |R P + t - Q|."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def icp(views, M0, iters=40, thr_start=0.02, thr_end=0.004):
    """views: list of (observed (N,3) in flange frame, model (K,3) in hand-flange frame). Returns M, stats."""
    M = M0.copy()
    for it in range(iters):
        thr = thr_start * (thr_end / thr_start) ** (it / max(iters - 1, 1))
        P_all, Q_all = [], []
        for obs, mod in views:
            mw = apply(mod, M)
            j, d = nearest(obs, mw)
            keep = d < thr
            P_all.append(mod[j[keep]])
            Q_all.append(obs[keep])
        P, Q = np.concatenate(P_all), np.concatenate(Q_all)
        if len(P) < 200:
            break
        R, t = kabsch(apply(P, M), Q)
        M = to_T(R, t) @ M
    stats = []
    for obs, mod in views:
        _, d = nearest(obs, apply(mod, M))
        inl = d < 0.005
        stats.append({'inlier_fraction': float(inl.mean()), 'rms_mm': float(1000 * np.sqrt(np.mean(d[inl] ** 2)))
                      if inl.any() else float('nan'), 'points': int(len(obs))})
    return M, stats


def solve(views, M_init, starts=12, keep=2):
    """Coarse ICP (12 iterations) from `starts` rotations about the flange axis, full ICP for the best `keep`."""
    coarse = []
    for k in range(starts):
        R0 = axis_rot(TOOL_AXIS, 2 * math.pi * k / starts) @ M_init[:3, :3]
        M, st = icp(views, to_T(R0, M_init[:3, 3]), iters=12, thr_end=0.008)
        coarse.append((float(np.mean([s['inlier_fraction'] for s in st])), M, k))
    best = None
    for score0, M0, k in sorted(coarse, key=lambda c: -c[0])[:keep]:
        M, st = icp(views, M0)
        score = float(np.mean([s['inlier_fraction'] for s in st]))
        if best is None or score > best[0]:
            best = (score, M, st, k)
    return best


def diff(Ma, Mb):
    dR = Ma[:3, :3].T @ Mb[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    return 1000 * float(np.linalg.norm(Ma[:3, 3] - Mb[:3, 3])), ang


def load_views(folders, ext, hand, depth_kind, radius, rng, max_obs=4000):
    from perception.geometry import deproject
    X = np.array(ext['T_world_cam'])
    views, used = [], []
    for f in folders:
        f = Path(f)
        meta = json.loads((f / 'meta.json').read_text())
        rob = json.loads((f / 'robot.json').read_text())
        depth = np.load(f / ('depth_fs_color.npy' if depth_kind == 'fs' else 'depth_raw_color.npy'))
        P, _ = deproject(depth, np.array(meta['K_color']))
        W = apply(P, X)
        A = np.array(rob['T_world_flange'])
        F = apply(W, np.linalg.inv(A))                                  # flange frame
        sel = (np.linalg.norm(F, axis=1) < radius) & (F @ TOOL_AXIS > 0.005)   # in front of the flange face
        if 'table_z_world_m' in rob and rob['table_z_world_m'] is not None:
            sel &= W[:, 2] > rob['table_z_world_m'] + 0.01
        obs = F[sel]
        if len(obs) < 300:
            print(f'  {f.name}: only {len(obs)} points near the hand -- skipped (is the hand in view?)')
            continue
        obs = obs[rng.choice(len(obs), min(max_obs, len(obs)), replace=False)]
        mod = hand.points_in_root(rob['hand_rad'], rob['hand_names'])
        views.append((obs, mod[rng.choice(len(mod), min(6000, len(mod)), replace=False)]))
        used.append(f)
    return views, used


def overlay(folder, ext, M, hand):
    import cv2
    folder = Path(folder)
    meta = json.loads((folder / 'meta.json').read_text())
    rob = json.loads((folder / 'robot.json').read_text())
    X, A = np.array(ext['T_world_cam']), np.array(rob['T_world_flange'])
    pts = apply(hand.points_in_root(rob['hand_rad'], rob['hand_names']), np.linalg.inv(X) @ A @ M)
    pts = pts[pts[:, 2] > 0.05]
    K = np.array(meta['K_color'])
    uv = (K @ pts.T).T
    uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
    img = cv2.imread(str(folder / 'color.png'))
    h, w = img.shape[:2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    img[uv[ok, 1], uv[ok, 0]] = (255, 0, 255)
    cv2.imwrite(str(folder / 'hand_mount_overlay.png'), img)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('views', nargs='+', help='view folders from collect_hand_views.py')
    p.add_argument('--extrinsics', default='calib/camera_extrinsics.json')
    p.add_argument('--init', help='initial mount (hand_mounts.yaml / .json); default: ~/ws_fanuc/src/'
                                  'bimanual_manipulation/config/hand_mounts.yaml, else identity')
    p.add_argument('--depth', choices=['fs', 'raw'], default='fs')
    p.add_argument('--radius', type=float, default=0.30, help='keep observed points within this distance of the flange')
    p.add_argument('--hand-urdf')
    p.add_argument('--out', default='calib/hand_mount.json')
    a = p.parse_args(argv)
    from bridge.planner import load_hand_mount
    init = a.init or str(Path('~/ws_fanuc/src/bimanual_manipulation/config/hand_mounts.yaml').expanduser())
    xyz, rpy = load_hand_mount(init) if Path(init).exists() else ([0, 0, 0], [0, 0, 0])
    M_init = to_T(rpy_matrix(*rpy), xyz)
    ext = json.loads(Path(a.extrinsics).read_text())
    hand = HandModel(a.hand_urdf)
    rng = np.random.default_rng(0)
    views, used = load_views(a.views, ext, hand, a.depth, a.radius, rng)
    if len(views) < 2:
        sys.exit('need >= 2 usable views (>= 5 recommended)')
    score, M, stats, k = solve(views, M_init)
    loo = []
    if len(views) >= 3:
        for i in range(len(views)):
            Mi, _ = icp([v for j, v in enumerate(views) if j != i], M)
            loo.append(diff(M, Mi))
    rms = float(np.mean([s['rms_mm'] for s in stats]))
    inl = float(np.mean([s['inlier_fraction'] for s in stats]))
    loo_mm = max((l[0] for l in loo), default=float('nan'))
    loo_deg = max((l[1] for l in loo), default=float('nan'))
    verdict = ('GOOD' if inl > 0.7 and loo_mm < 2 and loo_deg < 1 and len(views) >= 5 else
               'USABLE' if inl > 0.5 and loo_mm < 5 and loo_deg < 3 else 'CHECK')
    res = {'right': {'xyz': np.round(M[:3, 3], 5).tolist(), 'rpy': np.round(rpy_of(M[:3, :3]), 5).tolist()},
           'verdict': verdict, 'views_used': [str(u) for u in used], 'mean_inlier_fraction': round(inl, 3),
           'mean_rms_mm': round(rms, 2), 'leave_one_view_out_max': {'mm': round(loo_mm, 2), 'deg': round(loo_deg, 2)},
           'start_rotation_about_flange_axis_deg': 360 * k / 12,
           'change_from_init': dict(zip(('mm', 'deg'), np.round(diff(M_init, M), 2).tolist())),
           'per_view': stats, 'init': {'file': init, 'xyz': xyz, 'rpy': rpy}, 'depth': a.depth}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    for f in used:
        overlay(f, ext, M, hand)
    print(json.dumps({k: res[k] for k in ('right', 'verdict', 'mean_inlier_fraction', 'mean_rms_mm',
                                          'leave_one_view_out_max', 'change_from_init')}, indent=2))
    print(f'saved {a.out}; check <view>/hand_mount_overlay.png (magenta = fitted hand)')


if __name__ == '__main__':
    main()
