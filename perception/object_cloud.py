#!/usr/bin/env python3
"""Object point cloud from one capture -- works WITHOUT camera calibration (no GPU, robot not involved).

  depth (FoundationStereo or RealSense, color grid) -> camera points
  -> dominant table plane (RANSAC in the camera frame)
  -> table frame: z = table normal (up), x = camera viewing direction projected on the table,
     origin = the object's centroid projected onto the table          (needs no calibration)
  -> points 5 mm .. --max-height above the table -> 1 cm voxel clusters -> object cluster:
     --pixel: the cluster at that pixel (no other filter);
     otherwise the largest cluster inside --roi / the image that is a plausible tabletop object:
       not touching the image border (the robot arms enter the picture from its edges),
       horizontal size <= --max-size; all candidates are printed (pick another one with --pixel)

With --extrinsics (calibration done) the world frame is added: T_world_table = T_world_cam @ T_cam_table,
and the table normal is checked against world z.

Outputs <capture>/object_<depth>/:
  object_table.npy   (N,3) object points in the table frame (table z = 0, object centred at x = y = 0)
                     -> input of grasping/dexlearn_infer.py
  object_world.npy   (N,3) same points in the robot world frame (only with --extrinsics)
  object.json        T_cam_table, T_world_table (if calibrated), table fit, object size/height, settings
  overlay.png        selected object pixels in red on the color image
  scene.ply          coloured scene cloud in the table frame (open with MeshLab / CloudCompare)
  object_views.png   top / side / front views of the object cloud
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.geometry import deproject, plane_normal, transform  # noqa: E402


def fit_plane(P, rng, iters=500, thr=0.006, normal_hint=None, max_tilt_deg=None):
    """Largest plane (n, p0, inlier mask). Optional constraint: normal within max_tilt of normal_hint."""
    best, best_n = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if normal_hint is not None and abs(n @ normal_hint) < np.cos(np.radians(max_tilt_deg)):
            continue
        k = int((np.abs((P - s[0]) @ n) < thr).sum())
        if k > best_n:
            best, best_n = (n, s[0]), k
    if best is None:
        raise RuntimeError('no table plane found')
    n, p0 = best
    inl = np.abs((P - p0) @ n) < thr
    c = P[inl].mean(0)
    n = plane_normal(P[inl] - c)                             # least-squares refit
    return n, c, inl


def clusters(P, voxel=0.01):
    """Connected components of occupied voxels (26-neighbourhood). Returns a label per point."""
    keys = np.floor(P / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    index = {tuple(k): i for i, k in enumerate(uniq)}
    lab = -np.ones(len(uniq), np.int64)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    cur = 0
    for i in range(len(uniq)):
        if lab[i] >= 0:
            continue
        lab[i] = cur
        q = deque([i])
        while q:
            k = uniq[q.popleft()]
            for o in offs:
                nb = index.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]))
                if nb is not None and lab[nb] < 0:
                    lab[nb] = cur
                    q.append(nb)
        cur += 1
    return lab[inv.ravel()]


def table_frame(n, c, view_dir=np.array([0.0, 0.0, 1.0])):
    """Camera-frame table plane -> T_cam_table (origin c, z = n towards the camera, x = view dir on the plane)."""
    if n @ (np.zeros(3) - c) < 0:                            # normal must point towards the camera (up)
        n = -n
    x = view_dir - (view_dir @ n) * n
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, np.cross(n, x), n]
    T[:3, 3] = c
    return T


def extract(depth, K, rng, X=None, max_height=0.30, roi=None, pixel=None, workspace=None, min_points=200,
            max_range=2.0, max_size=0.30, border_px=8):
    """Returns dict with object points (table frame), T_cam_table, pixel indices, table stats."""
    P, (v, u) = deproject(depth, K)
    keep = P[:, 2] < max_range
    P, v, u = P[keep], v[keep], u[keep]
    sub = rng.choice(len(P), min(len(P), 80000), replace=False)
    hint = None if X is None else X[:3, :3].T @ np.array([0, 0, 1.0])     # world up in the camera frame
    n, c, _ = fit_plane(P[sub], rng, normal_hint=hint, max_tilt_deg=None if X is None else 10)
    T0 = table_frame(n, c)
    Q = transform(P, np.linalg.inv(T0))                      # table frame with a provisional origin
    resid = np.abs(Q[sub, 2])
    table_rms = float(np.sqrt(np.mean(resid[resid < 0.006] ** 2)))
    sel = (Q[:, 2] > 0.005) & (Q[:, 2] < max_height)
    if roi is not None:
        x0, y0, x1, y1 = roi
        sel &= (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
    if workspace is not None and X is not None:
        W = transform(P, X)
        x0, x1, y0, y1 = workspace
        sel &= (W[:, 0] >= x0) & (W[:, 0] <= x1) & (W[:, 1] >= y0) & (W[:, 1] <= y1)
    Qc, vc, uc = Q[sel], v[sel], u[sel]
    if len(Qc) < min_points:
        raise RuntimeError(f'only {len(Qc)} points above the table (check --roi / --pixel / depth)')
    lab = clusters(Qc)
    if pixel is not None:
        d2 = (uc - pixel[0]) ** 2 + (vc - pixel[1]) ** 2
        if np.sqrt(d2.min()) > 15:
            raise RuntimeError(f'no above-table point within 15 px of pixel {pixel}')
        target = lab[int(np.argmin(d2))]
    else:
        h, w = depth.shape[:2]
        cands = []
        for i in np.unique(lab):
            k = lab == i
            if k.sum() < min_points:
                continue
            P_i, u_i, v_i = Qc[k], uc[k], vc[k]
            size = float(np.linalg.norm(P_i[:, :2].max(0) - P_i[:, :2].min(0)))
            border = bool((u_i < border_px).any() or (v_i < border_px).any() or (u_i >= w - border_px).any()
                          or (v_i >= h - border_px).any())
            why = 'touches image border (robot/off-table)' if border else (
                f'too large ({size:.2f} m > --max-size)' if size > max_size else 'ok')
            cands.append((int(k.sum()), i, why, size, int(np.median(u_i)), int(np.median(v_i))))
        cands.sort(reverse=True)
        print('above-table clusters (largest first):')
        for n_, i, why, size, mu, mv in cands:
            print(f'  {n_:7d} pts  size {size:.2f} m  pixel ({mu}, {mv})  {why}')
        ok = [c for c in cands if c[2] == 'ok']
        if not ok:
            raise RuntimeError('no plausible object cluster (all touch the image border or are too large); '
                               'move the arms out of the view or give --pixel U V on the object')
        target = ok[0][1]
        print(f'selected: pixel ({ok[0][4]}, {ok[0][5]}), {ok[0][0]} pts (use --pixel U V to pick another)')
    m = lab == target
    if m.sum() < min_points:
        raise RuntimeError(f'selected cluster has only {int(m.sum())} points')
    obj = Qc[m]
    shift = np.array([obj[:, 0].mean(), obj[:, 1].mean(), 0.0])          # origin under the object centroid
    T_cam_table = T0.copy()
    T_cam_table[:3, 3] = T0[:3, 3] + T0[:3, :3] @ shift
    return {'object_table': obj - shift, 'T_cam_table': T_cam_table, 'pixels': (vc[m], uc[m]),
            'scene_table': Q - shift, 'scene_pixels': (v, u), 'table_rms_mm': 1000 * table_rms,
            'camera_height_m': float(np.linalg.inv(T_cam_table)[2, 3])}


def write_ply(path, pts, rgb):
    pts = np.asarray(pts, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    rec = np.empty(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'], rec['y'], rec['z'] = pts.T
    rec['r'], rec['g'], rec['b'] = rgb.T
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        f.write(rec.tobytes())


def plot_views(path, pts, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a, (i, j, name) in zip(ax, ((0, 1, 'top (x-y)'), (0, 2, 'side (x-z)'), (1, 2, 'front (y-z)'))):
        a.scatter(pts[:, i], pts[:, j], s=1, c=pts[:, 2], cmap='viridis')
        a.set_aspect('equal')
        a.set_title(name)
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('capture')
    p.add_argument('--depth', choices=['fs', 'raw'], default='fs')
    p.add_argument('--extrinsics', help='calib/camera_extrinsics.json (optional; adds the world frame)')
    p.add_argument('--max-height', type=float, default=0.30)
    p.add_argument('--max-size', type=float, default=0.30, help='largest horizontal extent of an object cluster (m)')
    p.add_argument('--roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='pixel box with the object')
    p.add_argument('--pixel', type=int, nargs=2, metavar=('U', 'V'), help='a color-image pixel on the object')
    p.add_argument('--workspace', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                   help='world xy box (only with --extrinsics)')
    a = p.parse_args(argv)
    import cv2
    cap = Path(a.capture)
    meta = json.loads((cap / 'meta.json').read_text())
    K = np.array(meta['K_color'])
    X = None
    if a.extrinsics:
        ext = json.loads(Path(a.extrinsics).read_text())
        X = np.array(ext['T_world_cam'])
        if np.abs(K - np.array(ext['intrinsics']['K'])).max() > 1.0:
            print('WARNING: capture intrinsics differ from the calibration (different resolution?)')
    depth = np.load(cap / ('depth_fs_color.npy' if a.depth == 'fs' else 'depth_raw_color.npy'))
    r = extract(depth, K, np.random.default_rng(0), X, a.max_height, a.roi, a.pixel, a.workspace, max_size=a.max_size)
    obj = r['object_table']
    out = cap / f'object_{a.depth}'
    out.mkdir(exist_ok=True)
    np.save(out / 'object_table.npy', obj.astype(np.float32))
    info = {'depth_source': a.depth, 'capture': str(cap.resolve()), 'points': int(len(obj)),
            'size_table_frame_m': np.round(obj.max(0) - obj.min(0), 4).tolist(),
            'height_above_table_m': round(float(obj[:, 2].max()), 4),
            'table_fit_rms_mm': round(r['table_rms_mm'], 2), 'camera_height_above_table_m': round(r['camera_height_m'], 4),
            'T_cam_table': np.round(r['T_cam_table'], 6).tolist(),
            'frame_note': 'table frame: z up (table normal), x = camera view direction on the table, '
                          'origin = object centroid projected on the table'}
    if X is not None:
        Tw = X @ r['T_cam_table']
        info['extrinsics'] = str(Path(a.extrinsics).resolve())
        info['T_world_table'] = np.round(Tw, 6).tolist()
        info['table_z_world_m'] = round(float(Tw[2, 3]), 4)
        info['table_normal_tilt_vs_world_z_deg'] = round(float(np.degrees(np.arccos(min(1.0, Tw[2, 2])))), 2)
        info['object_centroid_world_m'] = np.round(transform(obj, Tw).mean(0), 4).tolist()
        np.save(out / 'object_world.npy', transform(obj, Tw).astype(np.float32))
    (out / 'object.json').write_text(json.dumps(info, indent=2))
    color = cv2.imread(str(cap / 'color.png'))
    img = color.copy()
    v, u = r['pixels']
    img[v, u] = (0.4 * img[v, u] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(out / 'overlay.png'), img)
    sv, su = r['scene_pixels']
    step = max(1, len(sv) // 300000)
    write_ply(out / 'scene.ply', r['scene_table'][::step], color[sv[::step], su[::step]][:, ::-1])
    plot_views(out / 'object_views.png', obj, f'{cap.name} object ({a.depth} depth), {len(obj)} pts, table frame [m]')
    print(json.dumps({k: v for k, v in info.items() if not k.startswith('T_')}, indent=2))
    print(f'saved {out}/ : object_table.npy, overlay.png, scene.ply, object_views.png'
          + (', object_world.npy' if X is not None else ' (no calibration: world frame skipped)'))


if __name__ == '__main__':
    main()
