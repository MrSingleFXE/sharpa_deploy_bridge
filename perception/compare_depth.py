#!/usr/bin/env python3
"""Compare RealSense depth with FoundationStereo depth on one capture (numpy + OpenCV, no GPU).

Metrics (both on the color pixel grid, inside --roi):
  valid_fraction        share of pixels with depth
  table_rms_mm          noise of the dominant plane (table): RMS distance of plane inliers (< 2 cm) to the plane
  object_height_mm      (with --box and --extrinsics) highest object point above the table inside the pixel box;
                        compare with a ruler measurement of that object.
Writes <capture>/compare.png (color | RealSense | FoundationStereo, same colour scale) and compare.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.geometry import deproject, plane_normal, transform  # noqa: E402


def fit_plane(P, rng, iters=300, thr=0.01):
    best, best_n = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - s[0]) @ n) < thr).sum())
        if k > best_n:
            best, best_n = (n, s[0]), k
    n, p0 = best
    inl = P[np.abs((P - p0) @ n) < thr]
    c = inl.mean(0)
    n = plane_normal(inl - c)
    return n, c, inl


def stats(depth, K, roi, X=None, box=None, rng=None):
    h, w = depth.shape
    m = np.zeros_like(depth, bool)
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    m[y0:y1, x0:x1] = True
    out = {'valid_fraction': round(float((depth[m] > 0).mean()), 4)}
    P, _ = deproject(depth, K, m)
    P = P[P[:, 2] < 2.5]
    if len(P) < 1000:
        return out
    sub = P[rng.choice(len(P), min(len(P), 60000), replace=False)]
    n, c, inl = fit_plane(sub, rng)
    d = (inl - c) @ n
    out['table_rms_mm'] = round(1000 * float(np.sqrt(np.mean(d ** 2))), 2)
    out['table_inlier_fraction'] = round(len(inl) / len(sub), 3)
    if box is not None and X is not None:
        bm = np.zeros_like(depth, bool)
        bx0, by0, bx1, by1 = box
        bm[by0:by1, bx0:bx1] = True
        Pb, _ = deproject(depth, K, bm)
        W = transform(Pb, X)
        Wt = transform(inl, X)
        table_z = float(np.median(Wt[:, 2]))
        above = W[:, 2] - table_z
        top = np.percentile(above, 99) if len(above) else float('nan')
        out['table_z_world_m'] = round(table_z, 4)
        out['object_height_mm'] = round(1000 * float(top), 1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('capture')
    p.add_argument('--extrinsics', help='calib/camera_extrinsics.json (needed for object height in world)')
    p.add_argument('--roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='pixel box with table + objects')
    p.add_argument('--box', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='pixel box around one object')
    a = p.parse_args(argv)
    import cv2
    from perception.fs_depth import colorize
    cap = Path(a.capture)
    meta = json.loads((cap / 'meta.json').read_text())
    K = np.array(meta['K_color'])
    X = np.array(json.loads(Path(a.extrinsics).read_text())['T_world_cam']) if a.extrinsics else None
    color = cv2.imread(str(cap / 'color.png'))
    report, imgs = {}, [color]
    depths = {'realsense': cap / 'depth_raw_color.npy', 'foundation_stereo': cap / 'depth_fs_color.npy'}
    loaded = {k: np.load(v) for k, v in depths.items() if v.exists()}
    allv = np.concatenate([d[d > 0] for d in loaded.values()])
    lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
    for k, d in loaded.items():
        report[k] = stats(d, K, a.roi, X, a.box, np.random.default_rng(0))
        im = colorize(d, lo, hi)
        cv2.putText(im, k, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        imgs.append(im)
    for im in imgs:
        if a.roi:
            cv2.rectangle(im, tuple(a.roi[:2]), tuple(a.roi[2:]), (255, 255, 255), 2)
        if a.box:
            cv2.rectangle(im, tuple(a.box[:2]), tuple(a.box[2:]), (0, 0, 255), 2)
    cv2.imwrite(str(cap / 'compare.png'), np.concatenate(imgs, 1))
    (cap / 'compare.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'image: {cap / "compare.png"}  (lower table_rms_mm = flatter table; object height vs your ruler)')


if __name__ == '__main__':
    main()
