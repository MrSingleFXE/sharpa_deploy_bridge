"""Depth geometry shared by the FoundationStereo wrapper and the point-cloud tools (numpy only)."""

import numpy as np


def disparity_to_depth(disp, fx, baseline_m, remove_invisible=True, min_disp=0.5):
    """Left-image disparity (px) -> depth (m) in the left (IR-left) frame; invalid = 0."""
    disp = np.asarray(disp, np.float32)
    depth = np.zeros_like(disp)
    ok = np.isfinite(disp) & (disp > min_disp)
    if remove_invisible:                       # the matching right-image column must exist
        xx = np.arange(disp.shape[1], dtype=np.float32)[None, :]
        ok &= (xx - disp) >= 0
    depth[ok] = fx * baseline_m / disp[ok]
    return depth


def deproject(depth, K, mask=None):
    """Depth image -> (N,3) points in that camera frame, plus the (v,u) pixel indices used."""
    v, u = np.nonzero((depth > 0) if mask is None else (mask & (depth > 0)))
    z = depth[v, u].astype(np.float64)
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    return np.stack([x, y, z], 1), (v, u)


def register_depth(depth_src, K_src, T_dst_from_src, K_dst, size_dst):
    """Forward-warp a depth map into another camera (z-buffer, nearest pixel). size_dst = (w, h)."""
    pts, _ = deproject(depth_src, K_src)
    T = np.asarray(T_dst_from_src, float)
    p = pts @ T[:3, :3].T + T[:3, 3]
    p = p[p[:, 2] > 1e-3]
    K = np.asarray(K_dst, float)
    u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int64)
    w, h = size_dst
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[ok], v[ok], p[ok, 2]
    out = np.full(h * w, np.inf, np.float32)
    np.minimum.at(out, v * w + u, z.astype(np.float32))
    out[~np.isfinite(out)] = 0
    out = out.reshape(h, w)
    # Fill isolated 1-pixel holes from the forward warp with the min of valid 3x3 neighbours.
    hole = out == 0
    pad = np.pad(np.where(hole, np.inf, out), 1, constant_values=np.inf)
    neigh = np.min(np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]), 0)
    cnt = sum((pad[dy:dy + h, dx:dx + w] < np.inf).astype(int) for dy in range(3) for dx in range(3))
    fill = hole & (cnt >= 5) & np.isfinite(neigh)
    out[fill] = neigh[fill]
    return out


def transform(points, T):
    T = np.asarray(T, float)
    return points @ T[:3, :3].T + T[:3, 3]


def plane_normal(centered):
    """Normal of the least-squares plane through centred points (N,3): smallest-eigenvalue eigenvector of
    the 3x3 scatter matrix (an SVD of the N x 3 array would build an N x N matrix)."""
    w, V = np.linalg.eigh(centered.T @ centered)
    return V[:, 0]
