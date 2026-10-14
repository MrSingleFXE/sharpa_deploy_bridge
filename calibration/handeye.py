"""Eye-to-hand calibration math (numpy + OpenCV).

Unknowns: X = T_world_cam (camera fixed in the deployment `world` frame),
          Y = T_flange_board (board rigidly fixed to the right flange / hand).
Per sample i: A_i Y = X B_i, with A_i = T_world_flange (robot FK) and B_i = T_cam_board (PnP).
"""

import math

import cv2
import numpy as np


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def to_T(rvec, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3] = t
    return T


def log_err(T):
    """(rotation angle-axis [rad], translation [m]) of a 4x4 error transform."""
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def mean_transform(Ts):
    R = sum(T[:3, :3] for T in Ts)
    U, _, Vt = np.linalg.svd(R)
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    out = np.eye(4)
    out[:3, :3] = Rm
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def residuals(A, B, X, Y):
    """Per-sample board-pose disagreement: (A_i Y)^-1 X B_i -> (rot deg, trans mm)."""
    out = []
    for a, b in zip(A, B):
        r, t = log_err(inv(a @ Y) @ X @ b)
        out.append((math.degrees(np.linalg.norm(r)), 1000 * np.linalg.norm(t)))
    return np.array(out)


METHODS = {name: getattr(cv2, flag) for name, flag in (
    ('tsai', 'CALIB_HAND_EYE_TSAI'), ('park', 'CALIB_HAND_EYE_PARK'),
    ('horaud', 'CALIB_HAND_EYE_HORAUD'), ('daniilidis', 'CALIB_HAND_EYE_DANIILIDIS'))
    if hasattr(cv2, flag)} if hasattr(cv2, 'calibrateHandEye') else {}


def _rot_log(R):
    return cv2.Rodrigues(R)[0].ravel()


def park_martin(A, B):
    """Closed-form eye-to-hand (numpy only). From A_i Y = X B_i: (A_j A_i^-1) X = X (B_j B_i^-1)."""
    M, C_rows, d_rows = np.zeros((3, 3)), [], []
    pairs = [(i, j) for i in range(len(A)) for j in range(i + 1, len(A))]
    Cs, Ds = [], []
    for i, j in pairs:
        C, D = A[j] @ inv(A[i]), B[j] @ inv(B[i])
        a, b = _rot_log(C[:3, :3]), _rot_log(D[:3, :3])
        if np.linalg.norm(a) < math.radians(2):          # nearly pure translation: no rotation information
            continue
        M += np.outer(b, a)
        Cs.append(C), Ds.append(D)
    if len(Cs) < 2:
        raise ValueError('not enough rotation between samples')
    w, V = np.linalg.eigh(M.T @ M)
    if w.min() <= 1e-12:
        raise ValueError('rotations about a single axis only; add rotations about other axes')
    R = V @ np.diag(w ** -0.5) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    lhs = np.vstack([C[:3, :3] - np.eye(3) for C in Cs])
    rhs = np.concatenate([R @ D[:3, 3] - C[:3, 3] for C, D in zip(Cs, Ds)])
    t = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t
    return X


def initial(A, B, method='park_martin'):
    """Initial X (T_world_cam) and Y (T_flange_board).

    'park_martin' is implemented here (works with any OpenCV build); other names use cv2.calibrateHandEye
    with the classic eye-to-hand trick (base->gripper inputs, output cam->base) when that function exists.
    """
    if method == 'park_martin':
        X = park_martin(A, B)
    else:
        Ainv = [inv(a) for a in A]
        R, t = cv2.calibrateHandEye([a[:3, :3] for a in Ainv], [a[:3, 3] for a in Ainv],
                                    [b[:3, :3] for b in B], [b[:3, 3] for b in B], method=METHODS[method])
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = R, t.ravel()
    Y = mean_transform([inv(a) @ X @ b for a, b in zip(A, B)])
    return X, Y


def refine(A, B, X, Y, iters=50, rot_weight_m=0.1):
    """Joint Levenberg-Marquardt over X and Y; residual = log((A Y)^-1 X B), rotation scaled by rot_weight_m."""
    def pack(X, Y):
        return np.r_[cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], cv2.Rodrigues(Y[:3, :3])[0].ravel(), Y[:3, 3]]

    def unpack(p):
        return to_T(p[0:3], p[3:6]), to_T(p[6:9], p[9:12])

    def res(p):
        Xp, Yp = unpack(p)
        out = []
        for a, b in zip(A, B):
            r, t = log_err(inv(a @ Yp) @ Xp @ b)
            out.append(np.r_[r * rot_weight_m, t])
        return np.concatenate(out)

    p, lam = pack(X, Y), 1e-3
    f = res(p)
    for _ in range(iters):
        J = np.empty((len(f), 12))
        for k in range(12):
            d = np.zeros(12)
            d[k] = 1e-6
            J[:, k] = (res(p + d) - f) / 1e-6
        H, g = J.T @ J, J.T @ f
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        fn = res(p + step)
        if fn @ fn < f @ f:
            p, f, lam = p + step, fn, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10
    return unpack(p)


def rotation_diversity_deg(A):
    """Largest angle between relative-rotation axes; hand-eye needs >= 2 clearly different axes."""
    axes = []
    for i in range(1, len(A)):
        r, _ = log_err(inv(A[0]) @ A[i])
        if np.linalg.norm(r) > math.radians(5):
            axes.append(r / np.linalg.norm(r))
    if len(axes) < 2:
        return 0.0
    best = 0.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            best = max(best, math.degrees(math.acos(min(1.0, abs(float(axes[i] @ axes[j]))))))
    return best


def solve(A, B, drop_outliers=True):
    """Return dict with X, Y, per-sample residuals, kept indices and a method comparison."""
    A, B = list(A), list(B)
    keep = list(range(len(A)))
    comparison = {}
    for name in ['park_martin'] + list(METHODS):
        try:
            Xm, Ym = initial(A, B, name)
            r = residuals(A, B, Xm, Ym)
            comparison[name] = {'rms_mm': float(np.sqrt(np.mean(r[:, 1] ** 2))),
                                'rms_deg': float(np.sqrt(np.mean(r[:, 0] ** 2)))}
        except (cv2.error, ValueError, np.linalg.LinAlgError) as e:
            comparison[name] = {'error': str(e)}
    ok = [k for k in comparison if 'rms_mm' in comparison[k]]
    if not ok:
        raise ValueError(f'no initial solution: {comparison}')
    best = min(ok, key=lambda k: comparison[k]['rms_mm'])
    X, Y = refine(A, B, *initial(A, B, best))
    if drop_outliers and len(A) >= 8:
        r = residuals(A, B, X, Y)
        thr = max(3 * np.median(r[:, 1]), 5.0)
        keep = [i for i in range(len(A)) if r[i, 1] <= thr]
        if len(keep) < len(A) and len(keep) >= 6:
            X, Y = refine([A[i] for i in keep], [B[i] for i in keep], X, Y)
    r = residuals(A, B, X, Y)
    return {'X': X, 'Y': Y, 'residuals': r, 'keep': keep, 'init_method': best, 'methods': comparison,
            'rotation_diversity_deg': rotation_diversity_deg([A[i] for i in keep]),
            'loo_workspace_mm': loo_workspace_error([A[i] for i in keep], [B[i] for i in keep], X, Y)}


def loo_workspace_error(A, B, X, Y):
    """Leave-one-out stability where it matters: re-solve without each sample and measure how far
    camera-observed points in the robot workspace (the sampled flange positions) move in world. mm (max, mean)."""
    if len(A) < 7:
        return None
    pts_world = np.array([a[:3, 3] for a in A])
    pts_cam = (inv(X)[:3, :3] @ pts_world.T).T + inv(X)[:3, 3]
    devs = []
    for k in range(len(A)):
        Xk, _ = refine([a for i, a in enumerate(A) if i != k], [b for i, b in enumerate(B) if i != k], X, Y, iters=15)
        moved = (Xk[:3, :3] @ pts_cam.T).T + Xk[:3, 3]
        devs.append(1000 * np.linalg.norm(moved - pts_world, axis=1).max())
    return {'max_mm': round(float(max(devs)), 3), 'mean_mm': round(float(np.mean(devs)), 3)}
