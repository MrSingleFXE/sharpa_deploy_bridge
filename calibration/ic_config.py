#!/usr/bin/env python3
"""Write the industrial_calibration extrinsic hand-eye configuration (static camera) for our setup.

    python3 calibration/ic_config.py --data calib/ic_data/<saved folder> --out calib/ic_config.yaml
        [--camera-info-topic /camera/camera/color/camera_info]   (default; needs ROS + the realsense node running)
        [--intrinsics fx fy cx cy]                               (instead of the topic)

Contents (the calibration app's format, plus two informational keys):
  intrinsics                      fx, fy, cx, cy of the colour image used by the data collector
  target_finder                   CharucoGridTargetFinder from calib/board/board.json (your measured sizes)
  camera_mount_to_camera_guess    initial guess of the camera in `world` (camera mount = world)
  target_mount_to_target_guess    initial guess of the board on the flange (target mount = right_ee_mount)
  homography_threshold, static_camera: true
The guesses are computed from the collected data (PnP per image + OpenCV's closed-form hand-eye); industrial_calibration
then does the actual optimisation. The board frame used here is the same as industrial_calibration's (verified with
OpenCV 4.6 vs 4.10).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, detect_pose  # noqa: E402


def T_from_pose(d):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([d['qx'], d['qy'], d['qz'], d['qw']]).as_matrix()
    T[:3, 3] = [d['x'], d['y'], d['z']]
    return T


def pose_from_T(T):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return {'x': float(T[0, 3]), 'y': float(T[1, 3]), 'z': float(T[2, 3]),
            'qx': float(q[0]), 'qy': float(q[1]), 'qz': float(q[2]), 'qw': float(q[3])}


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def intrinsics_from_topic(topic, timeout=10.0):
    import time
    import rclpy
    from sensor_msgs.msg import CameraInfo
    rclpy.init()
    node = rclpy.create_node('ic_config_camera_info')
    got = {}
    node.create_subscription(CameraInfo, topic, lambda m: got.setdefault('m', m), 10)
    end = time.monotonic() + timeout
    while 'm' not in got and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.try_shutdown()
    if 'm' not in got:
        sys.exit(f'no CameraInfo on {topic} (is the realsense node running, same ROS_DOMAIN_ID?)')
    m = got['m']
    k = list(m.k)
    d = list(m.d)
    if d and max(abs(x) for x in d) > 1e-3:
        print(f'WARNING: distortion coefficients {np.round(d, 4).tolist()} are not ~0; industrial_calibration assumes '
              f'an undistorted (rectified) image. Use the rectified colour topic.')
    return {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}, (m.width, m.height)


def _park_martin(A, B):
    """Closed-form eye-to-hand initial guess (used only if this OpenCV build lacks calibrateHandEye):
    (A_j A_i^-1) X = X (B_j B_i^-1)."""
    M, Cs, Ds = np.zeros((3, 3)), [], []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            C, D = A[j] @ inv(A[i]), B[j] @ inv(B[i])
            a, b = cv2.Rodrigues(C[:3, :3])[0].ravel(), cv2.Rodrigues(D[:3, :3])[0].ravel()
            if np.linalg.norm(a) > np.radians(2):
                M += np.outer(b, a)
                Cs.append(C), Ds.append(D)
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(w ** -0.5) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    t = np.linalg.lstsq(np.vstack([C[:3, :3] - np.eye(3) for C in Cs]),
                        np.concatenate([R @ D[:3, 3] - C[:3, 3] for C, D in zip(Cs, Ds)]), rcond=None)[0]
    return R, t


def guesses(data_dir, spec, intr):
    data_dir = Path(data_dir)
    cal = yaml.safe_load((data_dir / 'cal_data.yaml').read_text())['data']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
    A, B = [], []
    for item in cal:
        img = cv2.imread(str(data_dir / item['image']))
        bp = detect_pose(img, K, np.zeros(5), spec) if img is not None else None
        if bp is None:
            continue
        A.append(T_from_pose(yaml.safe_load((data_dir / item['pose']).read_text())))   # world -> flange
        B.append(bp.T_cam_board)
    if len(A) < 4:
        sys.exit(f'board found in only {len(A)} images of {data_dir}; need more observations')
    if hasattr(cv2, 'calibrateHandEye'):
        Ainv = [inv(a) for a in A]                               # eye-to-hand: base->gripper in, cam->base out
        R, t = cv2.calibrateHandEye([a[:3, :3] for a in Ainv], [a[:3, 3] for a in Ainv],
                                    [b[:3, :3] for b in B], [b[:3, 3] for b in B], method=cv2.CALIB_HAND_EYE_PARK)
    else:
        R, t = _park_martin(A, B)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    Ys = [inv(a) @ X @ b for a, b in zip(A, B)]
    U, _, Vt = np.linalg.svd(sum(y[:3, :3] for y in Ys))
    Y = np.eye(4)
    Y[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    Y[:3, 3] = np.mean([y[:3, 3] for y in Ys], axis=0)
    return X, Y, len(A), len(cal)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='folder written by the data collector (cal_data.yaml, images/, poses/)')
    p.add_argument('--board', default='calib/board/board.json')
    p.add_argument('--camera-info-topic', default='/camera/camera/color/camera_info')
    p.add_argument('--intrinsics', type=float, nargs=4, metavar=('FX', 'FY', 'CX', 'CY'))
    p.add_argument('--homography-threshold', type=float, default=2.0)
    p.add_argument('--out', default='calib/ic_config.yaml')
    a = p.parse_args(argv)
    spec = BoardSpec.load(a.board)
    if a.intrinsics:
        intr = dict(zip(('fx', 'fy', 'cx', 'cy'), a.intrinsics))
    else:
        intr, size = intrinsics_from_topic(a.camera_info_topic)
        print(f'intrinsics from {a.camera_info_topic}: {size[0]}x{size[1]}, fx {intr["fx"]:.1f}')
    X, Y, n, total = guesses(a.data, spec, intr)
    cfg = {'intrinsics': {k: float(v) for k, v in intr.items()},
           'camera_mount_to_camera_guess': pose_from_T(X),
           'target_mount_to_target_guess': pose_from_T(Y),
           'target_finder': spec.ic_target_finder(),
           'homography_threshold': a.homography_threshold,
           'static_camera': True,
           'info_camera_mount_frame': 'world',
           'info_target_mount_frame': 'right_ee_mount'}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'board found in {n}/{total} images; initial camera guess in world: {np.round(X[:3, 3], 3).tolist()} m, '
          f'board {np.linalg.norm(Y[:3, 3]):.3f} m from the flange')
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
