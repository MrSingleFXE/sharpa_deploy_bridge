#!/usr/bin/env python3
"""End-to-end test of the industrial_calibration path with a known answer (no camera, no robot):
synthetic data in the data collector's format -> ic_config.py -> ic_extrinsic_cli (industrial_calibration)
-> ic_result.py -> compare with the truth.

    python3 calibration/test_ic_pipeline.py --cli <path to ic_extrinsic_cli> [--out runs/ic_test] [--n 25]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, synth_view  # noqa: E402
from calibration.ic_config import inv, pose_from_T  # noqa: E402


def to_T(rv, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rv, float))[0]
    T[:3, 3] = t
    return T


def err(Ta, Tb):
    D = inv(Ta) @ Tb
    return 1000 * np.linalg.norm(D[:3, 3]), math.degrees(np.linalg.norm(cv2.Rodrigues(D[:3, :3])[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--out', default='runs/ic_test')
    p.add_argument('--n', type=int, default=25)
    a = p.parse_args()
    out = Path(a.out)
    data = out / 'data'
    (data / 'images').mkdir(parents=True, exist_ok=True)
    (data / 'poses').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    spec = BoardSpec()
    K = np.array([[910.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
    X = to_T([2.25, 0.25, -0.35], [-0.05, -0.45, 0.95])                 # camera in world (truth)
    Y = to_T([0.2, -0.1, 0.1], [0.03, 0.02, 0.10])                      # board on the flange (truth)
    items, k = [], 0
    while len(items) < a.n:
        T_cam_board = to_T(rng.normal(0, 0.3, 3), [-0.06 + rng.uniform(-0.08, 0.08), -0.08 + rng.uniform(-0.06, 0.06),
                                                   0.75 + rng.uniform(-0.1, 0.1)])
        A = X @ T_cam_board @ inv(Y)                                     # world -> flange (what TF gives)
        img = synth_view(spec, T_cam_board, K).astype(np.float32)
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        name = f'{k:04d}'
        cv2.imwrite(str(data / 'images' / f'{name}.png'), cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0.4), cv2.COLOR_GRAY2BGR))
        (data / 'poses' / f'{name}.yaml').write_text(yaml.safe_dump(pose_from_T(A)))
        items.append({'image': f'images/{name}.png', 'pose': f'poses/{name}.yaml'})
        k += 1
    (data / 'cal_data.yaml').write_text(yaml.safe_dump({'data': items}))
    spec.save(out / 'board.json')
    py = sys.executable
    run = lambda *c: subprocess.run(c, check=True)
    run(py, 'calibration/ic_config.py', '--data', str(data), '--board', str(out / 'board.json'),
        '--intrinsics', '910', '910', '640', '360', '--out', str(out / 'ic_config.yaml'))
    r = subprocess.run([a.cli, str(out / 'ic_config.yaml'), str(data / 'cal_data.yaml'), str(out / 'ic_result.yaml')],
                       capture_output=True, text=True)
    (out / 'ic_cli.log').write_text(r.stdout + r.stderr)
    print('\n'.join(l for l in r.stdout.splitlines() if 'using' in l or 'converged' in l.lower()
                    or 'Final cost' in l or 'mm' in l)[:2000])
    if r.returncode not in (0, 1):
        sys.exit(f'ic_extrinsic_cli failed: {r.stderr[-1000:]}')
    run(py, 'calibration/ic_result.py', str(out / 'ic_result.yaml'), '--config', str(out / 'ic_config.yaml'),
        '--out', str(out / 'camera_extrinsics.json'))
    ext = json.loads((out / 'camera_extrinsics.json').read_text())
    cfg = yaml.safe_load((out / 'ic_config.yaml').read_text())
    from calibration.ic_config import T_from_pose
    eg = err(X, T_from_pose(cfg['camera_mount_to_camera_guess']))
    ex = err(X, np.array(ext['T_world_cam']))
    ey = err(Y, np.array(ext['T_flange_board']))
    pw = np.array([0.62, -0.766, 0.25])
    pc = inv(X)[:3, :3] @ pw + inv(X)[:3, 3]
    ws = 1000 * np.linalg.norm(np.array(ext['T_world_cam'])[:3, :3] @ pc + np.array(ext['T_world_cam'])[:3, 3] - pw)
    print(f'initial guess (ic_config): camera {eg[0]:.2f} mm / {eg[1]:.3f} deg')
    print(f'industrial_calibration   : camera {ex[0]:.2f} mm / {ex[1]:.3f} deg, board {ey[0]:.2f} mm / {ey[1]:.3f} deg, '
          f'hand-workspace point {ws:.2f} mm, converged {ext["converged"]}, residual {ext["residual_px"]} px')
    assert ext['converged'] and ex[0] < 3 and ex[1] < 0.3 and ws < 2
    print('ALL OK')


if __name__ == '__main__':
    main()
