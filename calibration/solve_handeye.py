#!/usr/bin/env python3
"""Solve camera extrinsics from collected samples (no ROS, no robot needed).

Input : one or more collection folders (samples.jsonl + intrinsics.json + board.json)
Output: camera_extrinsics.json  -- T_world_cam (deployment `world` = left arm base), T_flange_board,
        quality report. Prints a verdict (workspace = where the hand was during collection):
          GOOD   : translation RMS <= 3 mm, leave-one-out workspace shift <= 2 mm, >= 12 samples,
                   rotation-axis diversity >= 45 deg
          USABLE : translation RMS <= 6 mm, leave-one-out shift <= 5 mm, >= 8 samples, diversity >= 30 deg
          REDO   : otherwise (collect more varied poses / check board flatness and measured size)
        The per-sample rotation residual mostly reflects single-image board orientation noise (larger for a
        small board far away); it is reported, with a warning above 1 deg, but does not decide the verdict.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.handeye import solve  # noqa: E402


def quat_wxyz(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def rpy(R):
    return [math.atan2(R[2, 1], R[2, 2]), math.asin(-max(-1, min(1, R[2, 0]))), math.atan2(R[1, 0], R[0, 0])]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sessions', nargs='+', help='collection folders')
    p.add_argument('--out', default='calib/camera_extrinsics.json')
    p.add_argument('--max-reproj-px', type=float, default=1.0, help='drop captures with larger PnP error')
    p.add_argument('--max-spread-mm', type=float, default=2.0, help='drop captures whose frames disagree more')
    p.add_argument('--no-outlier-drop', action='store_true')
    args = p.parse_args(argv)

    A, B, used, dropped, intr = [], [], [], [], None
    for s in args.sessions:
        s = Path(s)
        this_intr = json.loads((s / 'intrinsics.json').read_text())
        if intr is not None and np.abs(np.array(this_intr['K']) - np.array(intr['K'])).max() > 1.0:
            sys.exit(f'{s}: camera intrinsics differ from the first session (different resolution/camera?)')
        intr = this_intr
        for line in open(s / 'samples.jsonl'):
            r = json.loads(line)
            tag = f'{s.name}#{r["index"]}'
            if r['reproj_px'] > args.max_reproj_px or r['frame_spread_mm'] > args.max_spread_mm:
                dropped.append((tag, f'reproj {r["reproj_px"]:.2f}px spread {r["frame_spread_mm"]:.2f}mm'))
                continue
            A.append(np.array(r['T_world_flange']))
            B.append(np.array(r['T_cam_board']))
            used.append(tag)
    if len(A) < 6:
        sys.exit(f'only {len(A)} usable samples; need >= 6 (>= 12 recommended)')

    res = solve(A, B, drop_outliers=not args.no_outlier_drop)
    X, Y, r, keep = res['X'], res['Y'], res['residuals'], res['keep']
    rk = r[keep]
    rms_mm, rms_deg = float(np.sqrt(np.mean(rk[:, 1] ** 2))), float(np.sqrt(np.mean(rk[:, 0] ** 2)))
    n = len(keep)
    div = res['rotation_diversity_deg']
    loo = res['loo_workspace_mm']
    loo_max = loo['max_mm'] if loo else float('inf')
    if rms_mm <= 3 and loo_max <= 2 and n >= 12 and div >= 45:
        verdict = 'GOOD'
    elif rms_mm <= 6 and loo_max <= 5 and n >= 8 and div >= 30:
        verdict = 'USABLE'
    else:
        verdict = 'REDO'
    report = {
        'verdict': verdict,
        'frame': 'world = deployment URDF root (left arm base); camera = color optical frame (x right, y down, z forward)',
        'T_world_cam': X.tolist(),
        'camera_position_world_m': np.round(X[:3, 3], 4).tolist(),
        'camera_quat_wxyz': np.round(quat_wxyz(X[:3, :3]), 6).tolist(),
        'camera_rpy_rad': np.round(rpy(X[:3, :3]), 6).tolist(),
        'camera_viewing_direction_world': np.round(X[:3, 2], 3).tolist(),
        'T_flange_board': Y.tolist(),
        'board_offset_from_flange_m': round(float(np.linalg.norm(Y[:3, 3])), 4),
        'samples_used': n, 'samples_total': len(A) + len(dropped),
        'rms_mm': round(rms_mm, 3), 'rms_deg': round(rms_deg, 4),
        'max_mm': round(float(rk[:, 1].max()), 3), 'max_deg': round(float(rk[:, 0].max()), 4),
        'rotation_axis_diversity_deg': round(div, 1),
        'leave_one_out_workspace_shift_mm': loo,
        'warnings': (['rotation residual > 1 deg: board far/small or loosely mounted; consider a larger (A3) board']
                     if rms_deg > 1.0 else []),
        'init_method': res['init_method'], 'method_comparison': res['methods'],
        'per_sample': [{'sample': used[i], 'mm': round(float(r[i, 1]), 3), 'deg': round(float(r[i, 0]), 4),
                        'kept': i in keep} for i in range(len(used))],
        'dropped_before_solve': [{'sample': t, 'why': w} for t, w in dropped],
        'intrinsics': intr, 'sessions': [str(s) for s in args.sessions],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('verdict', 'camera_position_world_m', 'camera_viewing_direction_world',
                                             'board_offset_from_flange_m', 'samples_used', 'rms_mm', 'rms_deg',
                                             'max_mm', 'max_deg', 'rotation_axis_diversity_deg',
                                             'leave_one_out_workspace_shift_mm', 'warnings', 'init_method')},
                     indent=2))
    print(f'saved {out}')
    return 0 if verdict != 'REDO' else 1


if __name__ == '__main__':
    sys.exit(main())
