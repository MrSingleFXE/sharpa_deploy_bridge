#!/usr/bin/env python3
"""Check a recorded mock run: feedback speeds, clearance and wrist accuracy (uses the same planner model)."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.planner import ARM, TIP, Limits, Planner, load_grasp  # noqa: E402

rec, urdf, grasp_path, plan_json = sys.argv[1:5]
d = np.load(rec)
g = load_grasp(grasp_path)
plan = json.loads(Path(plan_json).read_text())
pl = Planner(Path(urdf).read_text(), g.joint_names, Limits(table_z=plan['limits']['table_z'], clearance=1e-6))
arm, hand = d['arm'], d['hand']
t0 = arm[0, 0]
def speed(x):
    # finite differences over >= 0.1 s windows (feedback is ~100 Hz)
    k = 10
    return np.degrees(np.abs((x[k:, 1:] - x[:-k, 1:]) / (x[k:, :1] - x[:-k, :1])).max())
moving = np.abs(np.diff(arm[:, 1:], axis=0)).max(axis=1) > 1e-6
print('arm feedback samples', len(arm), 'duration s', round(arm[-1, 0] - t0, 1))
print('peak arm joint speed (deg/s)', round(speed(arm), 2), '| peak hand joint speed (deg/s)', round(speed(hand), 2))
# clearance along actual feedback, hand interpolated onto arm timestamps
hi = np.stack([np.interp(arm[:, 0], hand[:, 0], hand[:, 1 + j]) for j in range(22)], axis=1)
cl = [pl.clearance(arm[i, 1:], hi[i])[0] for i in range(0, len(arm), 5)]
print('min clearance along recorded motion (m)', round(min(cl), 4))
end = arm[-1, 1:]
print('final wrist xyz', np.round(pl.wrist_pose(end)[:3, 3], 4).tolist())
for s, xyz in plan['wrist_world_xyz'].items():
    dmin = min(np.linalg.norm(pl.wrist_pose(arm[i, 1:])[:3, 3] - np.array(xyz)) for i in range(0, len(arm), 5))
    print(f'closest recorded wrist to planned {s}: {dmin*1000:.2f} mm')
