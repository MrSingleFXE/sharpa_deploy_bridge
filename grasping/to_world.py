#!/usr/bin/env python3
"""Table-frame grasps -> robot world frame, ready for execute_grasp.py --wrist-frame world (needs calibration).

Input : selected.npy (grasping/select_grasps.py) and object.json written by perception/object_cloud.py WITH
        --extrinsics (it contains T_world_table = T_world_cam @ T_cam_table).
Output: <grasps dir>/grasps_world.npy  BODex-style dict (pregrasp/grasp/squeeze_qpos (K,29), wrist = right_hand_C_MC
        in world), and prints the execute_grasp.py command (dry run first) for the best candidate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.hand_model import qpos_to_T  # noqa: E402

STAGES = ('pregrasp_qpos', 'grasp_qpos', 'squeeze_qpos')


def T_to_qpos(T, joints):
    R = T[:3, :3]
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.r_[T[:3, 3], [w, x, y, z], joints]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('selected', help='selected.npy (table frame)')
    p.add_argument('object_json', help='object.json from object_cloud.py --extrinsics ...')
    a = p.parse_args(argv)
    info = json.loads(Path(a.object_json).read_text())
    if 'T_world_table' not in info:
        sys.exit('object.json has no T_world_table: run perception/object_cloud.py with --extrinsics (calibration)')
    Tw = np.array(info['T_world_table'])
    g = np.load(a.selected, allow_pickle=True).item()
    out = dict(g)
    for s in STAGES:
        out[s] = np.stack([T_to_qpos(Tw @ qpos_to_T(q), q[7:]) for q in g[s]])
    out['frame'] = 'world'
    out['T_world_table'] = Tw
    dst = Path(a.selected).parent / 'grasps_world.npy'
    np.save(dst, out, allow_pickle=True)
    obj_world = Path(a.object_json).parent / 'object_world.npy'
    print(f'saved {dst}: {len(out["grasp_qpos"])} grasps (best first), table z in world {info["table_z_world_m"]} m')
    for i in range(min(3, len(out['grasp_qpos']))):
        print(f'  #{i}: grasp wrist world xyz {np.round(out["grasp_qpos"][i][:3], 3).tolist()}, '
              f'log_prob {float(g["log_prob"][i]):.1f}')
    print('\nnext (dry run first; add --execute only after reviewing the plan):')
    print(f'  python3 execute_grasp.py --grasp {dst} --index 0 --wrist-frame world '
          f'--table-z {info["table_z_world_m"]} --clearance 0.005 --obstacles {obj_world} '
          f'--hand-mount calib/hand_mount.json --stop-after pregrasp')


if __name__ == '__main__':
    main()
