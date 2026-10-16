#!/usr/bin/env python3
"""industrial_calibration result -> calib/camera_extrinsics.json (used by object_cloud / verify / hand mount / run_scene).

    python3 calibration/ic_result.py calib/ic_result.yaml --config calib/ic_config.yaml

ic_result.yaml is what the calibration app saves ("Save calibration...") or what install/ic_tools'
ic_extrinsic_cli writes. With the data collected as camera mount = world, target mount = right_ee_mount:
  camera_mount_to_camera  = T_world_cam   (colour optical frame: x right, y down, z forward)
  target_mount_to_target  = T_flange_board
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.ic_config import T_from_pose  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('result', help='ExtrinsicHandEyeResult YAML from industrial_calibration')
    p.add_argument('--config', default='calib/ic_config.yaml', help='the config used for the calibration (intrinsics)')
    p.add_argument('--out', default='calib/camera_extrinsics.json')
    a = p.parse_args(argv)
    res = yaml.safe_load(Path(a.result).read_text())
    cfg = yaml.safe_load(Path(a.config).read_text())
    if not cfg.get('static_camera', False):
        sys.exit('config is not a static-camera calibration')
    X = T_from_pose(res['camera_mount_to_camera'])
    Y = T_from_pose(res['target_mount_to_target'])
    i = cfg['intrinsics']
    K = [[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1]]
    out = {
        'source': 'ROS-Industrial industrial_calibration (extrinsic hand-eye, static camera)',
        'converged': bool(res.get('converged')),
        'residual_px': round(math.sqrt(float(res['final_cost_per_obs'])), 3) if 'final_cost_per_obs' in res else None,
        'frame': 'world = deployment URDF root (left arm base); camera = colour optical frame (x right, y down, z forward)',
        'T_world_cam': X.tolist(),
        'camera_position_world_m': np.round(X[:3, 3], 4).tolist(),
        'camera_viewing_direction_world': np.round(X[:3, 2], 3).tolist(),
        'T_flange_board': Y.tolist(),
        'board_offset_from_flange_m': round(float(np.linalg.norm(Y[:3, 3])), 4),
        'intrinsics': {'K': K, 'source': 'ic_config.yaml'},
        'ic_result_file': str(Path(a.result).resolve()), 'ic_config_file': str(Path(a.config).resolve()),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('converged', 'residual_px', 'camera_position_world_m',
                                          'camera_viewing_direction_world', 'board_offset_from_flange_m')}, indent=2))
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
