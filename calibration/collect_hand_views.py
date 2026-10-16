#!/usr/bin/env python3
"""Collect views of the right hand for the hand-mount calibration (READ-ONLY: never sends motion commands).

Run with the robot launched read_only:=true; move the right arm with the teach pendant to 5-8 poses where the whole
hand is visible to the camera (different wrist orientations, 0.5-0.9 m from the camera, away from the table and
other objects). Keep the hand still in one shape (e.g. open/zero) -- its joint angles are recorded anyway.
At each pose press Enter: saves <out>/view_XX/ with color, IR pair, RealSense depth, meta.json and robot.json
(arm joints, T_world_flange from the robot model, hand joints).

Then:  perception/fs_depth.py <out>/view_*   (FoundationStereo depth, fs_venv)
       calibration/hand_mount_icp.py <out>/view_* --extrinsics calib/camera_extrinsics.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.urdf_kin import URDFModel  # noqa: E402
from calibration.robot_io import ARM, FLANGE, Abort, RobotIO, ask  # noqa: E402
from perception.capture import RealSenseRig  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='e.g. calib/hand_views')
    p.add_argument('--model', help='RealSense model when --serial is not given (default D415, env SHARPA_CAMERA_MODEL)')
    p.add_argument('--serial')
    p.add_argument('--table-z', type=float, help='world z of the table (points below are ignored by the fit)')
    p.add_argument('--urdf')
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    io = RobotIO('bodex_hand_views', urdf_path=a.urdf)
    rig = None
    try:
        io.wait_ready(need_hand=True)
        model = URDFModel(io.st['urdf'])
        rig = RealSenseRig(a.serial, a.model)
        n = len(list(out.glob('view_*')))
        print('READ-ONLY: move the arm with the pendant, let it stop, then Enter = capture, q = finish.')
        while True:
            c = ask(f'[{n} views] Enter=capture  q=finish > ')
            if c == 'q':
                break
            if not io.stationary():
                print('  arm still moving; wait and retry')
                continue
            q = io.st['arm'].copy()
            hand, names = io.st['hand'].copy(), list(io.st['hand_names'])
            f = out / f'view_{n:02d}'
            valid = rig.grab(f)
            io.pump(0.1)
            if abs(io.st['arm'] - q).max() > 1e-3:
                print('  arm moved during capture; discarded')
                continue
            A = model.fk(dict(zip(ARM, q)), FLANGE)
            (f / 'robot.json').write_text(json.dumps({
                'arm_rad': q.tolist(), 'T_world_flange': A.tolist(), 'flange_link': FLANGE,
                'hand_rad': hand.tolist(), 'hand_names': names, 'table_z_world_m': a.table_z}, indent=2))
            print(f'  saved {f} (raw depth valid {valid * 100:.0f}%)')
            n += 1
        return 0
    except Abort as e:
        print(f'STOPPED: {e}')
        return 2
    finally:
        if rig:
            rig.close()
        io.close()


if __name__ == '__main__':
    sys.exit(main())
