#!/usr/bin/env python3
"""Check the camera and board detection (no robot needed, nothing is moved).

Prints intrinsics, grabs frames, reports board detection (corners, reprojection error, distance) and saves
camera_check.png with the detected board axes drawn. Run it with the board held where calibration will happen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, detect_pose  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board', required=True)
    p.add_argument('--camera', choices=['realsense', 'ros'], default='realsense')
    p.add_argument('--color-topic', default='/camera/camera/color/image_raw')
    p.add_argument('--info-topic', default='/camera/camera/color/camera_info')
    p.add_argument('--serial')
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--out', default='calib/camera_check')
    a = p.parse_args()
    import cv2
    spec = BoardSpec.load(a.board)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    io = None
    if a.camera == 'realsense':
        import pyrealsense2 as rs
        devs = [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number),
                 d.get_info(rs.camera_info.firmware_version)) for d in rs.context().query_devices()]
        print('RealSense devices:', devs or 'NONE FOUND')
        from calibration.camera import RealSenseCamera
        cam = RealSenseCamera(serial=a.serial, depth=True)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from calibration.robot_io import RobotIO
        from calibration.camera import RosCamera
        io = RobotIO('bodex_camera_check')
        cam = RosCamera(io.node, io.pump, a.color_topic, a.info_topic, depth_topic='')
    print(json.dumps(cam.intr.to_dict(), indent=2))
    cam.intr.save(out / 'intrinsics.json')
    t0, res, color = time.time(), [], None
    for _ in range(a.frames):
        color, depth = cam.grab()
        res.append(detect_pose(color, cam.intr.K, cam.intr.dist, spec))
    fps = a.frames / (time.time() - t0)
    ok = [r for r in res if r is not None]
    print(f'{fps:.1f} fps, board detected in {len(ok)}/{len(res)} frames')
    if ok:
        r = ok[-1]
        d = [x.T_cam_board[:3, 3] for x in ok]
        print(f'corners {r.corners}, reproj {r.reproj_px:.2f} px, distance {r.T_cam_board[2, 3]:.3f} m, '
              f'position jitter {1000 * np.max(np.ptp(d, axis=0)):.2f} mm')
        rvec = cv2.Rodrigues(r.T_cam_board[:3, :3])[0]
        cv2.drawFrameAxes(color, cam.intr.K, cam.intr.dist, rvec, r.T_cam_board[:3, 3], spec.square_m * 2)
    cv2.imwrite(str(out / 'camera_check.png'), color)
    print(f'saved {out / "camera_check.png"}')
    cam.close()
    if io:
        io.close()


if __name__ == '__main__':
    main()
