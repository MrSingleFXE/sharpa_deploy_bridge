#!/usr/bin/env python3
"""Capture one RGB-D + IR-stereo set from the D435i (pyrealsense2, no ROS, robot not involved).

Writes <out>/:
  color.png              color image (BGR)
  ir_left.png/right.png  rectified IR pair (the input for FoundationStereo)
  depth_raw_ir.npy       RealSense depth in the IR-left frame (metres, 0 = no data)
  depth_raw_color.npy    RealSense depth aligned to the color frame (metres)
  meta.json              intrinsics, IR baseline, IR-left -> color transform, device info, settings

The IR projector (emitter) puts a dot pattern on the scene. It helps both the RealSense depth and
FoundationStereo on texture-less surfaces; use --emitter off only for comparison.
Do not run while a realsense2_camera ROS node is holding the camera.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.camera import select_device  # noqa: E402


def K_of(intr):
    return [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]


def T_of(ext):
    """pyrealsense2 extrinsics (column-major rotation) -> 4x4 mapping points from the source to the target frame."""
    T = np.eye(4)
    T[:3, :3] = np.array(ext.rotation).reshape(3, 3).T
    T[:3, 3] = ext.translation
    return T.tolist()


class D435Rig:
    """Open the D435(i) once (color + depth + both IR streams); grab() writes one capture folder."""

    def __init__(self, serial=None, model='D435I', width=1280, height=720, fps=15, emitter='on', warmup=45):
        import pyrealsense2 as rs
        self.rs = rs
        self.serial = select_device(serial, model)
        self.size, self.emitter = (width, height), emitter
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        cfg.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        self.prof = self.pipe.start(cfg)
        self.dev = self.prof.get_device()
        ds = self.dev.first_depth_sensor()
        if ds.supports(rs.option.emitter_enabled):
            ds.set_option(rs.option.emitter_enabled, 1.0 if emitter == 'on' else 0.0)
        self.depth_scale = ds.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        for _ in range(warmup):
            self.pipe.wait_for_frames()
        sp = {name: self.prof.get_stream(*key).as_video_stream_profile()
              for name, key in (('color', (rs.stream.color,)), ('ir1', (rs.stream.infrared, 1)),
                                ('ir2', (rs.stream.infrared, 2)))}
        ci, ii = sp['color'].get_intrinsics(), sp['ir1'].get_intrinsics()
        self.baseline = float(abs(sp['ir1'].get_extrinsics_to(sp['ir2']).translation[0]))
        self.meta = {
            'device': {'name': self.dev.get_info(rs.camera_info.name), 'serial': self.serial,
                       'firmware': self.dev.get_info(rs.camera_info.firmware_version)},
            'emitter': emitter, 'depth_scale': self.depth_scale, 'size': [width, height],
            'K_color': K_of(ci), 'dist_color': list(ci.coeffs)[:5], 'color_model': str(ci.model),
            'K_ir': K_of(ii), 'dist_ir': list(ii.coeffs)[:5], 'ir_model': str(ii.model),
            'baseline_m': self.baseline,
            'T_color_from_ir_left': T_of(sp['ir1'].get_extrinsics_to(sp['color'])),
            'note': 'IR streams are rectified by the device; depth_raw_ir shares the IR-left pixel grid'}

    def grab(self, out):
        import cv2
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        cv2.imwrite(str(out / 'color.png'), color)
        cv2.imwrite(str(out / 'ir_left.png'), np.asanyarray(frames.get_infrared_frame(1).get_data()))
        cv2.imwrite(str(out / 'ir_right.png'), np.asanyarray(frames.get_infrared_frame(2).get_data()))
        d_ir = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        d_col = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        np.save(out / 'depth_raw_ir.npy', d_ir)
        np.save(out / 'depth_raw_color.npy', d_col)
        meta = dict(self.meta, time=time.strftime('%Y-%m-%d %H:%M:%S'))
        (out / 'meta.json').write_text(json.dumps(meta, indent=2))
        return float((d_col > 0).mean())

    def close(self):
        self.pipe.stop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='capture folder, e.g. captures/scene01')
    p.add_argument('--model', default='D435I')
    p.add_argument('--serial')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--emitter', choices=['on', 'off'], default='on')
    p.add_argument('--warmup', type=int, default=45, help='frames to skip for auto-exposure')
    a = p.parse_args(argv)
    rig = D435Rig(a.serial, a.model, a.width, a.height, a.fps, a.emitter, a.warmup)
    try:
        valid = rig.grab(a.out)
        print(f'saved {a.out}: {rig.meta["device"]["name"]} {rig.serial}, baseline {rig.baseline * 1000:.1f} mm, '
              f'raw depth valid {valid * 100:.1f}% of color pixels, emitter {a.emitter}')
    finally:
        rig.close()


if __name__ == '__main__':
    main()
