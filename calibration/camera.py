"""Camera sources: RealSense SDK (pyrealsense2), ROS 2 topics, or a simulated camera for testing.

Every source returns (color_bgr uint8 HxWx3, depth_m float32 HxW or None) aligned to the color camera,
and exposes K (3x3), dist (5,), size (w, h).
"""

import json
import time
from pathlib import Path

import numpy as np


class Intrinsics:
    def __init__(self, K, dist, size, source):
        self.K, self.dist, self.size, self.source = np.asarray(K, float), np.asarray(dist, float), tuple(size), source

    def to_dict(self):
        return {'K': self.K.tolist(), 'dist': self.dist.tolist(), 'size': list(self.size), 'source': self.source}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def list_devices():
    """[(name, serial, firmware, usb)] for every connected RealSense (USB, not Ethernet)."""
    import pyrealsense2 as rs
    out = []
    for d in rs.context().query_devices():
        info = lambda k: d.get_info(k) if d.supports(k) else '?'
        out.append((info(rs.camera_info.name), info(rs.camera_info.serial_number),
                    info(rs.camera_info.firmware_version), info(rs.camera_info.usb_type_descriptor)))
    return out


def select_device(serial=None, model='D435I'):
    """Pick the camera by serial, else by model name; refuse when ambiguous (e.g. a D405 is also plugged in)."""
    devs = list_devices()
    if not devs:
        raise RuntimeError('no RealSense found (USB cable / USB3 port / udev rules?)')
    if serial:
        if serial not in [d[1] for d in devs]:
            raise RuntimeError(f'serial {serial} not connected; devices: {devs}')
        return serial
    match = [d for d in devs if model.upper() in d[0].upper().replace(' ', '')]
    if len(match) != 1:
        raise RuntimeError(f'{len(match)} devices match model {model!r}; pass --serial. Devices: {devs}')
    return match[0][1]


class RealSenseCamera:
    """Direct SDK access. Do not run at the same time as a realsense2_camera ROS node."""

    def __init__(self, width=1280, height=720, fps=30, serial=None, depth=True, model='D435I'):
        import pyrealsense2 as rs
        self.rs = rs
        serial = select_device(serial, model)
        self.serial = serial
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.depth = depth
        if depth:
            cfg.enable_stream(rs.stream.depth, 848 if width >= 848 else width, 480, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color) if depth else None
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale() if depth else None
        vs = prof.get_stream(rs.stream.color).as_video_stream_profile()
        i = vs.get_intrinsics()
        K = [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]
        self.intr = Intrinsics(K, list(i.coeffs)[:5], (i.width, i.height), f'realsense {serial} {i.model}')
        for _ in range(30):                  # let auto-exposure settle
            self.pipe.wait_for_frames()

    def grab(self):
        frames = self.pipe.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth = None
        if self.depth:
            d = frames.get_depth_frame()
            depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        return color, depth

    def close(self):
        self.pipe.stop()


class RosCamera:
    """Subscribe to realsense2_camera topics (color + camera_info, optional aligned depth).

    `node` is an rclpy node that the caller spins (RobotIO.pump). No cv_bridge needed.
    """

    def __init__(self, node, pump, color_topic='/camera/camera/color/image_raw',
                 info_topic='/camera/camera/color/camera_info',
                 depth_topic='/camera/camera/aligned_depth_to_color/image_raw', timeout=10.0):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        self.pump, self.msgs = pump, {'color': None, 'depth': None, 'info': None}
        node.create_subscription(Image, color_topic, lambda m: self.msgs.__setitem__('color', m), qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info_topic, lambda m: self.msgs.__setitem__('info', m), 10)
        self.depth = bool(depth_topic)
        if self.depth:
            node.create_subscription(Image, depth_topic, lambda m: self.msgs.__setitem__('depth', m),
                                     qos_profile_sensor_data)
        end = time.monotonic() + timeout
        while time.monotonic() < end and (self.msgs['color'] is None or self.msgs['info'] is None):
            pump(0.05)
        if self.msgs['info'] is None or self.msgs['color'] is None:
            raise RuntimeError(f'no camera data on {color_topic} / {info_topic}')
        info = self.msgs['info']
        self.intr = Intrinsics(np.array(info.k).reshape(3, 3), list(info.d)[:5] or [0] * 5,
                               (info.width, info.height), f'ros {info.distortion_model}')

    @staticmethod
    def _decode(msg):
        buf = np.frombuffer(msg.data, dtype=np.uint16 if msg.encoding in ('16UC1', 'mono16') else
                            (np.float32 if msg.encoding == '32FC1' else np.uint8))
        if msg.encoding in ('rgb8', 'bgr8'):
            img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
            return img[:, :, ::-1].copy() if msg.encoding == 'rgb8' else img.copy()
        if msg.encoding in ('16UC1', 'mono16'):
            return buf.reshape(msg.height, msg.step // 2)[:, :msg.width].astype(np.float32) / 1000.0
        if msg.encoding == '32FC1':
            return buf.reshape(msg.height, msg.step // 4)[:, :msg.width].copy()
        if msg.encoding == 'mono8':
            g = buf.reshape(msg.height, msg.step)[:, :msg.width]
            return np.repeat(g[:, :, None], 3, axis=2)
        raise ValueError(f'unsupported encoding {msg.encoding}')

    def grab(self):
        stamp = self.msgs['color'].header.stamp
        end = time.monotonic() + 2.0
        while time.monotonic() < end:           # wait for a frame newer than the last one returned
            self.pump(0.01)
            if self.msgs['color'].header.stamp != stamp:
                break
        depth = self._decode(self.msgs['depth']) if self.depth and self.msgs['depth'] is not None else None
        return self._decode(self.msgs['color']), depth

    def close(self):
        pass


class SimCamera:
    """Renders the board from robot FK and a ground-truth extrinsic (testing only)."""

    def __init__(self, spec, T_world_cam, T_flange_board, flange_pose_fn, size=(1280, 720), noise_px=0.3, seed=0):
        from .board import synth_view
        self.synth_view = synth_view
        f = 910.0
        K = [[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]]
        self.intr = Intrinsics(K, [0] * 5, size, 'sim')
        self.spec, self.X, self.Y, self.flange_pose_fn = spec, T_world_cam, T_flange_board, flange_pose_fn
        self.rng, self.noise = np.random.default_rng(seed), noise_px

    def grab(self):
        import cv2
        T_cam_board = np.linalg.inv(self.X) @ self.flange_pose_fn() @ self.Y
        img = self.synth_view(self.spec, T_cam_board, self.intr.K, self.intr.size).astype(np.float32)
        img += self.rng.normal(0, 2.0, img.shape)
        img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), self.noise)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None

    def close(self):
        pass
