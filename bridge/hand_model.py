"""Sharpa right hand as surface points (from the URDF meshes) for any wrist pose + 22 joints (numpy only).

The wrist pose is that of `right_hand_C_MC` (BODex / DexLearn convention). Meshes are read from the
deployment package (`package://dual_sharpa_wave/...` -> the dual_sharpa_wave_ros2 source folder).
"""

import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .urdf_kin import URDFModel, rpy_matrix


def apply(points, T):
    return points @ T[:3, :3].T + T[:3, 3]

ROOT_LINK = 'right_hand_C_MC'
SEARCH = ['~/ws_fanuc/src/dual_sharpa_wave_ros2']      # elsewhere: set SHARPA_HAND_URDF


def find_hand_urdf(path=None):
    cands = [path] if path else [os.environ.get('SHARPA_HAND_URDF')] + \
        [f'{d}/urdf/right_sharpa_wave_with_flange.urdf' for d in SEARCH]
    for c in cands:
        if c and Path(c).expanduser().exists():
            return Path(c).expanduser().resolve()
    raise FileNotFoundError('right_sharpa_wave_with_flange.urdf not found; pass --hand-urdf or set SHARPA_HAND_URDF')


def read_stl(path):
    data = Path(path).read_bytes()
    if data[:5].lower() == b'solid' and b'facet' in data[:1000]:
        v = [list(map(float, l.split()[1:4])) for l in data.decode(errors='ignore').splitlines()
             if l.strip().startswith('vertex')]
        return np.array(v, float).reshape(-1, 3, 3)
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    return rec['v'].astype(float)


def sample_triangles(tri, n, rng):
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if a.sum() <= 0:
        return np.zeros((0, 3))
    idx = rng.choice(len(tri), n, p=a / a.sum())
    r1, r2 = rng.random((2, n))
    s = np.sqrt(r1)
    t = tri[idx]
    return (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]


class HandModel:
    def __init__(self, urdf_path=None, kind='collision', points_per_m2=60000, min_per_link=40, seed=0):
        urdf_path = find_hand_urdf(urdf_path)
        text = urdf_path.read_text()
        self.kin = URDFModel(text)
        pkg_root = urdf_path.parent.parent                       # dual_sharpa_wave_ros2
        rng = np.random.default_rng(seed)
        self.link_points = {}
        for link in ET.fromstring(text).findall('link'):
            pts = []
            for el in link.findall(kind):
                mesh = el.find('geometry/mesh')
                if mesh is None:
                    continue
                fn = mesh.get('filename').replace('package://dual_sharpa_wave/', str(pkg_root) + '/')
                tri = read_stl(fn) * np.array([float(s) for s in mesh.get('scale', '1 1 1').split()])
                area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
                p = sample_triangles(tri, max(min_per_link, int(area * points_per_m2)), rng)
                o = el.find('origin')
                if o is not None:
                    T = np.eye(4)
                    T[:3, :3] = rpy_matrix(*map(float, o.get('rpy', '0 0 0').split()))
                    T[:3, 3] = list(map(float, o.get('xyz', '0 0 0').split()))
                    p = apply(p, T)
                pts.append(p)
            if pts:
                self.link_points[link.get('name')] = np.concatenate(pts)
        self.joint_names = [n for n in self.kin.movable]
        self.T_root_in_urdf = self.kin.fk({}, ROOT_LINK)         # right_hand_C_MC in the URDF root (flange)
        self.lower, self.upper = self.kin.limits(self.joint_names)

    def points(self, wrist_T, joints, joint_names, links=None):
        """Surface points (M,3) for wrist pose (4x4 of right_hand_C_MC) and joints (ordered as joint_names)."""
        q = dict(zip(joint_names, joints))
        poses = self.kin.link_poses(q)
        base = wrist_T @ np.linalg.inv(self.T_root_in_urdf)
        out = [apply(p, base @ poses[l]) for l, p in self.link_points.items() if links is None or l in links]
        return np.concatenate(out)

    def points_in_root(self, joints, joint_names, links=None):
        """Surface points in the URDF root frame (right_hand_flange = the hand's mounting face)."""
        q = dict(zip(joint_names, joints))
        poses = self.kin.link_poses(q)
        return np.concatenate([apply(p, poses[l]) for l, p in self.link_points.items() if links is None or l in links])

    def limit_violation(self, joints, joint_names):
        q = dict(zip(joint_names, joints))
        v = np.array([q[n] for n in self.joint_names])
        return float(np.max(np.maximum(self.lower - v, v - self.upper)))


def qpos_to_T(q):
    """BODex 29D -> 4x4 wrist pose."""
    from .urdf_kin import quat_wxyz_to_matrix
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(q[3:7])
    T[:3, 3] = q[:3]
    return T
