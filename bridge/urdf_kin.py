"""Minimal URDF forward kinematics, geometric Jacobian and DLS IK (numpy only).

Independent of the deployment workspace: it parses whatever URDF string it is
given (normally the live /bimanual/robot_description).
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, angle):
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def transform(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def quat_wxyz_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_error(R_target, R_current):
    """Rotation vector (world axes) taking R_current to R_target."""
    R = R_target @ R_current.T
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Near pi: take the axis from the symmetric part.
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        return v[:, np.argmax(w)] * angle
    return angle / (2 * math.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class Joint:
    def __init__(self, el):
        self.name, self.type = el.get('name'), el.get('type')
        self.parent, self.child = el.find('parent').get('link'), el.find('child').get('link')
        origin = el.find('origin')
        xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
        rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
        self.origin = transform(rpy_matrix(*rpy), xyz)
        axis = el.find('axis')
        a = np.array([float(v) for v in (axis.get('xyz') if axis is not None else '1 0 0').split()])
        self.axis = a / np.linalg.norm(a) if np.linalg.norm(a) > 0 else a
        limit = el.find('limit')
        if self.type in ('revolute', 'prismatic') and limit is not None:
            self.lower, self.upper = float(limit.get('lower')), float(limit.get('upper'))
        else:
            self.lower, self.upper = -math.inf, math.inf
        if el.find('mimic') is not None:
            raise ValueError(f'mimic joints are not supported: {self.name}')

    def local(self, q):
        if self.type in ('revolute', 'continuous'):
            return self.origin @ transform(axis_angle(self.axis, q))
        if self.type == 'prismatic':
            return self.origin @ transform(p=self.axis * q)
        return self.origin


class URDFModel:
    def __init__(self, urdf_xml):
        root = ET.fromstring(urdf_xml)
        self.joints = {j.name: j for j in (Joint(el) for el in root.findall('joint'))}
        self.child_joint = {j.child: j for j in self.joints.values()}
        links = {el.get('name') for el in root.findall('link')}
        roots = links - set(self.child_joint)
        if len(roots) != 1:
            raise ValueError(f'URDF must have exactly one root link, found {sorted(roots)}')
        self.root = roots.pop()
        self.links = links
        self.movable = [n for n, j in self.joints.items() if j.type != 'fixed']

    def chain(self, tip):
        """Joints from root to tip, root first."""
        out, link = [], tip
        while link != self.root:
            j = self.child_joint[link]
            out.append(j)
            link = j.parent
        return out[::-1]

    def link_poses(self, q):
        """World pose of every link, q: dict joint name -> value (missing = 0)."""
        poses = {self.root: np.eye(4)}

        def pose(link):
            if link not in poses:
                j = self.child_joint[link]
                poses[link] = pose(j.parent) @ j.local(q.get(j.name, 0.0))
            return poses[link]

        for link in self.links:
            pose(link)
        return poses

    def fk(self, q, tip):
        T = np.eye(4)
        for j in self.chain(tip):
            T = T @ j.local(q.get(j.name, 0.0))
        return T

    def jacobian(self, q, tip, joint_names):
        """6xN geometric Jacobian (linear; angular) in world axes for the given joints."""
        T, frames = np.eye(4), {}
        for j in self.chain(tip):
            T = T @ j.origin
            frames[j.name] = (T[:3, 3].copy(), T[:3, :3] @ j.axis)
            if j.type in ('revolute', 'continuous'):
                T = T @ transform(axis_angle(j.axis, q.get(j.name, 0.0)))
            elif j.type == 'prismatic':
                T = T @ transform(p=j.axis * q.get(j.name, 0.0))
        p_tip = T[:3, 3]
        J = np.zeros((6, len(joint_names)))
        for i, name in enumerate(joint_names):
            if name not in frames:
                continue
            p, z = frames[name]
            if self.joints[name].type == 'prismatic':
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p_tip - p)
                J[3:, i] = z
        return J, T

    def limits(self, joint_names):
        return (np.array([self.joints[n].lower for n in joint_names]),
                np.array([self.joints[n].upper for n in joint_names]))


def solve_ik(model, target, tip, joint_names, seed_q, fixed_q=None, *, iters=300,
             damping=0.02, max_step=0.05, pos_tol=1e-4, rot_tol=1e-3):
    """Damped least squares IK on joint_names; other joints held at fixed_q.

    Returns (success, q_vector, position_error_m, rotation_error_rad).
    """
    lower, upper = model.limits(joint_names)
    q = np.clip(np.asarray(seed_q, dtype=float).copy(), lower, upper)
    base = dict(fixed_q or {})
    for _ in range(iters + 1):
        full = dict(base, **dict(zip(joint_names, q)))
        J, T = model.jacobian(full, tip, joint_names)
        err = np.concatenate([target[:3, 3] - T[:3, 3], rotvec_error(target[:3, :3], T[:3, :3])])
        pe, re = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        if pe <= pos_tol and re <= rot_tol:
            return True, q, pe, re
        step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
        m = np.max(np.abs(step))
        if m > max_step:
            step *= max_step / m
        q = np.clip(q + step, lower, upper)
    return False, q, pe, re
