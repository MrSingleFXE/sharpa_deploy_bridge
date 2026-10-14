"""BODex Sharpa grasp -> right CRX arm + right Sharpa hand joint trajectories.

No ROS imports here; everything is testable offline with a URDF string.

Frames
- BODex wrist pose (first 7 of the 29D qpos) is the pose of `right_hand_C_MC`
  in the BODex scene frame (z up, table plane z=0).
- World = the deployment model's `world` (left arm base). We place the scene
  frame in world by a yaw about z plus a translation chosen so that the grasp
  wrist lands on an anchor point (default: where the hand currently is).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .urdf_kin import URDFModel, axis_angle, quat_wxyz_to_matrix, rotvec_error, solve_ik, transform

ARM = [f'right_J{i}' for i in range(1, 7)]
TIP = 'right_hand_C_MC'
STAGES = ('pregrasp', 'grasp', 'squeeze')


# ----------------------------------------------------------------- input
@dataclass
class GraspInput:
    joint_names: list
    wrist: dict          # stage -> 4x4 pose of right_hand_C_MC (scene or world frame)
    hand: dict           # stage -> 22 joint values ordered as joint_names
    source: str


def load_grasp(path, index=0):
    """Load a BODex succgrasp .npy (dict) or a .json with the same keys."""
    path = Path(path)
    if path.suffix == '.npy':
        data = np.load(path, allow_pickle=True)
        data = data.item() if data.dtype == object and data.shape == () else data
    else:
        data = json.loads(path.read_text())
    names = [str(n) for n in data['joint_names']]
    if len(names) != 22 or len(set(names)) != 22 or not all(n.startswith('right_') for n in names):
        raise ValueError('expected 22 unique right_* joint names')
    model_id = data.get('hand_model_id')
    if model_id is not None and str(model_id) != 'sharpa_right_hand_only_conservative_v1':
        raise ValueError(f'unexpected hand_model_id {model_id}')
    wrist, hand = {}, {}
    for stage in STAGES:
        q = np.asarray(data[f'{stage}_qpos'], dtype=float)
        if q.ndim == 2:
            q = q[index]
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError(f'{stage}_qpos must be 29 finite values')
        wrist[stage] = transform(quat_wxyz_to_matrix(q[3:7]), q[:3])
        hand[stage] = q[7:].copy()
    return GraspInput(names, wrist, hand, f'{path}#{index}')


# ----------------------------------------------------------------- config
@dataclass
class Limits:
    rate_hz: float = 100.0
    arm_vel: float = math.radians(5)      # rad/s, per joint
    arm_acc: float = math.radians(10)
    hand_vel: float = math.radians(10)
    hand_acc: float = math.radians(20)
    min_duration: float = 3.0
    table_z: float = 0.0                  # world z of the table top
    clearance: float = 0.10               # every checked link origin must stay this far above the table
    approach_lift: float = 0.08           # pre-approach point is this far above the pregrasp wrist
    left_arm_distance: float = 0.20       # min distance between right moving link origins and left arm link origins
    cart_step: float = 0.005              # m between Cartesian IK waypoints
    max_ik_jump: float = 0.08             # rad between consecutive Cartesian IK waypoints

    # Hard caps for first tests; the CLI refuses anything above them.
    ARM_VEL_CAP = math.radians(20)
    HAND_VEL_CAP = math.radians(30)

    def validate(self):
        vals = [self.rate_hz, self.arm_vel, self.arm_acc, self.hand_vel, self.hand_acc,
                self.min_duration, self.clearance, self.approach_lift, self.cart_step]
        if not np.isfinite(vals + [self.table_z]).all() or min(vals) <= 0:
            raise ValueError('limits must be finite and positive')
        if self.arm_vel > self.ARM_VEL_CAP or self.hand_vel > self.HAND_VEL_CAP:
            raise ValueError(f'speed above first-test cap (arm {math.degrees(self.ARM_VEL_CAP):g} deg/s, '
                             f'hand {math.degrees(self.HAND_VEL_CAP):g} deg/s)')


# ----------------------------------------------------------------- plan
@dataclass
class Segment:
    name: str
    arm: np.ndarray        # (N, 6) path samples (uniform in path parameter)
    hand: np.ndarray       # (N, 22)
    duration: float = 0.0

    def sample(self, t):
        """Joint targets at time t with quintic (zero vel/acc ends) timing."""
        u = min(max(t / self.duration, 0.0), 1.0)
        s = u ** 3 * (10 + u * (-15 + 6 * u)) * (len(self.arm) - 1)
        i = min(int(s), len(self.arm) - 2)
        f = s - i
        return (self.arm[i] * (1 - f) + self.arm[i + 1] * f,
                self.hand[i] * (1 - f) + self.hand[i + 1] * f)


@dataclass
class Plan:
    segments: list
    yaw_deg: float
    scene_in_world: np.ndarray
    wrist_world: dict
    report: dict = field(default_factory=dict)


class Planner:
    def __init__(self, urdf_xml, hand_names, limits: Limits):
        limits.validate()
        self.m, self.lim, self.hand_names = URDFModel(urdf_xml), limits, list(hand_names)
        missing = [n for n in ARM + self.hand_names + ['right_J1'] if n not in self.m.joints]
        if missing or TIP not in self.m.links:
            raise ValueError(f'URDF lacks right arm/hand joints or {TIP}: {missing[:5]}')
        self.arm_lo, self.arm_hi = self.m.limits(ARM)
        self.hand_lo, self.hand_hi = self.m.limits(self.hand_names)
        # Links checked for table clearance: everything moved by right_J2 and below.
        self.sub_root = self.m.joints['right_J2'].parent
        # Table clearance: links that can change height, i.e. from the elbow (J3) outward.
        # The J2 (shoulder) frame sits at a fixed height above the base and is excluded.
        self.check_links = self._descendants(self.m.joints['right_J3'].child)
        # Joints of the moving subtree in parent-before-child order.
        self.sub_joints, todo = [], [self.sub_root]
        while todo:
            l = todo.pop(0)
            for j in self.m.joints.values():
                if j.parent == l:
                    self.sub_joints.append(j)
                    todo.append(j.child)
        self.left_links = ([l for l in self._descendants(self.m.joints['left_J1'].child) if l in self.m.links]
                           if 'left_J1' in self.m.joints else [])
        self.set_left_arm(np.zeros(6))

    def _descendants(self, link):
        out, todo = [], [link]
        while todo:
            l = todo.pop()
            out.append(l)
            todo += [j.child for j in self.m.joints.values() if j.parent == l]
        return out

    def qdict(self, arm, hand):
        return dict(zip(ARM, arm), **dict(zip(self.hand_names, hand)))

    def wrist_pose(self, arm):
        return self.m.fk(dict(zip(ARM, arm)), TIP)

    def set_left_arm(self, q_left):
        """Left arm joints (6, rad) for the right-left distance check; left arm assumed static."""
        q = {f'left_J{i}': float(v) for i, v in enumerate(q_left, start=1)}
        poses = self.m.link_poses(q) if self.left_links else {}
        self.left_points = np.array([poses[l][:3, 3] for l in self.left_links]) if self.left_links else None

    def moving_poses(self, arm, hand):
        q = self.qdict(arm, hand)
        poses = {self.sub_root: self.m.fk(q, self.sub_root)}
        for j in self.sub_joints:
            poses[j.child] = poses[j.parent] @ j.local(q.get(j.name, 0.0))
        return poses

    def clearance(self, arm, hand):
        poses = self.moving_poses(arm, hand)
        z, link = min((poses[l][2, 3], l) for l in self.check_links)
        if self.left_points is not None:
            R = np.array([poses[l][:3, 3] for l in self.check_links])
            d = np.linalg.norm(R[:, None] - self.left_points[None], axis=2).min()
            if d < self.lim.left_arm_distance:
                raise PlanError(f'right arm/hand comes {d:.3f} m from the left arm')
        return z - self.lim.table_z, link

    # -- geometry
    def place(self, grasp, anchor_world, yaw):
        """Scene frame in world: rotate by yaw about z, grasp wrist lands on anchor."""
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        p = np.asarray(anchor_world) - R @ grasp.wrist['grasp'][:3, 3]
        return transform(R, p)

    # -- segments
    def _timed(self, seg):
        arm_d = np.abs(np.diff(seg.arm, axis=0)).max() * (len(seg.arm) - 1) if len(seg.arm) > 1 else 0.0
        hand_d = np.abs(np.diff(seg.hand, axis=0)).max() * (len(seg.hand) - 1) if len(seg.hand) > 1 else 0.0
        # Max of smootherstep derivatives: s' = 15/8, s'' = 10/sqrt(3) (unit time).
        t = self.lim.min_duration
        for d, v, a in ((arm_d, self.lim.arm_vel, self.lim.arm_acc), (hand_d, self.lim.hand_vel, self.lim.hand_acc)):
            t = max(t, 1.875 * d / v * 1.05, math.sqrt(10 / math.sqrt(3) * d / a) * 1.05)
        seg.duration = t
        return seg

    def joint_segment(self, name, a0, a1, h0, h1):
        # One path sample per <= 1 deg of the largest joint change (checked for clearance).
        n = max(20, int(math.ceil(math.degrees(max(np.abs(a1 - a0).max(), np.abs(h1 - h0).max())))) + 1)
        s = np.linspace(0, 1, n)[:, None]
        return self._timed(Segment(name, a0 + s * (a1 - a0), h0 + s * (h1 - h0)))

    def cartesian_segment(self, name, a0, T1, h0, h1):
        """Straight-line wrist motion from FK(a0) to T1 by sequential IK."""
        T0 = self.wrist_pose(a0)
        dist = np.linalg.norm(T1[:3, 3] - T0[:3, 3])
        ang = np.linalg.norm(rotvec_error(T1[:3, :3], T0[:3, :3]))
        n = max(2, int(math.ceil(max(dist / self.lim.cart_step, ang / math.radians(1)))) + 1)
        rv = rotvec_error(T1[:3, :3], T0[:3, :3])
        arms, q = [a0.copy()], a0.copy()
        for k in range(1, n):
            s = k / (n - 1)
            R = (axis_angle(rv / np.linalg.norm(rv), s * np.linalg.norm(rv)) if np.linalg.norm(rv) > 1e-12
                 else np.eye(3)) @ T0[:3, :3]
            T = transform(R, T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3]))
            ok, qn, pe, re = solve_ik(self.m, T, TIP, ARM, q)
            if not ok:
                raise PlanError(f'{name}: IK failed at waypoint {k}/{n - 1} (pos {pe:.4f} m, rot {re:.4f} rad)')
            if np.abs(qn - q).max() > self.lim.max_ik_jump:
                raise PlanError(f'{name}: IK branch jump {np.abs(qn - q).max():.3f} rad at waypoint {k}')
            arms.append(qn)
            q = qn
        s = np.linspace(0, 1, n)[:, None]
        return self._timed(Segment(name, np.array(arms), h0 + s * (h1 - h0)))

    # -- checks
    def check(self, seg, every=1, floor=None):
        floor = self.lim.clearance if floor is None else floor
        worst = (math.inf, '')
        for i in range(0, len(seg.arm), every):
            a, h = seg.arm[i], seg.hand[i]
            if np.any(a < self.arm_lo - 1e-9) or np.any(a > self.arm_hi + 1e-9):
                raise PlanError(f'{seg.name}: arm joint limit violated at sample {i}')
            if np.any(h < self.hand_lo - 1e-9) or np.any(h > self.hand_hi + 1e-9):
                bad = [self.hand_names[j] for j in np.where((h < self.hand_lo) | (h > self.hand_hi))[0]]
                raise PlanError(f'{seg.name}: hand joint outside URDF limit {bad}')
            c = self.clearance(a, h)
            worst = min(worst, c)
            if c[0] < floor:
                raise PlanError(f'{seg.name}: link {c[1]} only {c[0]:.3f} m above table '
                                f'(need >= {floor:.3f})')
        return worst

    # -- full plan
    def plan(self, grasp, arm_now, hand_now, *, anchor=None, anchor_offset=(0, 0, 0), wrist_frame='scene',
             stop_after='squeeze', release=False, yaw_list=None):
        if list(grasp.joint_names) != self.hand_names:
            raise PlanError('grasp joint order differs from hand driver order')
        arm_now, hand_now = np.asarray(arm_now, float), np.asarray(hand_now, float)
        # Measured hand pose may sit slightly outside the URDF limits; the MIT driver
        # clips commands anyway. Accept up to 3 deg, otherwise refuse.
        over = np.maximum(self.hand_lo - hand_now, hand_now - self.hand_hi).max()
        if over > math.radians(3):
            raise PlanError(f'current hand pose {math.degrees(over):.1f} deg outside URDF limits; home the hand first')
        hand_now = np.clip(hand_now, self.hand_lo, self.hand_hi)
        stages = STAGES[:STAGES.index(stop_after) + 1]
        if wrist_frame == 'world':
            yaws, anchor = [None], None
        else:
            anchor = (self.wrist_pose(arm_now)[:3, 3] if anchor is None else np.asarray(anchor, float))
            anchor = anchor + np.asarray(anchor_offset, float)
            yaws = [math.radians(y) for y in (yaw_list if yaw_list is not None else range(0, 360, 15))]
        errors, ranked = [], []
        for yaw in yaws:
            try:
                ranked.append((self._approach_ik(grasp, arm_now, anchor, yaw), yaw))
            except PlanError as e:
                errors.append(f'yaw {"-" if yaw is None else round(math.degrees(yaw))}: {e}')
        best = None
        for (motion, q_app), yaw in sorted(ranked, key=lambda r: r[0][0]):
            try:
                best = self._plan_yaw(grasp, arm_now, hand_now, anchor, yaw, stages, release, q_app)
                break
            except PlanError as e:
                errors.append(f'yaw {"-" if yaw is None else round(math.degrees(yaw))}: {e}')
        if best is None:
            raise PlanError('no feasible placement:\n  ' + '\n  '.join(errors))
        best.report['rejected_yaws'] = len(errors)
        return best

    def _stage_poses(self, grasp, anchor, yaw):
        S = np.eye(4) if yaw is None else self.place(grasp, anchor, yaw)
        W = {s: S @ grasp.wrist[s] for s in STAGES}
        approach = W['pregrasp'].copy()
        approach[2, 3] += self.lim.approach_lift
        return S, W, approach

    def _approach_ik(self, grasp, arm_now, anchor, yaw):
        """IK for the approach point from several seeds; returns (max joint motion, q)."""
        _, _, approach = self._stage_poses(grasp, anchor, yaw)
        sol = None
        for seed in [arm_now] + [np.clip(arm_now + d, self.arm_lo, self.arm_hi)
                                 for d in np.random.default_rng(0).normal(0, 0.6, (8, 6))]:
            ok, q, _, _ = solve_ik(self.m, approach, TIP, ARM, seed)
            if ok and (sol is None or np.abs(q - arm_now).max() < np.abs(sol - arm_now).max()):
                sol = q
        if sol is None:
            raise PlanError('approach pose unreachable')
        return float(np.abs(sol - arm_now).max()), sol

    def _plan_yaw(self, grasp, arm_now, hand_now, anchor, yaw, stages, release, sol):
        S, W, approach = self._stage_poses(grasp, anchor, yaw)
        H = {s: np.asarray(grasp.hand[s], float) for s in STAGES}
        segs = [self.joint_segment('1_arm_to_approach', arm_now, sol, hand_now, hand_now),
                self.joint_segment('2_hand_to_pregrasp', sol, sol, hand_now, H['pregrasp'])]
        segs.append(self.cartesian_segment('3_descend_to_pregrasp', sol, W['pregrasp'], H['pregrasp'], H['pregrasp']))
        prev = 'pregrasp'
        for k, stage in enumerate(stages[1:], start=4):
            segs.append(self.cartesian_segment(f'{k}_{prev}_to_{stage}', segs[-1].arm[-1], W[stage], H[prev], H[stage]))
            prev = stage
        if release:
            last = segs[-1].arm[-1]
            segs.append(self.joint_segment('r1_hand_open_to_pregrasp', last, last, H[prev], H['pregrasp']))
            lift = self.wrist_pose(last)
            lift[2, 3] += self.lim.approach_lift
            segs.append(self.cartesian_segment('r2_lift', last, lift, H['pregrasp'], H['pregrasp']))
        # The first move starts wherever the robot is; it may not go lower than its start
        # (minus 5 mm) when the start itself is below the requested clearance.
        start_c = self.clearance(arm_now, hand_now)[0]
        first_floor = min(self.lim.clearance, start_c - 0.005)
        if first_floor < 0.03:
            raise PlanError(f'start pose only {start_c:.3f} m above table; move up first')
        worst = min(self.check(seg, floor=first_floor if seg is segs[0] else None) for seg in segs)
        # Final wrist accuracy against requested poses.
        for seg, stage in zip(segs[2:], stages):
            T = self.wrist_pose(seg.arm[-1])
            pe = np.linalg.norm(T[:3, 3] - W[stage][:3, 3])
            if pe > 1e-3:
                raise PlanError(f'{stage}: final wrist error {pe:.4f} m')
        report = {
            'yaw_deg': None if yaw is None else round(math.degrees(yaw), 1),
            'arm_motion_rad': float(np.abs(sol - arm_now).max()),
            'min_clearance_m': round(float(worst[0]), 4), 'min_clearance_link': worst[1],
            'start_clearance_m': round(float(start_c), 4),
            'first_segment_floor_m': round(float(first_floor), 4),
            'min_left_arm_distance_check_m': self.lim.left_arm_distance,
            'virtual_scene_table_world_z': round(float(S[2, 3]), 4),
            'total_duration_s': round(sum(s.duration for s in segs), 2),
            'segments': [{'name': s.name, 'duration_s': round(s.duration, 2),
                          'arm_max_delta_deg': round(math.degrees(np.abs(s.arm[-1] - s.arm[0]).max()), 2),
                          'hand_max_delta_deg': round(math.degrees(np.abs(s.hand[-1] - s.hand[0]).max()), 2)}
                         for s in segs],
            'wrist_world_xyz': {s: np.round(W[s][:3, 3], 4).tolist() for s in stages},
        }
        return Plan(segs, report['yaw_deg'], S, W, report)


class PlanError(RuntimeError):
    pass
