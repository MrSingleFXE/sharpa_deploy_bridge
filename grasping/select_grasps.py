#!/usr/bin/env python3
"""Filter, rank and visualise DexLearn grasps for one object (numpy only; no calibration, no robot).

Input : grasps_table.npy (grasping/dexlearn_infer.py) + object_table.npy (perception/object_cloud.py),
        both in the table frame (table z = 0, z up).
Checks per candidate (hand = surface points of the Sharpa collision meshes):
  joints     all three poses inside the URDF joint limits (DexLearn evaluation counts violations as failures)
  table      no hand point below the table (pregrasp / grasp / squeeze) beyond --table-tol
  pregrasp   the open hand does not already penetrate the object: object points closer than --contact-mm to the
             pregrasp hand surface must be < --max-penetration of the object
Ranking: model log-likelihood among the candidates that pass (the same score the Top-k results used).
Output: <grasps dir>/selected.npy (BODex-style dict, table frame, best first), selection.json, selected_views.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.hand_model import HandModel, qpos_to_T  # noqa: E402

STAGES = ('pregrasp_qpos', 'grasp_qpos', 'squeeze_qpos')


def min_dist(A, B, chunk=4000):
    """For every point of A the distance to the nearest point of B (brute force, chunked)."""
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        d = ((A[i:i + chunk, None, :] - B[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out


def evaluate(hand, g, obj, names, a):
    rows = []
    obj_s = obj[np.random.default_rng(0).choice(len(obj), min(len(obj), 1500), replace=False)]
    for i in range(len(g['grasp_qpos'])):
        r = {'index': i, 'log_prob': float(g['log_prob'][i])}
        r['joint_violation_deg'] = round(float(np.degrees(max(
            hand.limit_violation(g[s][i][7:], names) for s in STAGES))), 2)
        zmin, pts = [], {}
        for s in STAGES:
            P = hand.points(qpos_to_T(g[s][i]), g[s][i][7:], names)
            pts[s] = P
            zmin.append(float(P[:, 2].min()))
        r['hand_min_z_mm'] = round(1000 * min(zmin), 1)
        d = min_dist(obj_s, pts['pregrasp_qpos'][::2])
        r['pregrasp_penetration'] = round(float((d < a.contact_mm / 1000).mean()), 3)
        r['grasp_contact_points'] = int((min_dist(obj_s, pts['grasp_qpos'][::2]) < 0.006).sum())
        reasons = []
        if r['joint_violation_deg'] > 0.0:
            reasons.append('joint limit')
        if min(zmin) < -a.table_tol / 1000:
            reasons.append('table')
        if r['pregrasp_penetration'] > a.max_penetration:
            reasons.append('pregrasp penetration')
        r['pass'] = not reasons
        r['reject'] = reasons
        rows.append(r)
    return rows


def plot(path, hand, g, obj, names, idxs, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(idxs)
    fig, ax = plt.subplots(n, 3, figsize=(12, 3.6 * n), squeeze=False)
    for row, i in enumerate(idxs):
        P = hand.points(qpos_to_T(g['grasp_qpos'][i]), g['grasp_qpos'][i][7:], names)[::3]
        for col, (a, b, name) in enumerate(((0, 1, 'top x-y'), (0, 2, 'side x-z'), (1, 2, 'front y-z'))):
            x = ax[row, col]
            x.scatter(obj[:, a], obj[:, b], s=1, c='tab:gray')
            x.scatter(P[:, a], P[:, b], s=1, c='tab:red', alpha=0.5)
            if b == 2:
                x.axhline(0, color='k', lw=0.8)
            x.set_aspect('equal')
            x.set_title(f'#{i} logp {g["log_prob"][i]:.1f}  {name}', fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('grasps', help='grasps_table.npy')
    p.add_argument('object', help='object_table.npy')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--table-tol', type=float, default=2.0, help='mm below the table still accepted (depth noise)')
    p.add_argument('--contact-mm', type=float, default=3.0)
    p.add_argument('--max-penetration', type=float, default=0.02)
    p.add_argument('--hand-urdf')
    p.add_argument('--plot', type=int, default=6, help='number of selected grasps to draw')
    a = p.parse_args(argv)
    g = np.load(a.grasps, allow_pickle=True).item()
    obj = np.load(a.object)
    names = [str(n) for n in g['joint_names']]
    hand = HandModel(a.hand_urdf)
    rows = evaluate(hand, g, obj, names, a)
    keep = [r['index'] for r in sorted(rows, key=lambda r: -r['log_prob']) if r['pass']][:a.top]
    out = Path(a.grasps).parent
    sel = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(rows) else v) for k, v in g.items()}
    sel['source_index'] = np.array(keep)
    np.save(out / 'selected.npy', sel, allow_pickle=True)
    reasons = {}
    for r in rows:
        for x in r['reject']:
            reasons[x] = reasons.get(x, 0) + 1
    report = {'candidates': len(rows), 'passed': sum(r['pass'] for r in rows), 'selected': keep,
              'rejected_by': reasons, 'per_candidate': rows}
    (out / 'selection.json').write_text(json.dumps(report, indent=2))
    if keep:
        plot(out / 'selected_views.png', hand, g, obj, names, keep[:a.plot],
             f'top {min(a.plot, len(keep))} selected grasps (grasp pose), table frame [m]')
    print(json.dumps({k: report[k] for k in ('candidates', 'passed', 'selected', 'rejected_by')}, indent=2))
    print(f'saved {out / "selected.npy"}, selection.json' + (', selected_views.png' if keep else ' (none passed)'))


if __name__ == '__main__':
    main()
