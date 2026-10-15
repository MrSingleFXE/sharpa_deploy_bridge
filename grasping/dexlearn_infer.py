#!/usr/bin/env python3
"""Generate Sharpa grasps for one object point cloud with the trained DexLearn model (run in the DexLearn env).

Input : object_table.npy from perception/object_cloud.py -- object points in the table frame
        (table z = 0, z up, object centred at x = y = 0). Calibration is NOT needed.
Process (same as DexLearn training/test data):
        shift the object to the training scene centre (default x=0.753, y=0.004 -- median object position of
        the V3/S34 training scenes), draw --clouds random 1024-point subsets (with replacement, like the
        dataset loader), MinkUNet (5 mm voxels) + normalizing flow -> --num samples with log-likelihood,
        shift the wrist positions back.
Output: <out>/grasps_table.npy  BODex-style dict: pregrasp_qpos/grasp_qpos/squeeze_qpos (N,29: wrist xyz,
        wxyz quaternion of right_hand_C_MC, 22 joints), log_prob (N,), joint_names, hand_model_id, frame='table'
        <out>/infer.json  settings, checkpoint identity, timing

    python grasping/dexlearn_infer.py captures/scene01/object_fs/object_table.npy --out captures/scene01/grasps
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_EXP = 'bodex_tabletop_sharpa_right_nflow_tabletop_seed_v3_s34_kp12_alpha1_8gpu_50k_official80_20_20261008'
TRAIN_CENTER = (0.753, 0.004)          # median object xy in the training scenes (5-95 %: x 0.705-0.796, y -0.042-0.045)


def load_model(code_root, model_dir, device):
    """code_root: DexLearn repository (python package `dexlearn`); model_dir: bundle with config.yaml + one *.pth
    (and MANIFEST.json with sha256), or a DexLearn experiment folder (.hydra/config.yaml + ckpts/step_XXXXXX.pth)."""
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(code_root).expanduser()))
    import dexlearn.network.models as models
    model_dir = Path(model_dir).expanduser()
    if (model_dir / 'config.yaml').exists():
        cfg = OmegaConf.load(model_dir / 'config.yaml')
        ckpts = sorted(model_dir.glob('*.pth'))
        if len(ckpts) != 1:
            raise FileNotFoundError(f'expected exactly one .pth in {model_dir}')
        ckpt_path = ckpts[0]
    else:
        cfg = OmegaConf.load(model_dir.parent.parent / '.hydra' / 'config.yaml')   # .../<exp>/ckpts/step_X.pth
        ckpt_path = model_dir
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if ckpt.get('hand_model_id') != cfg.data.hand_model_id or list(ckpt.get('joint_names', [])) != list(cfg.data.joint_names):
        raise ValueError('checkpoint hand identity / joint order does not match the config')
    model = getattr(models, cfg.algo.model.name)(cfg.algo.model)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model, cfg, ckpt, ckpt_path


def pick_device(requested):
    import torch
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        try:
            import MinkowskiEngine as ME
            if ME.is_cuda_available():
                return 'cuda:0'
        except Exception:
            pass
    return 'cpu'


class GraspGenerator:
    """Loads the DexLearn checkpoint once; generate() maps a table-frame object cloud to grasps (table frame)."""

    def __init__(self, code_root, model_dir, device='auto', center=TRAIN_CENTER):
        import torch
        self.torch = torch
        device = pick_device(device)
        self.model, self.cfg, self.ckpt, self.ckpt_path = load_model(code_root, model_dir, device)
        from dexlearn.dataset import get_sparse_tensor
        self.get_sparse_tensor = get_sparse_tensor
        self.device = device
        self.shift = np.array([center[0], center[1], 0.0], np.float32)
        self.n_pts = int(self.cfg.data.num_points)
        self.joint_names = list(self.cfg.data.joint_names)
        self.hand_model_id = self.cfg.data.hand_model_id

    def generate(self, obj, num=200, clouds=4, rng=None):
        """obj (N,3) table frame, centred. Returns pose (num,3,29) table frame and log_prob (num,), best first."""
        torch = self.torch
        rng = rng or np.random.default_rng(0)
        per = int(np.ceil(num / clouds))
        poses, logp = [], []
        with torch.no_grad():
            for _ in range(clouds):
                idx = rng.choice(len(obj), self.n_pts, replace=True)
                pc = torch.as_tensor(obj[idx].astype(np.float32) + self.shift, device=self.device)[None]
                data = self.get_sparse_tensor(pc, self.cfg.algo.model.backbone.voxel_size)
                data['feats'] = data['feats'].float()
                data['grasp_type_id'] = torch.zeros(1, dtype=torch.long, device=self.device)
                robot_pose, log_prob = self.model.sample(data, per)       # (1, per, 3, 29), (1, per)
                poses.append(robot_pose[0].float().cpu().numpy())
                logp.append(log_prob[0].float().cpu().numpy())
        pose = np.concatenate(poses)[:num]
        lp = np.concatenate(logp)[:num]
        pose[..., :3] -= self.shift
        order = np.argsort(-lp)
        return pose[order], lp[order]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('object', help='object_table.npy (table frame, object centred)')
    p.add_argument('--out', required=True)
    p.add_argument('--dexlearn-root', default=os.environ.get('DEXLEARN_ROOT', '~/bodex_sharpa_deploy/DexLearn'),
                   help='DexLearn code repository (env DEXLEARN_ROOT)')
    p.add_argument('--model', default=os.environ.get('DEXLEARN_MODEL', '~/bodex_sharpa_deploy/models/dexlearn_sharpa_v3_s34_50k'),
                   help='model bundle folder (config.yaml + .pth) or an experiment ckpt path (env DEXLEARN_MODEL)')
    p.add_argument('--num', type=int, default=200, help='total grasps to sample')
    p.add_argument('--clouds', type=int, default=4, help='random 1024-point subsets (samples split evenly)')
    p.add_argument('--center', type=float, nargs=2, default=TRAIN_CENTER)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto', help='auto (GPU if MinkowskiEngine has CUDA, else cpu), cpu, cuda:0')
    a = p.parse_args(argv)
    import torch
    torch.manual_seed(a.seed)
    obj = np.load(a.object).astype(np.float32)
    if obj.ndim != 2 or obj.shape[1] != 3 or len(obj) < 100:
        sys.exit(f'bad object cloud {obj.shape}')
    if obj[:, 2].min() < -0.01 or abs(obj[:, 0].mean()) > 0.02 or abs(obj[:, 1].mean()) > 0.02:
        print('WARNING: cloud does not look like a table-frame, centred object (z >= 0, xy mean ~ 0)')
    gen = GraspGenerator(a.dexlearn_root, a.model, a.device, a.center)
    cfg, ckpt, ckpt_path = gen.cfg, gen.ckpt, gen.ckpt_path
    t0 = time.time()
    pose, lp = gen.generate(obj, a.num, a.clouds, np.random.default_rng(a.seed))
    dt = time.time() - t0
    n_pts = gen.n_pts
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {'pregrasp_qpos': pose[:, 0], 'grasp_qpos': pose[:, 1], 'squeeze_qpos': pose[:, 2], 'log_prob': lp,
           'joint_names': np.array(list(cfg.data.joint_names)), 'hand_model_id': cfg.data.hand_model_id,
           'frame': 'table', 'object': str(Path(a.object).resolve())}
    np.save(out / 'grasps_table.npy', res, allow_pickle=True)
    info = {'checkpoint': str(ckpt_path), 'iter': int(ckpt.get('iter', -1)), 'hand_model_id': cfg.data.hand_model_id,
            'num': int(len(pose)), 'clouds': a.clouds, 'points_per_cloud': n_pts, 'train_center_xy': list(a.center),
            'object_points': int(len(obj)), 'runtime_s': round(dt, 2), 'device': gen.device,
            'log_prob': {'max': float(lp.max()), 'median': float(np.median(lp)), 'min': float(lp.min())},
            'wrist_height_above_table_m': {'median': float(np.median(pose[:, 1, 2])), 'min': float(pose[:, 1, 2].min())}}
    (out / 'infer.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f'saved {out / "grasps_table.npy"} (sorted by log_prob, highest first)')


if __name__ == '__main__':
    main()
