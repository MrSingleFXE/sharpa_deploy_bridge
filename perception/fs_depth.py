#!/usr/bin/env python3
"""FoundationStereo depth for a capture folder (run inside the FoundationStereo torch environment).

Input : <capture>/ir_left.png, ir_right.png, meta.json   (from perception/capture.py)
Output: <capture>/depth_fs_ir.npy     depth (m) on the IR-left pixel grid (0 = invalid)
        <capture>/depth_fs_color.npy  the same depth registered into the color camera (for masks/point clouds)
        <capture>/fs_vis.png          IR-left | FoundationStereo disparity | RealSense depth (for a quick look)
        <capture>/fs_meta.json        checkpoint, settings, runtime

    export FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo
    python perception/fs_depth.py captures/scene01 [captures/scene02 ...]

Fully offline at run time: DINOv2 code is loaded from the copy bundled in FoundationStereo and the timm
backbone is built without its ImageNet download (all weights come from the FoundationStereo checkpoint);
its weights come from the FoundationStereo checkpoint. xformers is optional (XFORMERS_DISABLED=1 is set
automatically when it is not installed). Unused top-level imports of FoundationStereo (open3d, trimesh, pandas,
joblib) are replaced by empty stubs, so they need not be installed.
"""

import argparse
import json
import os
import sys
import time
import warnings

# Known, harmless messages from FoundationStereo's own code (xformers is intentionally not installed; old autocast API).
warnings.filterwarnings('ignore', message='xFormers is')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*autocast.*')
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.geometry import disparity_to_depth, register_depth  # noqa: E402


def _stub_unused_modules():
    """FoundationStereo's Utils.py imports open3d/trimesh/pandas/joblib at top level, but inference never uses them.
    open3d is always stubbed (importing it together with torch segfaults on some platforms, e.g. macOS);
    the others are stubbed only when not installed, so a minimal environment is enough."""
    import importlib.machinery
    import importlib.util
    import types
    for name in ('open3d', 'trimesh', 'pandas', 'joblib'):
        if name in sys.modules:
            continue
        if name == 'open3d' or importlib.util.find_spec(name) is None:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None)   # torch inspects module specs
            sys.modules[name] = mod


class FoundationStereoRunner:
    def __init__(self, fs_root, ckpt=None, valid_iters=32, device='cuda'):
        try:
            import xformers  # noqa: F401
        except ImportError:
            os.environ.setdefault('XFORMERS_DISABLED', '1')
        import torch
        from omegaconf import OmegaConf
        self.torch = torch
        fs_root = Path(fs_root).expanduser().resolve()
        sys.path.insert(0, str(fs_root))
        ckpt = Path(ckpt) if ckpt else fs_root / 'pretrained_models' / '23-51-11' / 'model_best_bp2.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f'checkpoint not found: {ckpt} (download the 23-51-11 folder, see README)')
        # Route torch.hub's DINOv2 lookup to the bundled local copy (offline, same code version).
        real_load = torch.hub.load

        def hub_load(repo, model, *args, **kwargs):
            if repo == 'facebookresearch/dinov2':
                kwargs.pop('source', None)
                return real_load(str(fs_root / 'dinov2'), model, *args, source='local', **kwargs)
            return real_load(repo, model, *args, **kwargs)

        torch.hub.load = hub_load
        # FoundationStereo builds a timm EdgeNeXt with pretrained=True (ImageNet download from huggingface.co).
        # All of its parameters are overwritten by the FoundationStereo checkpoint below (strict load),
        # so build it without the download.
        import timm
        real_create = timm.create_model

        def create_model(name, *args, **kwargs):
            kwargs['pretrained'] = False
            return real_create(name, *args, **kwargs)

        timm.create_model = create_model
        _stub_unused_modules()
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        self.InputPadder = InputPadder
        cfg = OmegaConf.load(ckpt.parent / 'cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        cfg['valid_iters'] = valid_iters
        self.args = OmegaConf.create(cfg)
        self.model = FoundationStereo(self.args)
        state = torch.load(str(ckpt), map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'], strict=True)   # fails loudly if any weight is missing
        self.model.to(device).eval()
        self.device, self.ckpt, self.valid_iters = device, str(ckpt), valid_iters

    def disparity(self, left, right, hiera=False):
        """left/right: HxW (gray) or HxWx3 uint8, rectified. Returns HxW float32 disparity (px)."""
        torch = self.torch
        if left.ndim == 2:
            left, right = np.repeat(left[..., None], 3, 2), np.repeat(right[..., None], 3, 2)
        H, W = left.shape[:2]
        t0 = torch.as_tensor(left).to(self.device).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(right).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        with torch.no_grad(), torch.autocast('cuda', enabled=self.device.startswith('cuda')):
            if hiera:
                disp = self.model.run_hierachical(t0, t1, iters=self.valid_iters, test_mode=True, small_ratio=0.5)
            else:
                disp = self.model.forward(t0, t1, iters=self.valid_iters, test_mode=True)
        return padder.unpad(disp.float()).cpu().numpy().reshape(H, W).astype(np.float32)


def colorize(depth, lo=None, hi=None):
    import cv2
    valid = depth > 0
    if not valid.any():
        return np.zeros(depth.shape + (3,), np.uint8)
    lo = np.percentile(depth[valid], 2) if lo is None else lo
    hi = np.percentile(depth[valid], 98) if hi is None else hi
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def process(runner, cap, scale=1.0, hiera='auto'):
    import cv2
    cap = Path(cap)
    meta = json.loads((cap / 'meta.json').read_text())
    left = cv2.imread(str(cap / 'ir_left.png'), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(cap / 'ir_right.png'), cv2.IMREAD_GRAYSCALE)
    K_ir = np.array(meta['K_ir'], float)
    if scale != 1.0:
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        K_ir[:2] *= scale
    use_hiera = (left.shape[1] > 1000) if hiera == 'auto' else bool(int(hiera))
    cuda = runner.device.startswith('cuda')
    if cuda:
        runner.torch.cuda.synchronize()
    t = time.time()
    disp = runner.disparity(left, right, hiera=use_hiera)
    if cuda:
        runner.torch.cuda.synchronize()
    dt = time.time() - t
    depth_ir = disparity_to_depth(disp, K_ir[0, 0], meta['baseline_m'])
    w, h = meta['size']
    depth_col = register_depth(depth_ir, K_ir, meta['T_color_from_ir_left'], meta['K_color'], (w, h))
    np.save(cap / 'depth_fs_ir.npy', depth_ir)
    np.save(cap / 'depth_fs_color.npy', depth_col)
    raw = np.load(cap / 'depth_raw_ir.npy') if (cap / 'depth_raw_ir.npy').exists() else np.zeros_like(depth_ir)
    if scale != 1.0 and raw.any():
        raw = cv2.resize(raw, (depth_ir.shape[1], depth_ir.shape[0]), interpolation=cv2.INTER_NEAREST)
    if raw.shape != depth_ir.shape:
        raw = cv2.resize(raw, (depth_ir.shape[1], depth_ir.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = depth_ir[depth_ir > 0]
    lo, hi = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size else (0, 1)
    vis = np.concatenate([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), colorize(depth_ir, lo, hi), colorize(raw, lo, hi)], 1)
    cv2.imwrite(str(cap / 'fs_vis.png'), vis)
    info = {'checkpoint': runner.ckpt, 'valid_iters': runner.valid_iters, 'scale': scale, 'hierarchical': use_hiera,
            'runtime_s': round(dt, 3), 'device': runner.torch.cuda.get_device_name(0) if cuda else runner.device,
            'valid_fraction_ir': round(float((depth_ir > 0).mean()), 4),
            'valid_fraction_color': round(float((depth_col > 0).mean()), 4)}
    (cap / 'fs_meta.json').write_text(json.dumps(info, indent=2))
    print(f'{cap}: {dt:.2f} s, valid {info["valid_fraction_ir"] * 100:.1f}% (IR grid), '
          f'{info["valid_fraction_color"] * 100:.1f}% (color grid); fs_vis.png = IR | FoundationStereo | RealSense')
    return info


def make_demo_capture(fs_root, out):
    """FoundationStereo's own demo pair (assets/left.png, right.png, K.txt) as a capture folder, for install tests."""
    import cv2
    fs_root, out = Path(fs_root).expanduser(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = (fs_root / 'assets' / 'K.txt').read_text().split('\n')
    K = np.array(list(map(float, lines[0].split()))).reshape(3, 3)
    left, right = cv2.imread(str(fs_root / 'assets' / 'left.png')), cv2.imread(str(fs_root / 'assets' / 'right.png'))
    cv2.imwrite(str(out / 'ir_left.png'), left)
    cv2.imwrite(str(out / 'ir_right.png'), right)
    cv2.imwrite(str(out / 'color.png'), left)
    meta = {'device': {'name': 'FoundationStereo demo'}, 'size': [left.shape[1], left.shape[0]],
            'K_color': K.tolist(), 'K_ir': K.tolist(), 'baseline_m': float(lines[1]),
            'T_color_from_ir_left': np.eye(4).tolist()}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('captures', nargs='*')
    p.add_argument('--demo', metavar='OUT', help='install self-test: run on FoundationStereo assets, write to OUT')
    p.add_argument('--fs-root', default=os.environ.get('FS_ROOT', '~/bodex_sharpa_deploy/FoundationStereo'))
    p.add_argument('--ckpt', help='default: <fs-root>/pretrained_models/23-51-11/model_best_bp2.pth')
    p.add_argument('--valid-iters', type=int, default=32)
    p.add_argument('--device', default='cuda', help='cuda (default) or cpu (slow; for tests only)')
    p.add_argument('--scale', type=float, default=1.0, help='<=1; 0.5 is ~4x faster with half resolution')
    p.add_argument('--hiera', default='auto', help="hierarchical inference: auto (on if width > 1000), 0, 1")
    a = p.parse_args(argv)
    if a.demo:
        a.captures = [str(make_demo_capture(a.fs_root, a.demo))] + list(a.captures)
    if not a.captures:
        p.error('give capture folders or --demo OUT')
    runner = FoundationStereoRunner(a.fs_root, a.ckpt, a.valid_iters, a.device)
    for c in a.captures:
        process(runner, c, a.scale, a.hiera)


if __name__ == '__main__':
    main()
