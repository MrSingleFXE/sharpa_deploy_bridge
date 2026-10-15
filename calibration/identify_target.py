#!/usr/bin/env python3
"""Find out what calibration target you printed and write calib/board/board.json for it (no robot needed).

Hold the printed target in front of the camera (0.4-0.8 m, flat, well lit), then:
    python3 calibration/identify_target.py --size-m 0.120          # 0.120 = MEASURED side of the black square, metres
or on a saved image:   --image photo.png --size-m 0.120

Tries: ChArUco (our board), single ArUco markers (all predefined dictionaries), AprilTag families, QR code.
Prints what was found and writes board.json (kind charuco / marker / qr) that the calibration scripts use.
For a single marker / QR code measure the OUTER edge of the black square (not the white margin).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, detect_markers, detect_pose, detect_qr  # noqa: E402

APRIL = ['DICT_APRILTAG_36h11', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_25h9', 'DICT_APRILTAG_16h5']
ARUCO = [f'DICT_{b}X{b}_{n}' for b in (4, 5, 6, 7) for n in (50, 100, 250, 1000)] + ['DICT_ARUCO_ORIGINAL']


def grab(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f'cannot read {args.image}')
        return img, None
    from calibration.camera import RealSenseCamera
    cam = RealSenseCamera(serial=args.serial, depth=False, model=args.model)
    img, _ = cam.grab()
    cam.close()
    return img, cam.intr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--size-m', type=float, help='measured side of the black square (marker / QR), metres')
    p.add_argument('--image')
    p.add_argument('--model', default='D435I')
    p.add_argument('--serial')
    p.add_argument('--out', default='calib/board/board.json')
    a = p.parse_args()
    img, intr = grab(a)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(a.out).parent / 'identify_input.png'), img)
    found = None
    if intr is not None and detect_pose(img, intr.K, intr.dist, BoardSpec()) is not None:
        found = BoardSpec()
        print('found: our ChArUco board (5x7). Use make_board.py --set-measured for its sizes.')
    if found is None:
        for d in APRIL + ARUCO:
            corners, ids = detect_markers(gray, d)
            if ids:
                side_px = float(np.mean([np.linalg.norm(c[0] - c[1]) for c in corners]))
                print(f'found: {len(ids)} marker(s) of {d}, ids {ids}, about {side_px:.0f} px wide')
                if len(ids) > 1:
                    print('  several markers in view: the first id is used; keep only one in view while calibrating')
                found = BoardSpec(kind='marker', dictionary=d, marker_id=int(ids[0]), marker_m=a.size_m or 0.0)
                break
    if found is None:
        pts = detect_qr(gray)
        if pts is not None:
            side_px = float(np.linalg.norm(pts[0] - pts[1]))
            print(f'found: QR code, about {side_px:.0f} px wide')
            found = BoardSpec(kind='qr', marker_m=a.size_m or 0.0)
    if found is None:
        sys.exit('nothing detected (see ' + str(Path(a.out).parent / 'identify_input.png') +
                 '): improve lighting / move closer / avoid glare; a plain QR code must be fully in view')
    if found.kind != 'charuco':
        if not a.size_m:
            sys.exit('run again with --size-m <measured side of the black square in metres>')
        found.save(a.out)
        print(f'saved {a.out}: {found}')
        if found.kind == 'qr':
            print('note: a QR code gives only 4 corners and its corner localisation is coarser than ArUco/AprilTag; '
                  'use a large print (>= 12 cm), 20-30 samples, and check the solve verdict')
        else:
            print('note: a single marker gives 4 corners per image (ChArUco gives 24); use >= 10 cm, 20-30 samples')


if __name__ == '__main__':
    main()
