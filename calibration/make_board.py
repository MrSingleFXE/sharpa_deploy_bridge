#!/usr/bin/env python3
"""Write a printable ChArUco board (PNG at 300 dpi + PDF if Pillow is present) and board.json.

Print at 100 % / "actual size" (no "fit to page"), glue flat on a rigid plate, then MEASURE one
square and one marker with calipers and write the measured sizes:
    python3 calibration/make_board.py --out calib/board --set-measured 0.0298 0.0219
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.board import BoardSpec, render_board  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='calib/board')
    p.add_argument('--squares', type=int, nargs=2, default=[5, 7], metavar=('X', 'Y'))
    p.add_argument('--square-m', type=float, default=0.030)
    p.add_argument('--marker-m', type=float, default=0.022)
    p.add_argument('--dictionary', default='DICT_5X5_100')
    p.add_argument('--set-measured', type=float, nargs=2, metavar=('SQUARE_M', 'MARKER_M'),
                   help='only update board.json with caliper-measured sizes')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.set_measured:
        spec = BoardSpec.load(out / 'board.json')
        spec.square_m, spec.marker_m = a.set_measured
        spec.save(out / 'board.json')
        print(f'updated {out / "board.json"}: square {spec.square_m} m, marker {spec.marker_m} m')
        return
    if a.squares[1] % 2 == 0:
        sys.exit('use an odd number of rows (OpenCV 4.6/4.7 pattern compatibility)')
    spec = BoardSpec(a.squares[0], a.squares[1], a.square_m, a.marker_m, a.dictionary)
    dpi = 300
    px_per_m = dpi / 0.0254
    import cv2
    img = render_board(spec, px_per_m, margin_px=int(0.010 * px_per_m))
    cv2.imwrite(str(out / 'charuco_print_300dpi.png'), img)
    try:
        from PIL import Image
        Image.fromarray(img).save(out / 'charuco_print.pdf', resolution=dpi)
    except ImportError:
        pass
    spec.save(out / 'board.json')
    w_mm, h_mm = (img.shape[1] / px_per_m * 1000, img.shape[0] / px_per_m * 1000)
    print(f'board {spec.squares_x}x{spec.squares_y}, square {spec.square_m*1000:.1f} mm, marker {spec.marker_m*1000:.1f} mm,'
          f' printed size {w_mm:.0f} x {h_mm:.0f} mm (fits A4). Files in {out}/')


if __name__ == '__main__':
    main()
