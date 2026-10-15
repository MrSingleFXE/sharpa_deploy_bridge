"""Calibration targets: ChArUco board, single ArUco / AprilTag marker, or QR code (OpenCV 4.6 / >= 4.7 / 5.x).

kind = 'charuco' : squares_x, squares_y, square_m, marker_m, dictionary
kind = 'marker'  : one ArUco/AprilTag marker; dictionary (e.g. DICT_APRILTAG_36h11), marker_id (None = any single
                   marker), marker_m = MEASURED side of the black square (outer edge of the black border)
kind = 'qr'      : QR code; marker_m = MEASURED side of the QR code's black square (finder-pattern outer edges)
Frame of single targets: origin at the centre, x to the right, y down (as seen when the code is upright), z into it.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

NEW_API = hasattr(cv2.aruco, 'CharucoDetector')


@dataclass
class BoardSpec:
    kind: str = 'charuco'         # charuco | marker | qr
    squares_x: int = 5            # columns
    squares_y: int = 7            # rows (odd: avoids the OpenCV 4.6 / 4.7 legacy-pattern difference)
    square_m: float = 0.030       # MEASURED printed square side, metres
    marker_m: float = 0.022       # MEASURED printed marker side, metres
    dictionary: str = 'DICT_5X5_100'
    marker_id: int = None         # kind=marker: which id to use (None = the only marker seen)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d.setdefault('kind', 'charuco')
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def corners_obj(self):
        """3D corners (4,3) of a single square target in its frame, order TL, TR, BR, BL."""
        h = self.marker_m / 2
        return np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)


def make_board(spec: BoardSpec):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    if NEW_API:
        return cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_m, spec.marker_m, d), d
    return cv2.aruco.CharucoBoard_create(spec.squares_x, spec.squares_y, spec.square_m, spec.marker_m, d), d


def render_board(spec: BoardSpec, px_per_m, margin_px=0):
    board, _ = make_board(spec)
    size = (int(round(spec.squares_x * spec.square_m * px_per_m)) + 2 * margin_px,
            int(round(spec.squares_y * spec.square_m * px_per_m)) + 2 * margin_px)
    if NEW_API:
        return board.generateImage(size, marginSize=margin_px)
    return board.draw(size, marginSize=margin_px)


@dataclass
class BoardPose:
    T_cam_board: np.ndarray       # 4x4, board frame in camera frame (metres)
    corners: int
    reproj_px: float              # RMS reprojection error


def _square_pose(img_pts, gray, K, dist, spec):
    """PnP for a single square target from its 4 image corners (TL, TR, BR, BL)."""
    img = cv2.cornerSubPix(gray, img_pts.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)).reshape(-1, 2)
    obj = spec.corners_obj
    # Planar PnP has two candidate poses; keep the one in front of the camera with the lowest reprojection error.
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rv, tv in zip(rvecs, tvecs):
        if float(tv.ravel()[2]) <= 0:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj, img.astype(np.float64), K, dist, rv.copy(), tv.copy())
        pr, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        e = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, axis=1))))
        if float(tv.ravel()[2]) > 0 and (best is None or e < best[0]):
            best = (e, rv, tv)
    if best is None:
        return None
    _, rvec, tvec = best
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return BoardPose(T, 4, err)


def detect_markers(gray, dictionary):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
    return ([] if ids is None else [c.reshape(4, 2) for c in corners]), ([] if ids is None else ids.ravel().tolist())


def detect_qr(gray):
    ok, pts = cv2.QRCodeDetector().detect(gray)
    return pts.reshape(4, 2) if ok and pts is not None else None


def detect_pose(image, K, dist, spec: BoardSpec, min_corners=8):
    """Return BoardPose or None. image: BGR or grayscale uint8."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if spec.kind == 'marker':
        corners, ids = detect_markers(gray, spec.dictionary)
        if spec.marker_id is not None:
            corners = [c for c, i in zip(corners, ids) if i == spec.marker_id]
        return _square_pose(corners[0], gray, K, dist, spec) if len(corners) == 1 else None
    if spec.kind == 'qr':
        pts = detect_qr(gray)
        return None if pts is None else _square_pose(pts, gray, K, dist, spec)
    board, d = make_board(spec)
    if NEW_API:
        detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            return None
        obj, img = board.matchImagePoints(ch_corners, ch_ids)
    else:
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, d)
        if m_ids is None or len(m_ids) == 0:
            return None
        n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, gray, board)
        if ch_ids is None or n < min_corners:
            return None
        all_obj = board.chessboardCorners
        obj = all_obj[ch_ids.flatten()].reshape(-1, 1, 3)
        img = ch_corners.reshape(-1, 1, 2)
    obj, img = obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)
    if len(obj) < min_corners:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return BoardPose(T, len(obj), err)


def render_single(spec: BoardSpec, px_per_m):
    """Texture of a single marker / QR target with a white quiet zone; returns (image, metres per pixel offset)."""
    side = int(round(spec.marker_m * px_per_m))
    if spec.kind == 'marker':
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
        gen = cv2.aruco.generateImageMarker if hasattr(cv2.aruco, 'generateImageMarker') else cv2.aruco.drawMarker
        core = gen(d, spec.marker_id or 0, side)
    else:
        q = cv2.QRCodeEncoder.create().encode('sharpa-calibration')             # includes a quiet zone
        ys, xs = np.nonzero(q == 0)
        q = q[ys.min():ys.max() + 1, xs.min():xs.max() + 1]                      # crop to the black square
        core = cv2.resize(q, (side, side), interpolation=cv2.INTER_NEAREST)
    pad = side // 4
    img = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    img[pad:pad + side, pad:pad + side] = core
    return img, pad


def synth_view(spec: BoardSpec, T_cam_board, K, size=(1280, 720), px_per_m=4000):
    """Render what a pinhole camera (no distortion) sees of the target at T_cam_board (for tests/sim)."""
    if spec.kind in ('marker', 'qr'):
        tex, pad = render_single(spec, px_per_m)
        h, w = tex.shape[:2]
        corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        # texture pixel -> target frame (origin at the centre of the black square)
        corners_m = np.c_[(corners_px - pad) / px_per_m - spec.marker_m / 2, np.zeros(4)]
        return _warp(tex, corners_px, corners_m, T_cam_board, K, size)
    margin = int(spec.square_m * px_per_m * 0.5)
    tex = render_board(spec, px_per_m, margin_px=margin)
    h, w = tex.shape[:2]
    # texture pixel (u, v) -> board metric point (x, y, 0); OpenCV board origin is the first square corner.
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_m = np.c_[(corners_px - margin) / px_per_m, np.zeros(4)]
    return _warp(tex, corners_px, corners_m, T_cam_board, K, size)


def _warp(tex, corners_px, corners_m, T_cam_board, K, size):
    pts_cam = (T_cam_board[:3, :3] @ corners_m.T).T + T_cam_board[:3, 3]
    if np.any(pts_cam[:, 2] <= 0.05):
        return np.full((size[1], size[0]), 128, np.uint8)
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), uv.astype(np.float32))
    out = cv2.warpPerspective(tex, H, size, borderValue=128)
    return out
