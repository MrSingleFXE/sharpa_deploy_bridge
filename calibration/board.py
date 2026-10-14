"""ChArUco board definition, rendering and pose detection (OpenCV 4.6 and >= 4.7 APIs)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

NEW_API = hasattr(cv2.aruco, 'CharucoDetector')


@dataclass
class BoardSpec:
    squares_x: int = 5            # columns
    squares_y: int = 7            # rows (odd: avoids the OpenCV 4.6 / 4.7 legacy-pattern difference)
    square_m: float = 0.030       # MEASURED printed square side, metres
    marker_m: float = 0.022       # MEASURED printed marker side, metres
    dictionary: str = 'DICT_5X5_100'

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


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


def detect_pose(image, K, dist, spec: BoardSpec, min_corners=8):
    """Return BoardPose or None. image: BGR or grayscale uint8."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def synth_view(spec: BoardSpec, T_cam_board, K, size=(1280, 720), px_per_m=4000):
    """Render what a pinhole camera (no distortion) sees of the board at T_cam_board (for tests/sim)."""
    margin = int(spec.square_m * px_per_m * 0.5)
    tex = render_board(spec, px_per_m, margin_px=margin)
    h, w = tex.shape[:2]
    # texture pixel (u, v) -> board metric point (x, y, 0); OpenCV board origin is the first square corner.
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_m = np.c_[(corners_px - margin) / px_per_m, np.zeros(4)]
    pts_cam = (T_cam_board[:3, :3] @ corners_m.T).T + T_cam_board[:3, 3]
    if np.any(pts_cam[:, 2] <= 0.05):
        return np.full((size[1], size[0]), 128, np.uint8)
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), uv.astype(np.float32))
    out = cv2.warpPerspective(tex, H, size, borderValue=128)
    return out
