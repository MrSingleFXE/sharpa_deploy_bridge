"""ChArUco board: definition, printing, detection (OpenCV 4.6 and >= 4.7 APIs) and the industrial_calibration
target-finder config.

The same board is used by ROS-Industrial industrial_calibration (CharucoGridTargetFinder: rows = squares_y,
cols = squares_x, chessboard_dim = square_m, aruco_marker_dim = marker_m, dictionary = OpenCV enum value).
An odd number of rows avoids the OpenCV 4.6 / >= 4.7 legacy-pattern difference, so a board printed with any
OpenCV version is found by industrial_calibration (built against the system OpenCV 4.6 on Ubuntu 24.04).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

NEW_API = hasattr(cv2.aruco, 'CharucoDetector')


@dataclass
class BoardSpec:
    squares_x: int = 5            # columns  (industrial_calibration: cols)
    squares_y: int = 7            # rows     (industrial_calibration: rows; keep odd)
    square_m: float = 0.030       # MEASURED printed square side, metres   (chessboard_dim)
    marker_m: float = 0.022       # MEASURED printed marker side, metres   (aruco_marker_dim)
    dictionary: str = 'DICT_5X5_100'

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def ic_target_finder(self):
        """industrial_calibration target_finder config for this board."""
        return {'type': 'CharucoGridTargetFinder', 'rows': self.squares_y, 'cols': self.squares_x,
                'chessboard_dim': float(self.square_m), 'aruco_marker_dim': float(self.marker_m),
                'dictionary': int(getattr(cv2.aruco, self.dictionary))}


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
    """Return BoardPose or None. image: BGR or grayscale uint8. (Used for camera checks and initial guesses;
    the calibration itself is done by industrial_calibration.)"""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board, d = make_board(spec)
    if NEW_API:
        ch_corners, ch_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
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
        obj = board.chessboardCorners[ch_ids.flatten()].reshape(-1, 1, 3)
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
    """Render what a pinhole camera (no distortion) sees of the board at T_cam_board (tests only).
    Board frame as in OpenCV >= 4.7: origin at the outer corner next to marker 0, x along columns, y along rows."""
    margin = int(spec.square_m * px_per_m * 0.5)
    tex = render_board(spec, px_per_m, margin_px=margin)
    h, w = tex.shape[:2]
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_m = np.c_[(corners_px - margin) / px_per_m, np.zeros(4)]
    pts_cam = (T_cam_board[:3, :3] @ corners_m.T).T + T_cam_board[:3, 3]
    if np.any(pts_cam[:, 2] <= 0.05):
        return np.full((size[1], size[0]), 128, np.uint8)
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), uv.astype(np.float32))
    return cv2.warpPerspective(tex, H, size, borderValue=128)
