"""
Perception module for Team L2.

Reads camera frames over TCP and returns stable 6x6 board state based on ArUco IDs.
"""

import logging
import socket
import struct
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    ARUCO_CORNER_IDS,
    ARUCO_PIECE_IDS,
    BOARD_SIZE,
    CAMERA_CONNECT_TIMEOUT,
    CAMERA_IP,
    CAMERA_MATRIX,
    CAMERA_PORT,
    CAMERA_RECV_TIMEOUT,
    CELL_CENTERS_X,
    CELL_CENTERS_Y,
    CELL_THRESHOLD_MM,
    CORNER_WORLD_COORDS,
    DIST_COEFFS,
    MAX_RETRIES,
    MOVE_TIMEOUT,
    RETRY_DELAY_BASE,
    STABILITY_FRAMES,
    STABILITY_TIMEOUT,
    VERIFY_TIMEOUT,
)

log = logging.getLogger("l2.perception")


class PerceptionSystem:
    def __init__(self) -> None:
        self._sock: Optional[socket.socket] = None
        self._recv_buf = b""
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_board: Optional[np.ndarray] = None

        self.camera_matrix = CAMERA_MATRIX.copy()
        self.dist_coeffs = DIST_COEFFS.copy()
        self.corner_world = dict(CORNER_WORLD_COORDS)
        self.H_matrix: Optional[np.ndarray] = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = self._detector_params()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.frame_counter = 0
        self.frame_failures = 0
        self.homography_failures = 0
        self.last_marker_count = 0
        log.info("PerceptionSystem initialized for %s:%s", CAMERA_IP, CAMERA_PORT)

    @staticmethod
    def _detector_params() -> cv2.aruco.DetectorParameters:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 8
        params.adaptiveThreshConstant = 7
        return params

    def connect_camera(self) -> bool:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self._sock is not None:
                    self._sock.close()
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(CAMERA_CONNECT_TIMEOUT)
                self._sock.connect((CAMERA_IP, CAMERA_PORT))
                self._sock.settimeout(CAMERA_RECV_TIMEOUT)
                self._recv_buf = b""
                log.info("camera connected to %s:%s", CAMERA_IP, CAMERA_PORT)
                return True
            except OSError as exc:
                log.warning("camera connect attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
                time.sleep(RETRY_DELAY_BASE * attempt)
        self._sock = None
        log.error("camera connect failed after %d attempts", MAX_RETRIES)
        return False

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        log.info(
            "perception closed | frames=%d frame_failures=%d homography_failures=%d",
            self.frame_counter,
            self.frame_failures,
            self.homography_failures,
        )

    def _recv_exact(self, nbytes: int) -> Optional[bytes]:
        if self._sock is None:
            return None
        while len(self._recv_buf) < nbytes:
            try:
                chunk = self._sock.recv(max(4096, nbytes - len(self._recv_buf)))
            except OSError:
                return None
            if not chunk:
                return None
            self._recv_buf += chunk
        out = self._recv_buf[:nbytes]
        self._recv_buf = self._recv_buf[nbytes:]
        return out

    def recv_frame(self) -> Optional[np.ndarray]:
        header = self._recv_exact(8)
        if header is None:
            self.frame_failures += 1
            return None
        size = struct.unpack("Q", header)[0]
        if size <= 0 or size > 12_000_000:
            self.frame_failures += 1
            log.debug("invalid frame size: %s", size)
            return None
        payload = self._recv_exact(size)
        if payload is None:
            self.frame_failures += 1
            return None
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.frame_failures += 1
            return None
        self.frame_counter += 1
        return frame

    def detect_markers(self, frame: np.ndarray):
        und = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        best_corners = None
        best_ids = None
        best_count = 0

        for variant in (
            gray,
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        ):
            corners, ids, _ = self.detector.detectMarkers(variant)
            count = 0 if ids is None else len(ids)
            if count > best_count:
                best_count = count
                best_corners = corners
                best_ids = ids

            self.last_marker_count = best_count
            log.debug("detected markers: %d", best_count)

        return best_corners, best_ids

    def compute_homography(self, corners, ids) -> bool:
        if ids is None:
            self.homography_failures += 1
            log.debug("homography skipped: no ids")
            return False

        pixel_pts = []
        world_pts = []
        for idx, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id in ARUCO_CORNER_IDS:
                mc = corners[idx][0]
                pixel_pts.append([float(np.mean(mc[:, 0])), float(np.mean(mc[:, 1]))])
                world_pts.append(list(self.corner_world[marker_id]))

        if len(pixel_pts) < 4:
            self.homography_failures += 1
            log.debug("homography skipped: corner markers=%d", len(pixel_pts))
            return False

        pixel_arr = np.array(pixel_pts, dtype=np.float32)
        world_arr = np.array(world_pts, dtype=np.float32)
        H, _ = cv2.findHomography(pixel_arr, world_arr, cv2.RANSAC, 5.0)
        if H is None:
            self.homography_failures += 1
            log.warning("homography computation failed")
            return False

        reproj = cv2.perspectiveTransform(pixel_arr.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(reproj - world_arr, axis=1)))
        if err > 50.0:
            self.homography_failures += 1
            log.warning("homography rejected, reprojection error %.1fmm", err)
            return False

        self.H_matrix = H
        log.info("homography established (reprojection error %.1fmm)", err)
        return True

    def _pixel_to_world(self, px: float, py: float) -> Tuple[Optional[float], Optional[float]]:
        if self.H_matrix is None:
            return None, None
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.H_matrix)
        return float(world[0][0][0]), float(world[0][0][1])

    @staticmethod
    def _nearest_cell(wx: float, wy: float) -> Tuple[int, int, float]:
        best_row = 0
        best_col = 0
        best_dist = 1e9
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                dx = wx - CELL_CENTERS_X[col]
                dy = wy - CELL_CENTERS_Y[row]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_row = row
                    best_col = col
        return best_row, best_col, best_dist

    def build_board(self, corners, ids) -> np.ndarray:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        if self.H_matrix is None or ids is None:
            return board

        for idx, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in ARUCO_PIECE_IDS:
                continue

            mc = corners[idx][0]
            cx = float(np.mean(mc[:, 0]))
            cy = float(np.mean(mc[:, 1]))
            wx, wy = self._pixel_to_world(cx, cy)
            if wx is None or wy is None:
                continue

            row, col, dist = self._nearest_cell(wx, wy)
            if dist <= CELL_THRESHOLD_MM and board[row, col] == 0:
                board[row, col] = marker_id

        return board

    def get_board_state(self) -> Optional[np.ndarray]:
        frame = self.recv_frame()
        if frame is None:
            return None

        corners, ids = self.detect_markers(frame)
        if self.H_matrix is None:
            self.compute_homography(corners, ids)

        board = self.build_board(corners, ids)
        with self._lock:
            self._latest_board = board
        log.debug("board state updated; markers=%d", self.last_marker_count)
        return board

    def _background_loop(self) -> None:
        while self._running:
            board = self.get_board_state()
            if board is None:
                # Try reconnect quickly without crashing the game loop.
                if self._sock is None and not self.connect_camera():
                    time.sleep(0.5)
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.02)

    def start_background(self) -> bool:
        if self._running:
            return True
        if self._sock is None and not self.connect_camera():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        return True

    def latest_board(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_board is None:
                return None
            return self._latest_board.copy()

    def wait_for_stable_board(
        self,
        timeout: float = STABILITY_TIMEOUT,
        stable_frames: int = STABILITY_FRAMES,
    ) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout
        last = None
        stable = 0

        while time.monotonic() < deadline:
            board = self.latest_board() if self._running else self.get_board_state()
            if board is None:
                log.debug("wait_for_stable_board: no board frame")
                time.sleep(0.05)
                continue
            if last is not None and np.array_equal(board, last):
                stable += 1
            else:
                stable = 1
                last = board.copy()
            if stable >= stable_frames:
                log.info("stable board acquired after %d frames", stable)
                return board
            time.sleep(0.05)

        log.warning("wait_for_stable_board timeout (%.2fs)", timeout)
        return None

    def wait_for_board_change(self, reference: np.ndarray, timeout: float = MOVE_TIMEOUT) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            board = self.wait_for_stable_board(timeout=min(1.5, max(0.3, deadline - time.monotonic())))
            if board is not None and not np.array_equal(board, reference):
                log.info("board change detected")
                return board
        log.warning("wait_for_board_change timeout (%.2fs)", timeout)
        return None

    def verify_board(self, expected: np.ndarray, timeout: float = VERIFY_TIMEOUT) -> bool:
        board = self.wait_for_stable_board(timeout=timeout)
        ok = board is not None and np.array_equal(board, expected)
        if not ok:
            log.warning("verify_board mismatch")
        else:
            log.info("verify_board success")
        return ok
