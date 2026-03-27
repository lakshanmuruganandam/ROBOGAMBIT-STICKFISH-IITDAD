"""
RoboGambit Perception Module — Team L
=======================================
Receives camera frames via TCP socket, detects 6x6 board state
using ArUco markers. Thread-safe, multi-pass detection, robust
homography, board stability checking, move verification.

Usage:
    from perception import PerceptionSystem
    perc = PerceptionSystem()
    perc.connect_camera()
    board = perc.get_board_state()
"""

import math
import socket
import struct
import threading
import time
import logging

import cv2
import numpy as np

from config import (
    CAMERA_IP, CAMERA_PORT,
    CAMERA_MATRIX, DIST_COEFFS,
    SQUARE_SIZE_MM, BOARD_SIZE,
    ARUCO_PIECE_IDS, ARUCO_CORNER_IDS,
    CORNER_WORLD_COORDS,
    CAMERA_CONNECT_TIMEOUT, CAMERA_RECV_TIMEOUT,
    CELL_THRESHOLD_MM,
    STABILITY_FRAMES, STABILITY_TIMEOUT,
    MOVE_TIMEOUT, VERIFY_TIMEOUT,
    CELL_CENTERS_X, CELL_CENTERS_Y,
    MAX_RETRIES, RETRY_DELAY_BASE,
)

log = logging.getLogger("perception")


class PerceptionSystem:
    """
    Thread-safe perception: reads camera frames over TCP, detects ArUco
    markers, computes homography, maintains live 6x6 board state.
    """

    def __init__(self):
        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = self._create_detector_params()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Camera intrinsics
        self.camera_matrix = CAMERA_MATRIX.copy()
        self.dist_coeffs = DIST_COEFFS.copy()

        # Homography
        self.H_matrix = None
        self._homography_error = None

        # Board state
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        self.prev_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

        # Corner world coordinates
        self.corner_world = dict(CORNER_WORLD_COORDS)

        # Thread safety
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_board = None
        self._running = False
        self._bg_thread = None

        # Socket
        self._sock = None
        self._recv_buf = b""

        log.info("PerceptionSystem initialized")

    # ──────────────────────────────────────────────────────────────────────
    #  ArUco detector params
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_detector_params():
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.minCornerDistanceRate = 0.05
        params.minDistanceToBorder = 1
        params.perspectiveRemovePixelPerCell = 4
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        return params

    # ──────────────────────────────────────────────────────────────────────
    #  Camera socket connection (with retry)
    # ──────────────────────────────────────────────────────────────────────

    def connect_camera(self, host=None, port=None) -> bool:
        """Connect to camera server via TCP. Retries on failure.
        Args host/port override config values (useful for testing)."""
        import config as _cfg
        cam_ip = host or _cfg.CAMERA_IP
        cam_port = port or _cfg.CAMERA_PORT
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass

                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(CAMERA_CONNECT_TIMEOUT)
                self._sock.connect((cam_ip, cam_port))
                self._sock.settimeout(CAMERA_RECV_TIMEOUT)
                self._recv_buf = b""
                log.info(f"Camera connected: {cam_ip}:{cam_port}")
                return True
            except (socket.error, OSError) as exc:
                log.warning(f"Camera connect attempt {attempt}/{MAX_RETRIES}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_BASE * attempt)

        log.error("Camera connection FAILED after all retries")
        self._sock = None
        return False

    # ──────────────────────────────────────────────────────────────────────
    #  Frame reception
    # ──────────────────────────────────────────────────────────────────────

    def _recv_exact(self, n: int):
        """Receive exactly n bytes, or None on error."""
        while len(self._recv_buf) < n:
            try:
                chunk = self._sock.recv(max(4096, n - len(self._recv_buf)))
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            self._recv_buf += chunk
        result = self._recv_buf[:n]
        self._recv_buf = self._recv_buf[n:]
        return result

    def recv_frame(self):
        """Read one frame from TCP stream. Returns BGR frame or None."""
        if self._sock is None:
            return None
        try:
            header = self._recv_exact(8)
            if header is None:
                return None
            size = struct.unpack("Q", header)[0]
            if size == 0 or size > 10_000_000:
                return None
            jpeg_data = self._recv_exact(size)
            if jpeg_data is None:
                return None
            buf = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return frame
        except Exception as exc:
            log.debug(f"Frame recv error: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    #  Multi-pass ArUco detection
    # ──────────────────────────────────────────────────────────────────────

    def detect_markers(self, frame):
        """
        Multi-pass ArUco detection for maximum robustness.
        Three pre-processing passes; the one finding the most markers wins.
        Returns (corners, ids).
        """
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs,
                                    None, self.camera_matrix)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        best_corners, best_ids = None, None
        best_count = 0

        # Pass 1: original grayscale
        c1, i1, _ = self.detector.detectMarkers(gray)
        n1 = 0 if i1 is None else len(i1)
        if n1 > best_count:
            best_corners, best_ids, best_count = c1, i1, n1

        # Pass 2: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        c2, i2, _ = self.detector.detectMarkers(enhanced)
        n2 = 0 if i2 is None else len(i2)
        if n2 > best_count:
            best_corners, best_ids, best_count = c2, i2, n2

        # Pass 3: adaptive threshold
        try:
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            c3, i3, _ = self.detector.detectMarkers(adaptive)
            n3 = 0 if i3 is None else len(i3)
            if n3 > best_count:
                best_corners, best_ids, best_count = c3, i3, n3
        except Exception:
            pass

        return best_corners, best_ids

    # ──────────────────────────────────────────────────────────────────────
    #  Homography
    # ──────────────────────────────────────────────────────────────────────

    def compute_homography(self, corners, ids):
        """Compute pixel→world homography from corner markers (21-24)."""
        if ids is None:
            return

        ids_flat = ids.flatten()
        pixel_pts, world_pts = [], []

        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid in self.corner_world:
                mc = corners[i][0]
                cx = float(np.mean(mc[:, 0]))
                cy = float(np.mean(mc[:, 1]))
                pixel_pts.append([cx, cy])
                world_pts.append(list(self.corner_world[mid]))

        if len(pixel_pts) < 4:
            return

        pixel_arr = np.array(pixel_pts, dtype=np.float32)
        world_arr = np.array(world_pts, dtype=np.float32)

        H, _ = cv2.findHomography(pixel_arr, world_arr, cv2.RANSAC, 5.0)
        if H is None:
            return

        # Validate reprojection error
        reproj = cv2.perspectiveTransform(
            pixel_arr.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        mean_err = float(np.mean(np.linalg.norm(reproj - world_arr, axis=1)))

        if mean_err > 50.0:
            log.warning(f"Homography rejected: {mean_err:.1f}mm error")
            return

        self.H_matrix = H
        self._homography_error = mean_err

    # ──────────────────────────────────────────────────────────────────────
    #  Coordinate transforms
    # ──────────────────────────────────────────────────────────────────────

    def pixel_to_world(self, px, py):
        """Pixel → world (mm). Returns (wx, wy) or (None, None)."""
        if self.H_matrix is None:
            return None, None
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.H_matrix)
        return float(world[0][0][0]), float(world[0][0][1])

    # ──────────────────────────────────────────────────────────────────────
    #  Board building
    # ──────────────────────────────────────────────────────────────────────

    def build_board(self, corners, ids):
        """Map piece markers (IDs 1-10) to 6x6 board cells."""
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        if self.H_matrix is None or ids is None:
            return board

        ids_flat = ids.flatten()
        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid not in ARUCO_PIECE_IDS:
                continue

            mc = corners[i][0]
            cx = float(np.mean(mc[:, 0]))
            cy = float(np.mean(mc[:, 1]))
            wx, wy = self.pixel_to_world(cx, cy)
            if wx is None:
                continue

            # Find nearest cell centre
            best_r, best_c, min_dist = 0, 0, float("inf")
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    d = math.hypot(wx - CELL_CENTERS_X[c], wy - CELL_CENTERS_Y[r])
                    if d < min_dist:
                        min_dist, best_r, best_c = d, r, c

            if min_dist <= CELL_THRESHOLD_MM:
                board[best_r][best_c] = mid

        return board

    # ──────────────────────────────────────────────────────────────────────
    #  Single-frame board read
    # ──────────────────────────────────────────────────────────────────────

    def get_board_state(self):
        """Read one frame → detect → return 6x6 board, or None."""
        frame = self.recv_frame()
        if frame is None:
            return None

        corners, ids = self.detect_markers(frame)
        self.compute_homography(corners, ids)
        new_board = self.build_board(corners, ids)

        with self._lock:
            self.prev_board = self.board.copy()
            self.board = new_board
            self._latest_frame = frame
            self._latest_board = new_board.copy()

        return new_board

    def capture_board(self):
        """Alias for get_board_state()."""
        return self.get_board_state()

    # ──────────────────────────────────────────────────────────────────────
    #  Piece poses (exact world coords via homography)
    # ──────────────────────────────────────────────────────────────────────

    def get_piece_poses(self, corners, ids):
        """Returns {piece_id: (world_x, world_y, row, col)}."""
        poses = {}
        if self.H_matrix is None or ids is None:
            return poses

        ids_flat = ids.flatten()
        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid not in ARUCO_PIECE_IDS:
                continue
            mc = corners[i][0]
            cx = float(np.mean(mc[:, 0]))
            cy = float(np.mean(mc[:, 1]))
            wx, wy = self.pixel_to_world(cx, cy)
            if wx is None:
                continue
            row = max(0, min(5, int(round((wy - CELL_CENTERS_Y[0]) / SQUARE_SIZE_MM))))
            col = max(0, min(5, int(round((wx - CELL_CENTERS_X[0]) / SQUARE_SIZE_MM))))
            poses[mid] = (wx, wy, row, col)

        return poses

    def get_board_with_poses(self):
        """Read one frame → (board, {piece_id: (wx, wy, row, col)})."""
        frame = self.recv_frame()
        if frame is None:
            return None, {}

        corners, ids = self.detect_markers(frame)
        self.compute_homography(corners, ids)
        board = self.build_board(corners, ids)
        poses = self.get_piece_poses(corners, ids)

        with self._lock:
            self.prev_board = self.board.copy()
            self.board = board
            self._latest_frame = frame
            self._latest_board = board.copy()

        return board, poses

    # ──────────────────────────────────────────────────────────────────────
    #  Board change / stability waiting
    # ──────────────────────────────────────────────────────────────────────

    def wait_for_board_change(self, timeout=None):
        """Block until board state differs from current. Returns new board or None."""
        timeout = timeout or MOVE_TIMEOUT
        deadline = time.monotonic() + timeout
        ref_board = self.board.copy()

        while time.monotonic() < deadline:
            new_board = self.get_board_state()
            if new_board is None:
                time.sleep(0.05)
                continue
            if not np.array_equal(new_board, ref_board):
                return new_board
            time.sleep(0.05)

        log.warning("wait_for_board_change timed out")
        return None

    def wait_for_stable_board(self, frames=None, timeout=None):
        """Block until board is identical for N consecutive reads."""
        frames = frames or STABILITY_FRAMES
        timeout = timeout or STABILITY_TIMEOUT
        deadline = time.monotonic() + timeout
        streak = 0
        last_board = None

        while time.monotonic() < deadline:
            current = self.get_board_state()
            if current is None:
                streak = 0
                time.sleep(0.05)
                continue
            if last_board is not None and np.array_equal(current, last_board):
                streak += 1
            else:
                streak = 1
            last_board = current.copy()
            if streak >= frames:
                return current
            time.sleep(0.05)

        log.warning("wait_for_stable_board timed out")
        return None

    # ──────────────────────────────────────────────────────────────────────
    #  Move verification
    # ──────────────────────────────────────────────────────────────────────

    def verify_move_executed(self, expected_board, timeout=None):
        """Verify board matches expected after arm execution."""
        timeout = timeout or VERIFY_TIMEOUT
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            current = self.get_board_state()
            if current is not None and np.array_equal(current, expected_board):
                return True
            time.sleep(0.1)

        stable = self.wait_for_stable_board(frames=3, timeout=2.0)
        if stable is not None and np.array_equal(stable, expected_board):
            return True

        log.warning("Move verification FAILED — board mismatch")
        return False

    # ──────────────────────────────────────────────────────────────────────
    #  Opponent move detection
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def detect_opponent_move(old_board, new_board):
        """Compare boards → {'from_cell', 'to_cell', 'piece', 'captured_piece'} or None."""
        diff = (old_board != new_board)
        changed = list(zip(*np.where(diff)))
        if len(changed) < 2:
            return None

        vacated, occupied = [], []
        for r, c in changed:
            old_val, new_val = int(old_board[r][c]), int(new_board[r][c])
            if old_val != 0 and new_val == 0:
                vacated.append((r, c, old_val))
            elif new_val != 0:
                occupied.append((r, c, new_val, old_val))

        if len(vacated) == 1 and len(occupied) == 1:
            fr, fc, piece = vacated[0]
            tr, tc, new_piece, old_piece = occupied[0]
            if piece == new_piece:
                return {
                    "from_cell": (fr, fc), "to_cell": (tr, tc),
                    "piece": piece, "captured_piece": old_piece,
                }
        return None

    # ──────────────────────────────────────────────────────────────────────
    #  Background thread
    # ──────────────────────────────────────────────────────────────────────

    def start_background(self):
        """Start daemon thread for continuous board updates."""
        if self._running:
            return
        self._running = True
        self._bg_thread = threading.Thread(
            target=self._bg_loop, daemon=True, name="PerceptionBG",
        )
        self._bg_thread.start()
        log.info("Background perception started")

    def _bg_loop(self):
        while self._running:
            frame = self.recv_frame()
            if frame is None:
                time.sleep(0.5)
                try:
                    self.connect_camera()
                except Exception:
                    pass
                continue
            corners, ids = self.detect_markers(frame)
            self.compute_homography(corners, ids)
            new_board = self.build_board(corners, ids)
            with self._lock:
                self.prev_board = self.board.copy()
                self.board = new_board
                self._latest_frame = frame
                self._latest_board = new_board.copy()

    def stop(self):
        """Stop background thread and close socket."""
        self._running = False
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=3.0)
            self._bg_thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        log.info("PerceptionSystem stopped")
