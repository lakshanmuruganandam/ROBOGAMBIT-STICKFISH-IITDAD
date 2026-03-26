"""
perception_hw.py - Hardware perception module for RoboGambit.

Receives camera frames via TCP socket and detects 6x6 board state
using ArUco markers. Designed for robust real-time use during
competition with thread-safe background frame acquisition,
multi-pass marker detection, board stability checking, and
move verification.

Adapts the competition perception.py for the hardware pipeline.
"""

import math
import socket
import struct
import threading
import time

import cv2
import numpy as np

from config_l1 import (
    CAMERA_IP,
    CAMERA_PORT,
    CAMERA_MATRIX,
    DIST_COEFFS,
    SQUARE_SIZE_MM,
    BOARD_SIZE,
    ARUCO_PIECE_IDS,
    ARUCO_CORNER_IDS,
    CORNER_WORLD_COORDS,
    CAMERA_CONNECT_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)

# ── Perception-specific constants (not in config.py) ──
CELL_THRESHOLD_MM  = 60    # max distance (mm) from cell centre to accept a piece
STABILITY_FRAMES   = 5     # consecutive identical reads required for "stable" board
STABILITY_TIMEOUT  = 10    # seconds to wait for board stability
MOVE_TIMEOUT       = 30    # seconds to wait for an opponent move
VERIFY_TIMEOUT     = 5     # seconds to verify our move was executed


class PerceptionSystem:
    """
    Thread-safe perception system that reads camera frames over TCP,
    detects ArUco markers, computes board homography, and maintains
    a live 6x6 board state for the RoboGambit game engine.
    """

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def __init__(self):
        # ArUco detector — DICT_4X4_50 with subpixel corner refinement
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = self._create_detector_params()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Camera intrinsics (provided by competition organisers)
        self.camera_matrix = np.array(CAMERA_MATRIX, dtype=np.float32)
        self.dist_coeffs = np.array(DIST_COEFFS, dtype=np.float32)

        # Homography — computed once and reused until invalidated
        self.H_matrix = None
        self._homography_error = None  # mean reprojection error in mm

        # Board state
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        self.prev_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

        # Corner world coordinates for homography computation
        self.corner_world = dict(CORNER_WORLD_COORDS)

        # Thread safety
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_board = None
        self._running = False
        self._bg_thread = None

        # Socket
        self._sock = None

        print("[PerceptionSystem] Initialized (multi-pass, thread-safe)")

    # ------------------------------------------------------------------ #
    #  ArUco detector configuration
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_detector_params():
        """Return ArUco DetectorParameters with subpixel corner refinement."""
        params = cv2.aruco.DetectorParameters()
        # Subpixel refinement for accurate corner localisation
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        # Adaptive threshold tuning
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        # Perspective removal
        params.perspectiveRemovePixelPerCell = 4
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        return params

    # ------------------------------------------------------------------ #
    #  Camera socket
    # ------------------------------------------------------------------ #

    def connect_camera(self):
        """
        Connect to the camera server via TCP.
        Returns True on success, False on failure.
        """
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(CAMERA_CONNECT_TIMEOUT)
            self._sock.connect((CAMERA_IP, CAMERA_PORT))
            self._sock.settimeout(10.0)
            print(f"[PerceptionSystem] Connected to camera at {CAMERA_IP}:{CAMERA_PORT}")
            return True
        except (socket.error, OSError) as exc:
            print(f"[PerceptionSystem] Camera connection failed: {exc}")
            self._sock = None
            return False

    def _recv_exact(self, n):
        """Receive exactly *n* bytes from the socket, or return None."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def recv_frame(self):
        """
        Read one JPEG frame from the socket stream.

        Protocol:
            8-byte header  — struct 'Q' (uint64, big-endian) — JPEG payload size
            N bytes         — raw JPEG data

        Returns a decoded BGR OpenCV frame, or None on error.
        """
        if self._sock is None:
            return None

        header = self._recv_exact(8)
        if header is None:
            return None

        size = struct.unpack("Q", header)[0]
        if size == 0 or size > 10_000_000:  # sanity: max ~10 MB
            return None

        jpeg_data = self._recv_exact(size)
        if jpeg_data is None:
            return None

        buf = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return frame

    # ------------------------------------------------------------------ #
    #  Multi-pass ArUco detection
    # ------------------------------------------------------------------ #

    def detect_markers(self, frame):
        """
        Multi-pass ArUco detection for maximum robustness.

        Three image pre-processing passes are attempted; the pass that
        finds the **most** markers wins.

        Returns (corners, ids) — both may be None if nothing is found.
        """
        # Undistort and convert to grayscale
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs,
                                    None, self.camera_matrix)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        best_corners, best_ids = None, None
        best_count = 0

        # Pass 1 — original grayscale
        c1, i1, _ = self.detector.detectMarkers(gray)
        count1 = 0 if i1 is None else len(i1)
        if count1 > best_count:
            best_corners, best_ids = c1, i1
            best_count = count1

        # Pass 2 — CLAHE enhanced (clipLimit=2.0, tileGridSize 8x8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        c2, i2, _ = self.detector.detectMarkers(enhanced)
        count2 = 0 if i2 is None else len(i2)
        if count2 > best_count:
            best_corners, best_ids = c2, i2
            best_count = count2

        # Pass 3 — adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        c3, i3, _ = self.detector.detectMarkers(adaptive)
        count3 = 0 if i3 is None else len(i3)
        if count3 > best_count:
            best_corners, best_ids = c3, i3
            best_count = count3

        return best_corners, best_ids

    # ------------------------------------------------------------------ #
    #  Homography
    # ------------------------------------------------------------------ #

    def compute_homography(self, corners, ids):
        """
        Compute pixel-to-world homography from corner markers (IDs 21-24).

        Uses cv2.findHomography with RANSAC.  The homography is cached in
        self.H_matrix and only recomputed when new corner data is available.

        Corner world coordinates (mm, from config.py):
            21 -> (-350, -350)   bottom-left  (near A1)
            22 -> ( 350, -350)   bottom-right (near F1)
            23 -> ( 350,  350)   top-right    (near F6)
            24 -> (-350,  350)   top-left     (near A6)
        """
        if ids is None:
            print("[PerceptionSystem] Homography FAILED: No markers detected")
            return

        ids_flat = ids.flatten()
        detected_corners = [int(m) for m in ids_flat if int(m) in self.corner_world]

        pixel_pts = []
        world_pts = []

        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid in self.corner_world:
                marker_corners = corners[i][0]  # shape (4, 2)
                cx = float(np.mean(marker_corners[:, 0]))
                cy = float(np.mean(marker_corners[:, 1]))
                pixel_pts.append([cx, cy])
                world_pts.append(list(self.corner_world[mid]))

        if len(pixel_pts) < 4:
            print(f"[PerceptionSystem] Homography FAILED: Only {len(pixel_pts)} corner markers (need 4). Detected: {detected_corners}")
            return

        pixel_arr = np.array(pixel_pts, dtype=np.float32)
        world_arr = np.array(world_pts, dtype=np.float32)

        H, status = cv2.findHomography(pixel_arr, world_arr, cv2.RANSAC, 5.0)
        if H is None:
            print("[PerceptionSystem] Homography FAILED: findHomography returned None")
            return

        # Validate via reprojection error
        reproj = cv2.perspectiveTransform(
            pixel_arr.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(reproj - world_arr, axis=1)
        mean_err = float(np.mean(errors))

        if mean_err > 50.0:
            print(f"[PerceptionSystem] Homography FAILED: Reprojection error {mean_err:.1f} mm (max 50mm)")
            return

        self.H_matrix = H
        self._homography_error = mean_err
        print(f"[PerceptionSystem] Homography OK: {mean_err:.1f} mm reprojection error")

    # ------------------------------------------------------------------ #
    #  Coordinate transforms
    # ------------------------------------------------------------------ #

    def pixel_to_world(self, px, py):
        """
        Transform a single pixel coordinate to world coordinates (mm)
        using the current homography.

        Returns (world_x, world_y) or (None, None) if no homography.
        """
        if self.H_matrix is None:
            return None, None

        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.H_matrix)
        return float(world[0][0][0]), float(world[0][0][1])

    # ------------------------------------------------------------------ #
    #  Board building
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cell_center_world(row, col):
        """
        World coordinates (mm) of the centre of cell (row, col).

        Uses the config.py convention:
            col 0 (file A) = -250, col 5 (file F) = +250
            row 0 (rank 1) = -250, row 5 (rank 6) = +250
        """
        wx = -250.0 + col * SQUARE_SIZE_MM
        wy = -250.0 + row * SQUARE_SIZE_MM
        return wx, wy

    def build_board(self, corners, ids):
        """
        Map piece markers (IDs 1-10) to 6x6 board cells.

        For each detected piece marker the centre pixel is transformed
        to world coordinates, and the nearest cell within
        CELL_THRESHOLD_MM (default 60 mm) is assigned.

        Returns a 6x6 numpy int32 array.
        """
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

        if self.H_matrix is None or ids is None:
            return board

        ids_flat = ids.flatten()

        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid not in ARUCO_PIECE_IDS:
                continue

            marker_corners = corners[i][0]  # (4, 2)
            cx = float(np.mean(marker_corners[:, 0]))
            cy = float(np.mean(marker_corners[:, 1]))

            wx, wy = self.pixel_to_world(cx, cy)
            if wx is None:
                continue

            # Find nearest cell
            best_row, best_col = 0, 0
            min_dist = float("inf")

            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    cell_wx, cell_wy = self._cell_center_world(r, c)
                    dist = math.sqrt((wx - cell_wx) ** 2 + (wy - cell_wy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_row, best_col = r, c

            if min_dist <= CELL_THRESHOLD_MM:
                board[best_row][best_col] = mid

        return board

    # ------------------------------------------------------------------ #
    #  Single-frame board read
    # ------------------------------------------------------------------ #

    def get_board_state(self):
        """
        Process one frame from the camera and return the current 6x6 board.

        Thread-safe: acquires the internal lock while updating state.
        Returns a 6x6 numpy int32 array, or None on frame read failure.
        """
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
        """Alias for get_board_state() - returns 6x6 board array.
        Used by setup_phase for initial board verification."""
        return self.get_board_state()

    def get_piece_poses(self, corners, ids):
        """
        Get world coordinates (x, y) of all detected pieces.

        Returns dict: {piece_id: (world_x, world_y, row, col)}
        For each piece marker, computes the exact world position via homography.
        """
        poses = {}

        if self.H_matrix is None or ids is None:
            return poses

        ids_flat = ids.flatten()

        for i, mid in enumerate(ids_flat):
            mid = int(mid)
            if mid not in ARUCO_PIECE_IDS:
                continue

            marker_corners = corners[i][0]
            cx = float(np.mean(marker_corners[:, 0]))
            cy = float(np.mean(marker_corners[:, 1]))

            wx, wy = self.pixel_to_world(cx, cy)
            if wx is None:
                continue

            row = int(round((wy + 250.0) / SQUARE_SIZE_MM))
            col = int(round((wx + 250.0) / SQUARE_SIZE_MM))
            row = max(0, min(BOARD_SIZE - 1, row))
            col = max(0, min(BOARD_SIZE - 1, col))

            poses[mid] = (wx, wy, row, col)

        return poses

    def get_board_with_poses(self):
        """
        Get both board state and piece poses in one frame read.
        Returns (board, poses_dict) where poses_dict = {piece_id: (wx, wy, row, col)}
        """
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

    # ------------------------------------------------------------------ #
    #  Move detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_opponent_move(old_board, new_board):
        """
        Compare two board states and determine what changed.

        Returns a dict with keys:
            'from_cell'     — (row, col) the piece moved from
            'to_cell'       — (row, col) the piece moved to
            'piece'         — int piece ID that moved
            'captured_piece' — int piece ID that was captured, or 0

        Returns None if no move is detected or the diff is ambiguous.
        """
        diff = (old_board != new_board)
        changed = list(zip(*np.where(diff)))

        if len(changed) < 2:
            return None

        # Identify squares that lost a piece and squares that gained one
        vacated = []   # had a piece, now empty (or different piece)
        occupied = []  # was empty (or different), now has a piece

        for r, c in changed:
            old_val = int(old_board[r][c])
            new_val = int(new_board[r][c])
            if old_val != 0 and new_val == 0:
                vacated.append((r, c, old_val))
            elif new_val != 0:
                occupied.append((r, c, new_val, old_val))

        if len(vacated) == 1 and len(occupied) == 1:
            fr, fc, piece = vacated[0]
            tr, tc, new_piece, old_piece = occupied[0]
            if piece == new_piece:
                return {
                    "from_cell": (fr, fc),
                    "to_cell": (tr, tc),
                    "piece": piece,
                    "captured_piece": old_piece,
                }

        # Ambiguous or complex change — return None
        return None

    # ------------------------------------------------------------------ #
    #  Board change / stability waiting
    # ------------------------------------------------------------------ #

    def wait_for_board_change(self, timeout=None):
        """
        Block until the board state differs from the last known state
        (i.e., the opponent has made a move).

        Returns the new board, or None on timeout.
        """
        if timeout is None:
            timeout = MOVE_TIMEOUT

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

        print("[PerceptionSystem] wait_for_board_change timed out")
        return None

    def wait_for_stable_board(self, stability_frames=None, timeout=None):
        """
        Block until the detected board state is identical for
        *stability_frames* consecutive reads.  This ensures no hand
        or arm is obscuring the board.

        Returns the stable board, or None on timeout.
        """
        if stability_frames is None:
            stability_frames = STABILITY_FRAMES
        if timeout is None:
            timeout = STABILITY_TIMEOUT

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

            if streak >= stability_frames:
                return current

            time.sleep(0.05)

        print("[PerceptionSystem] wait_for_stable_board timed out")
        return None

    # ------------------------------------------------------------------ #
    #  Background thread
    # ------------------------------------------------------------------ #

    def start_background_thread(self):
        """
        Start a daemon thread that continuously reads frames and
        updates the board state.  The latest frame and board are
        stored in thread-safe attributes.
        """
        if self._running:
            return

        self._running = True
        self._bg_thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="PerceptionBG",
        )
        self._bg_thread.start()
        print("[PerceptionSystem] Background thread started")

    def _background_loop(self):
        """Continuously read frames and update board state."""
        while self._running:
            frame = self.recv_frame()
            if frame is None:
                # Attempt reconnection
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
        """Stop the background thread and close the camera socket."""
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

        print("[PerceptionSystem] Stopped")

    # ------------------------------------------------------------------ #
    #  Move verification
    # ------------------------------------------------------------------ #

    def verify_move_executed(self, expected_board, timeout=None):
        """
        After our robotic arm executes a move, verify that perception
        confirms the expected board state.

        Repeatedly reads frames until the board matches *expected_board*
        or *timeout* seconds elapse.

        Returns True if the board matches, False otherwise.
        """
        if timeout is None:
            timeout = VERIFY_TIMEOUT

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            current = self.get_board_state()
            if current is not None and np.array_equal(current, expected_board):
                return True
            time.sleep(0.1)

        # Final attempt — wait for a stable reading before giving up
        stable = self.wait_for_stable_board(stability_frames=3, timeout=2.0)
        if stable is not None and np.array_equal(stable, expected_board):
            return True

        print("[PerceptionSystem] Move verification FAILED — board mismatch")
        return False
