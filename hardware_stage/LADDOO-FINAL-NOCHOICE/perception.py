"""LADDOO-FINAL-NOCHOICE perception pipeline.

Ingests camera frames, detects ArUco markers, estimates board state with
homography mapping, and stabilizes output via temporal consensus.
"""

import math
from collections import deque
import cv2
import cv2.aruco as aruco
import numpy as np
import socket
import struct

# Network endpoint for the camera stream server.
SERVER_IP   = '10.194.26.222' 
SERVER_PORT = 9999

# Camera intrinsics calibrated for the current stream profile.
CAMERA_MATRIX = np.array([
    [343.49, 0,      320.0],
    [0,      457.99, 240.0],
    [0,      0,      1.0]
], dtype=np.float32)
DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

CORNER_WORLD = {
    21: (212.5,  212.5),
    22: (212.5, -212.5),
    23: (-212.5, -212.5),
    24: (-212.5,  212.5),
}
SQUARE_SIZE = 60
TOP_LEFT_X  = 180
TOP_LEFT_Y  = 180
BOARD_SIZE  = 6
PIECE_IDS   = set(range(1, 11))
CELL_THRESHOLD_MM = 75.0
DEBUG_PERCEPTION = True
DEBUG_EVERY_N_FRAMES = 20

ARUCO_DICT_CANDIDATES = [
    aruco.DICT_4X4_50,
    aruco.DICT_4X4_100,
    aruco.DICT_4X4_250,
    aruco.DICT_5X5_50,
]

EXPECTED_IDS = set(CORNER_WORLD.keys()) | PIECE_IDS

class BoardPerception:
    def __init__(self, connect_socket=True):
        """Initialize ArUco detector and connect to the camera stream."""
        params     = aruco.DetectorParameters()
        params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin    = 3
        params.adaptiveThreshWinSizeMax    = 35
        params.adaptiveThreshWinSizeStep   = 10
        params.minMarkerPerimeterRate      = 0.01
        params.maxMarkerPerimeterRate      = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.minCornerDistanceRate       = 0.05
        params.minDistanceToBorder         = 1
        params.cornerRefinementWinSize     = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        params.adaptiveThreshConstant      = 7
        self._detectors = []
        for dict_id in ARUCO_DICT_CANDIDATES:
            d = aruco.getPredefinedDictionary(dict_id)
            self._detectors.append((dict_id, aruco.ArucoDetector(d, params)))

        self.H_matrix      = None
        self.corner_pixels = {}
        self._board_history = deque(maxlen=15)
        self._pose_history = deque(maxlen=15)
        self._frame_count = 0
        self._last_detect_dict = None
        
        # Optional socket connect supports offline tests using saved frames.
        if connect_socket:
            print(f"Connecting to {SERVER_IP}:{SERVER_PORT} ...")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((SERVER_IP, SERVER_PORT))
            print("Connected ✓")
            self.payload_size = struct.calcsize("Q")
            self.data_buffer  = b""
        else:
            self.client_socket = None

    def _recv_frame(self):
        """Read one length-prefixed JPEG frame from socket."""
        while len(self.data_buffer) < self.payload_size:
            packet = self.client_socket.recv(4096)
            if not packet: return None
            self.data_buffer += packet

        packed_msg_size = self.data_buffer[:self.payload_size]
        self.data_buffer = self.data_buffer[self.payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(self.data_buffer) < msg_size:
            packet = self.client_socket.recv(4096)
            if not packet: return None
            self.data_buffer += packet

        frame_data = self.data_buffer[:msg_size]
        self.data_buffer = self.data_buffer[msg_size:]

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def _pixel_to_world(self, px, py):
        """Map image pixel coordinates into board-world coordinates."""
        pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), self.H_matrix)
        return float(pt[0][0][0]), float(pt[0][0][1])

    def _detect_markers_best(self, frame):
        """Run marker detection on multiple variants and keep the most plausible result."""
        und = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.uint8)

        variants = [
            gray,
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            ),
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                15,
                3,
            ),
            cv2.GaussianBlur(gray, (5, 5), 0),
            cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel),
            cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel),
        ]

        best_corners = None
        best_ids = None
        best_score = -1
        best_dict = None
        for img in variants:
            for dict_id, detector in self._detectors:
                corners, ids, _ = detector.detectMarkers(img)
                if ids is None:
                    continue

                id_list = [int(x) for x in ids.flatten().tolist()]
                valid = sum(1 for x in id_list if x in EXPECTED_IDS)
                corners_seen = sum(1 for x in id_list if x in CORNER_WORLD)
                piece_seen = sum(1 for x in id_list if x in PIECE_IDS)

                # Prefer detections that match expected IDs and preserve corner markers.
                score = (valid * 5) + (corners_seen * 12) + piece_seen
                if score > best_score:
                    best_score = score
                    best_corners = corners
                    best_ids = ids
                    best_dict = dict_id
        self._last_detect_dict = best_dict
        return best_corners, best_ids

    def _world_to_cell(self, wx, wy):
        """Snap world coordinates to nearest board cell and report distance."""
        best_row, best_col, min_dist = None, None, float('inf')
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                # Axis convention: columns map along X, rows map along Y.
                cx = TOP_LEFT_X - (col * SQUARE_SIZE + SQUARE_SIZE / 2)
                cy = TOP_LEFT_Y - (row * SQUARE_SIZE + SQUARE_SIZE / 2)
                d  = math.hypot(wx - cx, wy - cy)
                if d < min_dist:
                    min_dist, best_row, best_col = d, row, col
        return best_row, best_col, min_dist

    @staticmethod
    def _consensus_board(boards):
        """Cell-wise vote with non-zero preference to resist brief marker dropouts."""
        if not boards:
            return None
        stack = np.stack(boards, axis=0).astype(np.int32)
        consensus = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                vals = stack[:, r, c]
                counts = np.bincount(vals, minlength=11)
                zero_count = int(counts[0])
                nonzero_id = int(np.argmax(counts[1:]) + 1)
                nonzero_count = int(counts[nonzero_id])

                # Prefer a recurring non-zero marker when it appears reliably,
                # even if occasional missed frames produced extra zeros.
                if nonzero_count >= 3 and nonzero_count >= int(0.6 * zero_count):
                    consensus[r, c] = nonzero_id
                else:
                    consensus[r, c] = int(np.argmax(counts))
        return consensus

    @staticmethod
    def _median_poses(pose_histories):
        """Robustly aggregate recent piece poses with per-piece medians."""
        if not pose_histories:
            return {}
        merged = {}
        for pose_map in pose_histories:
            for pid, pvals in pose_map.items():
                if isinstance(pvals, tuple):
                    merged.setdefault(pid, []).append(pvals)
                else:
                    merged.setdefault(pid, []).extend(list(pvals))

        out = {}
        for pid, pts in merged.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out[pid] = (float(np.median(xs)), float(np.median(ys)))
        return out

    def get_latest_state(self):
        """Read a live frame and return stabilized board plus piece poses."""
        if not self.client_socket:
            print("[ERROR] Cannot fetch live frame: Socket not connected.")
            return None, None
            
        frame = self._recv_frame()
        if frame is None:
            return None, None
            
        return self.get_latest_state_from_frame(frame)

    def get_latest_state_from_frame(self, frame):
        """Processes a single provided frame to extract board state and poses."""
        corners, ids = self._detect_markers_best(frame)
        self._frame_count += 1

        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        poses = {}

        if DEBUG_PERCEPTION and (self._frame_count % DEBUG_EVERY_N_FRAMES == 0):
            if ids is None:
                print("[PERCEPTION] no markers detected")
            else:
                id_list = [int(x) for x in ids.flatten().tolist()]
                seen_corners = sorted([i for i in id_list if i in CORNER_WORLD])
                print(
                    f"[PERCEPTION] markers={id_list} corners_seen={seen_corners} "
                    f"dict={self._last_detect_dict} homography={self.H_matrix is not None}"
                )

        if ids is not None:
            # Draw overlays for debug visualization.
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Cache detected corner markers used to compute homography.
            for i, mid in enumerate(ids.flatten()):
                if mid in CORNER_WORLD:
                    self.corner_pixels[mid] = np.mean(corners[i][0], axis=0)

            # Lock homography once all board corners are observed.
            if self.H_matrix is None and len(self.corner_pixels) == 4:
                pixel_pts = np.array([self.corner_pixels[m] for m in [21, 22, 23, 24]], dtype=np.float32)
                world_pts = np.array([CORNER_WORLD[m]  for m in [21, 22, 23, 24]], dtype=np.float32)
                self.H_matrix, _ = cv2.findHomography(pixel_pts, world_pts)
                print("Homography locked ✓")

            # Build per-piece poses and board occupancy for this frame.
            if self.H_matrix is not None:
                # Resolve overlaps by keeping nearest marker for each square.
                cell_best = {}
                for i, mid in enumerate(ids.flatten()):
                    if mid not in PIECE_IDS:
                        continue
                    
                    c = corners[i][0]
                    px, py = float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))
                    
                    wx, wy = self._pixel_to_world(px, py)
                    poses.setdefault(int(mid), []).append((wx, wy))
                    
                    row, col, dist = self._world_to_cell(wx, wy)
                    if row is None:
                        continue
                    # Reject far-off snaps that likely come from transient noise.
                    if dist > CELL_THRESHOLD_MM:
                        continue
                    prev = cell_best.get((row, col))
                    if prev is None or dist < prev[0]:
                        cell_best[(row, col)] = (dist, int(mid))

                for (row, col), (_dist, pid) in cell_best.items():
                    board[row][col] = pid

        self._board_history.append(board.copy())
        self._pose_history.append(poses)

        consensus = self._consensus_board(list(self._board_history))
        stable_poses = self._median_poses(list(self._pose_history))
        return (consensus if consensus is not None else board), stable_poses

    def cleanup(self):
        """Close the socket connection."""
        if self.client_socket:
            self.client_socket.close()
