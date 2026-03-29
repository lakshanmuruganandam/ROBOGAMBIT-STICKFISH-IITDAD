import math
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import socket
import struct

# ── Socket config ─────────────────────────────────────────────────────────────
SERVER_IP   = '10.191.203.117' #update this to server's IP address
SERVER_PORT = 9998

# ── Camera intrinsics ─────────────────────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [1030.4890823364258, 0,   960],
    [0, 1030.489103794098, 540],
    [0,                0,   1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

# ── Board geometry ────────────────────────────────────────────────────────────
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

ROBOT_REALITY = {  # Regularized to a square-like rectangle for stable homography
    21: (477, 179),
    22: (477, -76),
    23: (-184, -76),
    24: (-184, 179),
}

world_pts = np.array([CORNER_WORLD[m]  for m in [21, 22, 23, 24]], dtype=np.float32)
robot_pts = np.array([ROBOT_REALITY[m] for m in [21, 22, 23, 24]], dtype=np.float32)
H_WORLD_TO_ROBOT, _ = cv2.findHomography(world_pts, robot_pts)
# ── ArUco setup ───────────────────────────────────────────────────────────────
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params     = aruco.DetectorParameters()
params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin    = 3
params.adaptiveThreshWinSizeMax    = 35
params.adaptiveThreshWinSizeStep   = 10
params.minMarkerPerimeterRate      = 0.03
params.maxMarkerPerimeterRate      = 4.0
params.polygonalApproxAccuracyRate = 0.03
params.minCornerDistanceRate       = 0.05
params.minDistanceToBorder         = 1
detector = aruco.ArucoDetector(aruco_dict, params)

# ── State ─────────────────────────────────────────────────────────────────────
H_matrix      = None
corner_pixels = {}
prev_board    = None

FRAME_DUMP_DIR = os.path.join(os.path.dirname(__file__), "camera_frames")
FRAME_DUMP_ENABLED = False
FRAME_DUMP_EVERY_N = 1
_FRAME_INDEX = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def pixel_to_world(H, px, py):
    pt = cv2.perspectiveTransform(
        np.array([[[px, py]]], dtype=np.float32), H
    )
    return float(pt[0][0][0]), float(pt[0][0][1])


def world_to_cell(wx, wy):
   
    best_row, best_col, min_dist = None, None, float('inf')
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cx = TOP_LEFT_X - (row * SQUARE_SIZE + SQUARE_SIZE / 2)
            cy = TOP_LEFT_Y - (col * SQUARE_SIZE + SQUARE_SIZE / 2)
            d  = math.hypot(wx - cx, wy - cy)
            if d < min_dist:
                min_dist, best_row, best_col = d, row, col
            
    return best_row, best_col


def build_board(ids, corners, H):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    for i, mid in enumerate(ids.flatten()):
        if mid not in PIECE_IDS:
            continue
        c  = corners[i][0]
        px = float(np.mean(c[:, 0]))
        py = float(np.mean(c[:, 1]))
        wx, wy   = pixel_to_world(H, px, py)
        row, col = world_to_cell(wx, wy)
        if row is not None:
            board[row][col] = mid
    return board


def enhance_image(gray_image):
    """Apply CLAHE contrast enhancement for tougher lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def detect_markers_robust(gray_image):
    """Detect markers using multiple passes and keep the best result."""
    best_corners, best_ids, best_count = None, None, 0

    # Pass 1: original grayscale
    c1, i1, _ = detector.detectMarkers(gray_image)
    n1 = 0 if i1 is None else len(i1)
    if n1 > best_count:
        best_corners, best_ids, best_count = c1, i1, n1

    # Pass 2: CLAHE-enhanced
    enhanced = enhance_image(gray_image)
    c2, i2, _ = detector.detectMarkers(enhanced)
    n2 = 0 if i2 is None else len(i2)
    if n2 > best_count:
        best_corners, best_ids, best_count = c2, i2, n2

    # Pass 3: adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    c3, i3, _ = detector.detectMarkers(adaptive)
    n3 = 0 if i3 is None else len(i3)
    if n3 > best_count:
        best_corners, best_ids, best_count = c3, i3, n3

    return best_corners, best_ids


def _recompute_world_to_robot_homography():
    """Recompute world->robot homography after calibration updates."""
    global H_WORLD_TO_ROBOT
    wpts = np.array([CORNER_WORLD[m] for m in [21, 22, 23, 24]], dtype=np.float32)
    rpts = np.array([ROBOT_REALITY[m] for m in [21, 22, 23, 24]], dtype=np.float32)
    H_WORLD_TO_ROBOT, _ = cv2.findHomography(wpts, rpts)


def set_robot_reality_from_cells(a1_xy, a6_xy, f6_xy, f1_xy):
    """Update runtime robot calibration using board-corner cell coordinates."""
    global ROBOT_REALITY
    ROBOT_REALITY = {
        21: (float(a1_xy[0]), float(a1_xy[1])),
        22: (float(a6_xy[0]), float(a6_xy[1])),
        23: (float(f6_xy[0]), float(f6_xy[1])),
        24: (float(f1_xy[0]), float(f1_xy[1])),
    }
    _recompute_world_to_robot_homography()
    print(f"[CALIB] Runtime ROBOT_REALITY updated: {ROBOT_REALITY}")


def configure_frame_dump(enabled: bool, every_n: int = 1):
    """Enable/disable frame dump and set sampling interval."""
    global FRAME_DUMP_ENABLED, FRAME_DUMP_EVERY_N, _FRAME_INDEX
    FRAME_DUMP_ENABLED = bool(enabled)
    FRAME_DUMP_EVERY_N = max(1, int(every_n))
    _FRAME_INDEX = 0
    if FRAME_DUMP_ENABLED:
        os.makedirs(FRAME_DUMP_DIR, exist_ok=True)
        print(f"[PERCEPTION] Frame dump enabled: dir={FRAME_DUMP_DIR}, every_n={FRAME_DUMP_EVERY_N}")


def _dump_frame(frame):
    global _FRAME_INDEX
    _FRAME_INDEX += 1
    if not FRAME_DUMP_ENABLED:
        return
    if _FRAME_INDEX % FRAME_DUMP_EVERY_N != 0:
        return
    out_path = os.path.join(FRAME_DUMP_DIR, f"frame_{_FRAME_INDEX:06d}.png")
    cv2.imwrite(out_path, frame)


def recv_frame(sock, data, payload_size):
    """Read one frame from the socket stream. Returns (frame, data_remainder)."""
    # Read header
    while len(data) < payload_size:
        packet = sock.recv(4096)
        if not packet:
            return None, data
        data += packet

    packed_msg_size = data[:payload_size]
    data            = data[payload_size:]
    msg_size        = struct.unpack("Q", packed_msg_size)[0]

    # Read frame bytes
    while len(data) < msg_size:
        packet = sock.recv(4096)
        if not packet:
            return None, data
        data += packet

    frame_data = data[:msg_size]
    data       = data[msg_size:]

    frame = cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    return frame, data



def get_piece_poses(ids, corners, H_matrix):
    """Returns a dictionary: { piece_id: [(x1, y1), (x2, y2), ...] }"""
    piece_data = {}
    if ids is None or H_matrix is None:
        return piece_data

    for i, marker_id in enumerate(ids.flatten()):
        # Assuming piece IDs are 1-10 (change if needed)
        if 1 <= marker_id <= 10: 
            # Get center of the marker in pixels
            c = corners[i][0]
            px, py = np.mean(c[:, 0]), np.mean(c[:, 1])
            
            # Transform to Robot World Coordinates (mm)
            rx, ry = pixel_to_world(H_matrix, px, py)
            
            if marker_id not in piece_data:
                piece_data[marker_id] = []
            piece_data[marker_id].append((rx, ry))
            
    return piece_data

# Maintain the buffer as a global so it persists between function calls
data_buffer = b""
H_matrix = None
corner_pixels = {}

def init_perception():
    """Initializes the socket connection."""
    global client_socket
    print(f"Connecting to {SERVER_IP}:{SERVER_PORT} ...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))
    print("Connected ✓")
    return client_socket

def get_stable_board(sock, stability_required=5):
    """
    Runs cycles until the board state is identical for 
    'stability_required' frames in a row.
    """
    global data_buffer, H_matrix, corner_pixels
    
    stable_count = 0
    last_board = None
    payload_size = struct.calcsize("Q")

    print(f"Waiting for stable board (need {stability_required} matching frames)...")

    while True:
        # print("Capturing frame...")
        frame, data_buffer = recv_frame(sock, data_buffer, payload_size)
        if frame is None:
            return None, None
 
        # 1. Standard processing logic from your original code
        frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX)
        _dump_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers_robust(gray)

        current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        if ids is not None:
            # Update corners
            for i, mid in enumerate(ids.flatten()):
                if mid in CORNER_WORLD:
                    corner_pixels[mid] = np.mean(corners[i][0], axis=0)

            # Lock homography
            if H_matrix is None and len(corner_pixels) == 4:
                pixel_pts = np.array([corner_pixels[m] for m in [21, 22, 23, 24]], dtype=np.float32)
                world_pts = np.array([CORNER_WORLD[m]  for m in [21, 22, 23, 24]], dtype=np.float32)
                H_matrix, _ = cv2.findHomography(pixel_pts, world_pts)

            if H_matrix is not None:
                current_board = build_board(ids, corners, H_matrix)

        # 2. Stability Logic: Check if this board matches the last one
        if last_board is not None and np.array_equal(current_board, last_board):
            stable_count += 1
        else:
            stable_count = 0 # Reset if board changed or first frame
        
        last_board = current_board.copy()

        # 3. Return once stable
        if stable_count >= stability_required:
            # Also get the exact poses for your precision pickup
            poses = get_piece_poses(ids, corners, H_matrix)
            print("Board stable ✓")
            # build_board fills row/col in transposed orientation for this camera setup.
            current_board = current_board.T.copy()
            return current_board, poses

        # Optional: brief sleep to prevent CPU spiking
        cv2.waitKey(50)
