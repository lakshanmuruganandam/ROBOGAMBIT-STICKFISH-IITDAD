"""
RoboGambit Hardware Configuration — Team L
============================================
Central config for RoboGambit Round 2 hardware stage.

Hardware stack:
  - Robot Arm:  Waveshare RoArm M2-S (4-DOF), HTTP + WiFi
  - Gripper:    Electromagnetic (serial) + Servo fingers (HTTP)
  - Camera:     1920x1080 @ 30fps, TCP socket stream
  - Board:      6x6, 60mm squares, ArUco marker tracking

KEY FIX:  The arm HTTP API blocks until movement completes.
          Large movements take 15-30s → timeout errors.
          Solution: fire-and-forget HTTP + poll feedback.
"""

import re
import numpy as np


# =============================================================================
# 1. NETWORK & SERIAL
# =============================================================================

ARM_IP              = "192.168.4.1"
ARM_PORT            = 80
ARM_BASE_URL        = f"http://{ARM_IP}:{ARM_PORT}"

CAMERA_IP           = "192.168.4.6"
CAMERA_PORT         = 9999
CAMERA_RESOLUTION   = (1920, 1080)

SERIAL_PORT         = "/dev/ttyUSB0"
BAUD_RATE           = 115200
SERIAL_TIMEOUT      = 1.0


# =============================================================================
# 2. BOARD GEOMETRY
# =============================================================================

BOARD_SIZE          = 6
SQUARE_SIZE_MM      = 60
BOARD_SPAN_MM       = BOARD_SIZE * SQUARE_SIZE_MM
FILES               = "ABCDEF"
RANKS               = "123456"

# Cell centres in world coords (mm). Row 0 = rank 1, Col 0 = file A.
CELL_CENTERS_X = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
CELL_CENTERS_Y = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]

CELL_WORLD = [
    [(CELL_CENTERS_X[c], CELL_CENTERS_Y[r]) for c in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE)
]

CORNER_WORLD_COORDS = {
    21: ( 212.5,  212.5),
    22: ( 212.5, -212.5),
    23: (-212.5, -212.5),
    24: (-212.5,  212.5),
}
CORNER_MARKER_IDS   = list(CORNER_WORLD_COORDS.keys())
CELL_THRESHOLD_MM   = 60.0


# =============================================================================
# 3. ARM MOVEMENT
# =============================================================================

Z_SAFE              = 180.0
Z_HOVER             = 60.0
Z_PICK              = 8.0
Z_PLACE             = 10.0

HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, Z_SAFE
HOME_POSITION       = (HOME_X, HOME_Y, HOME_Z)

SPEED_FAST          = 800
SPEED_NORMAL        = 500
SPEED_SLOW          = 200
DEFAULT_SPEED       = SPEED_NORMAL

GRIPPER_OPEN_ANGLE  = 30.0
GRIPPER_CLOSE_ANGLE = 150.0
GRIPPER_SERVO_ID    = 5
GRIPPER_SERVO_SPEED = 200


# =============================================================================
# 4. GRAVEYARD POSITIONS
# =============================================================================

_GX = 250.0
_GY0 = -150.0
_GS = 60.0
GRAVEYARD_WHITE = [( _GX, _GY0 + i * _GS) for i in range(12)]
GRAVEYARD_BLACK = [(-_GX, _GY0 + i * _GS) for i in range(12)]


# =============================================================================
# 5. TIMING (seconds) — THIS IS THE CRITICAL SECTION
# =============================================================================

ARM_SETTLE_TIME         = 0.5
GRIPPER_ACTIVATE_TIME   = 0.4
GRIPPER_SERVO_TIME      = 0.35

# ── HTTP timeouts ──
# CONNECT timeout: just for TCP handshake (should be fast if arm is reachable)
HTTP_CONNECT_TIMEOUT    = 5.0
# READ timeout for FEEDBACK queries (T:105) — fast, arm responds immediately
HTTP_FEEDBACK_TIMEOUT   = 3.0
# READ timeout for MOVEMENT commands (T:104) — arm blocks until move complete
# We use fire-and-forget, so this is SHORT on purpose. A timeout is EXPECTED
# and NOT an error — it means the arm received the command and is moving.
HTTP_MOVE_SEND_TIMEOUT  = 3.0
# How long to poll feedback waiting for arm to arrive at destination
ARM_ARRIVAL_TIMEOUT     = 30.0
# How often to poll arm position during movement
ARM_POLL_INTERVAL       = 0.3

# ── Retry ──
MAX_RETRIES             = 5
RETRY_DELAY_BASE        = 1.0
RETRY_DELAY_MAX         = 8.0

# ── Perception stability ──
STABILITY_FRAMES        = 5
STABILITY_TIMEOUT       = 10.0
MOVE_TIMEOUT            = 300.0
VERIFY_TIMEOUT          = 5.0
CAMERA_CONNECT_TIMEOUT  = 10.0
CAMERA_RECV_TIMEOUT     = 10.0

# ── Arm feedback position check ──
POSITION_TOLERANCE      = 8.0    # mm — how close is "arrived"
MOVEMENT_SPEED_FACTOR   = 1.5

# ── Game clock ──
GAME_CLOCK_TOTAL        = 900.0
CLOCK_SAFETY_BUFFER     = 30.0


# =============================================================================
# 6. CAMERA INTRINSICS
# =============================================================================

CAMERA_MATRIX = np.array([
    [1030.4890823364258, 0.0, 960.0],
    [0.0, 1030.489103794098, 540.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)


# =============================================================================
# 7. ARUCO
# =============================================================================

ARUCO_DICT_NAME     = "DICT_4X4_50"
ARUCO_CORNER_IDS    = set(CORNER_MARKER_IDS)
ARUCO_PIECE_IDS     = set(range(1, 11))


# =============================================================================
# 8. PIECE IDS
# =============================================================================

EMPTY    = 0
W_PAWN   = 1;  W_KNIGHT = 2;  W_BISHOP = 3;  W_QUEEN = 4;  W_KING = 5
B_PAWN   = 6;  B_KNIGHT = 7;  B_BISHOP = 8;  B_QUEEN = 9;  B_KING = 10

PIECE_NAMES = {
    0: ".",  1: "wP", 2: "wN", 3: "wB", 4: "wQ", 5: "wK",
    6: "bP", 7: "bN", 8: "bB", 9: "bQ", 10: "bK",
}

WHITE_PIECE_IDS = {W_PAWN, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING}
BLACK_PIECE_IDS = {B_PAWN, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING}

MOVE_REGEX = re.compile(r"^(\d+):([A-F][1-6])->([A-F][1-6])(?:=(\d+))?$")


# =============================================================================
# 9. HELPERS
# =============================================================================

def cell_to_world(row: int, col: int) -> tuple:
    """(row, col) -> (world_x, world_y) mm."""
    return (CELL_CENTERS_X[col], CELL_CENTERS_Y[row])

def cell_name_to_world(cell: str) -> tuple:
    """'D3' -> (world_x, world_y) mm."""
    col = ord(cell[0].upper()) - ord("A")
    row = int(cell[1]) - 1
    return cell_to_world(row, col)

def world_to_cell(x: float, y: float) -> tuple:
    """Snap world (x,y) to nearest (row, col)."""
    col = int(round((x - CELL_CENTERS_X[0]) / SQUARE_SIZE_MM))
    row = int(round((y - CELL_CENTERS_Y[0]) / SQUARE_SIZE_MM))
    return (max(0, min(5, row)), max(0, min(5, col)))

def rc_to_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"

def is_white_piece(pid: int) -> bool:
    return pid in WHITE_PIECE_IDS

def is_black_piece(pid: int) -> bool:
    return pid in BLACK_PIECE_IDS
