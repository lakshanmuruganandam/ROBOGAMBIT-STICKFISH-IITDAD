"""
RoboGambit Hardware Configuration
==================================
Central configuration for the RoboGambit Round 2 hardware stage.
All hardware modules import their constants from here.

Hardware stack:
  - Robot Arm:  Waveshare RoArm M2-S (4-DOF), HTTP control
  - Gripper:    Electromagnetic, serial control
  - Camera:     1920x1080 @ 30fps, socket stream
  - Board:      6x6, 100mm squares, ArUco marker tracking

Coordinate system (world frame, millimetres):
  - Origin at board centre
  - +X = towards file F (white's right)
  - +Y = towards rank 6 (away from white)
  - +Z = upward from the board surface
"""

import re
import numpy as np


# =============================================================================
# 1. NETWORK & SERIAL CONFIGURATION
# =============================================================================

ARM_IP              = "192.168.4.1"
ARM_PORT            = 80
ARM_BASE_URL        = f"http://{ARM_IP}:{ARM_PORT}"

CAMERA_IP           = "10.168.70.199"
CAMERA_HOST         = CAMERA_IP                  # alias for perception_hw
CAMERA_PORT         = 9999
CAMERA_RESOLUTION   = (1920, 1080)
CAMERA_FPS          = 30

SERIAL_PORT         = "COM3"
BAUD_RATE           = 115200
SERIAL_BAUD         = BAUD_RATE                  # alias for arm_controller
SERIAL_TIMEOUT      = 1.0


# =============================================================================
# 2. BOARD GEOMETRY
# =============================================================================

BOARD_SIZE          = 6
SQUARE_SIZE_MM      = 100
CELL_SIZE_MM        = SQUARE_SIZE_MM             # alias for perception_hw
CELL_SIZE           = SQUARE_SIZE_MM             # alias for setup_phase
BOARD_SPAN_MM       = BOARD_SIZE * SQUARE_SIZE_MM

FILES               = "ABCDEF"
RANKS               = "123456"

# Cell centres in world coords (mm). Row 0 = rank 1, Col 0 = file A.
CELL_CENTERS_X = [-250.0, -150.0, -50.0, 50.0, 150.0, 250.0]   # A..F
CELL_CENTERS_Y = [-250.0, -150.0, -50.0, 50.0, 150.0, 250.0]   # rank 1..6

# Pre-built lookup: CELL_WORLD[row][col] = (world_x, world_y)
CELL_WORLD = [
    [(CELL_CENTERS_X[c], CELL_CENTERS_Y[r]) for c in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE)
]
CELL_CENTERS = CELL_WORLD                        # alias for main_l1

# Board origin for setup_phase (bottom-left cell A1 world coords)
BOARD_ORIGIN = (CELL_CENTERS_X[0], CELL_CENTERS_Y[0], 10.0)

# Corner marker world positions (+/-350mm outside the board)
CORNER_WORLD_COORDS = {
    21: ( 350.0,  350.0),
    22: ( 350.0, -350.0),
    23: (-350.0, -350.0),
    24: (-350.0,  350.0),
}
CORNER_MARKER_IDS = list(CORNER_WORLD_COORDS.keys())

# Piece-to-cell threshold (mm) for perception snapping
CELL_THRESHOLD_MM   = 60.0


# =============================================================================
# 3. ARM MOVEMENT PARAMETERS (mm and mm/s)
# =============================================================================

# Z-axis heights
Z_SAFE              = 180.0
Z_HOVER             = 60.0
Z_PICK              = 8.0
Z_PLACE             = 10.0

# Aliases used by different modules
ARM_SAFE_Z          = Z_SAFE
ARM_HOVER_Z         = Z_HOVER
ARM_PICK_Z          = Z_PICK
ARM_PLACE_Z         = Z_PLACE
SAFE_Z              = Z_SAFE
PICK_Z              = Z_PICK
PLACE_Z             = Z_PLACE
PICKUP_HEIGHT       = Z_PICK
PLACE_HEIGHT        = Z_PLACE
TRAVEL_HEIGHT       = Z_SAFE

# Home position (arm rest above board centre)
HOME_X              = 0.0
HOME_Y              = 0.0
HOME_Z              = Z_SAFE
HOME_POSITION       = (HOME_X, HOME_Y, HOME_Z)

# Speeds (arm firmware units, higher = faster)
SPEED_TRANSIT       = 800
SPEED_APPROACH      = 300
SPEED_PICK_PLACE    = 150
DEFAULT_SPEED       = SPEED_TRANSIT

ARM_SPEED_FAST      = 800
ARM_SPEED_NORMAL    = 500
ARM_SPEED_SLOW      = 200
ARM_SPEED_SAFE      = ARM_SPEED_NORMAL


# =============================================================================
# 4. GRAVEYARD POSITIONS (captured piece staging)
# =============================================================================

_GRAVEYARD_X_OFFSET = 400.0
_GRAVEYARD_Y_START  = -250.0
_GRAVEYARD_Y_STEP   = 100.0

GRAVEYARD_SLOTS_WHITE = [
    (_GRAVEYARD_X_OFFSET, _GRAVEYARD_Y_START + i * _GRAVEYARD_Y_STEP)
    for i in range(BOARD_SIZE)
]
GRAVEYARD_SLOTS_BLACK = [
    (-_GRAVEYARD_X_OFFSET, _GRAVEYARD_Y_START + i * _GRAVEYARD_Y_STEP)
    for i in range(BOARD_SIZE)
]

# Aliases
GRAVEYARD_WHITE     = GRAVEYARD_SLOTS_WHITE
GRAVEYARD_BLACK     = GRAVEYARD_SLOTS_BLACK
GRAVEYARD_POSITIONS = GRAVEYARD_SLOTS_WHITE + GRAVEYARD_SLOTS_BLACK
GRAVEYARD_Z_PLACE   = 10.0


# =============================================================================
# 5. STAGING AREA (for setup phase - pieces start here before being placed)
# =============================================================================

STAGING_AREA_ORIGIN   = (450.0, -250.0, 10.0)   # (x, y, z) of first slot
STAGING_ROW_SPACING   = 120.0                     # between rows of piece types
STAGING_COL_SPACING   = 100.0                     # between pieces in a row


# =============================================================================
# 6. TIMING CONSTANTS (seconds)
# =============================================================================

ARM_SETTLE_TIME         = 0.4
SETTLE_DELAY            = ARM_SETTLE_TIME        # alias
GRIPPER_ACTIVATE_TIME   = 0.3
GRIPPER_CONFIRM_TIME    = 0.15
GRIPPER_SETTLE_TIME     = GRIPPER_ACTIVATE_TIME  # alias
MAGNET_ENERGIZE_DELAY   = GRIPPER_ACTIVATE_TIME  # alias
MAGNET_RELEASE_DELAY    = GRIPPER_ACTIVATE_TIME  # alias
PERCEPTION_WAIT         = 0.5
MOVE_COMPLETE_WAIT      = 0.2

RETRY_DELAY             = 1.0
MAX_RETRIES             = 3
RETRY_BACKOFF_BASE      = 1.5

CAMERA_CONNECT_TIMEOUT  = 5.0
ARM_HTTP_TIMEOUT        = 3.0
HTTP_TIMEOUT            = ARM_HTTP_TIMEOUT       # alias

# Perception stability
PERCEPTION_STABLE_FRAMES   = 5
PERCEPTION_STABLE_TIMEOUT  = 10.0
STABILITY_FRAMES           = PERCEPTION_STABLE_FRAMES     # alias
STABILITY_TIMEOUT          = PERCEPTION_STABLE_TIMEOUT    # alias
MOVE_TIMEOUT               = 300.0                        # wait for opponent
MOVE_VERIFY_TIMEOUT        = 5.0
VERIFY_TIMEOUT             = MOVE_VERIFY_TIMEOUT          # alias

# Game clock
GAME_CLOCK_TOTAL        = 900.0
CLOCK_SAFETY_BUFFER     = 30.0

# Arm feedback
MOVEMENT_SPEED_FACTOR   = 1.5
FEEDBACK_POLL_INTERVAL  = 0.1
POSITION_TOLERANCE      = 5.0    # mm


# =============================================================================
# 7. CAMERA INTRINSICS
# =============================================================================

CAMERA_MATRIX = [
    [1030.4890823364258, 0.0, 960.0],
    [0.0, 1030.489103794098, 540.0],
    [0.0, 0.0, 1.0],
]

DIST_COEFFS = [0.0, 0.0, 0.0, 0.0, 0.0]


# =============================================================================
# 8. ARUCO PARAMETERS
# =============================================================================

ARUCO_DICT_NAME     = "DICT_4X4_50"
ARUCO_MARKER_SIZE   = 40.0

ARUCO_CORNER_IDS    = set(CORNER_MARKER_IDS)
ARUCO_PIECE_IDS     = set(range(1, 11))
PIECE_IDS           = ARUCO_PIECE_IDS            # alias for perception_hw
CORNER_IDS          = ARUCO_CORNER_IDS           # alias for perception_hw
ARUCO_ALL_IDS       = ARUCO_CORNER_IDS | ARUCO_PIECE_IDS


# =============================================================================
# 9. PIECE ID MAPPINGS & NAMES
# =============================================================================

EMPTY       = 0
W_PAWN      = 1
W_KNIGHT    = 2
W_BISHOP    = 3
W_QUEEN     = 4
W_KING      = 5
B_PAWN      = 6
B_KNIGHT    = 7
B_BISHOP    = 8
B_QUEEN     = 9
B_KING      = 10

PIECE_NAMES = {
    0: ".",  1: "wP", 2: "wN", 3: "wB", 4: "wQ", 5: "wK",
    6: "bP", 7: "bN", 8: "bB", 9: "bQ", 10: "bK",
}

PIECE_FULL_NAMES = {
    0: "Empty",
    1: "White Pawn", 2: "White Knight", 3: "White Bishop",
    4: "White Queen", 5: "White King",
    6: "Black Pawn", 7: "Black Knight", 8: "Black Bishop",
    9: "Black Queen", 10: "Black King",
}

PIECE_SYMBOLS = {
    0: ".", 1: "P", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "q", 10: "k",
}

WHITE_PIECE_IDS = {W_PAWN, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING}
BLACK_PIECE_IDS = {B_PAWN, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING}
PIECE_IDS_WHITE = WHITE_PIECE_IDS                # alias for main_l1
PIECE_IDS_BLACK = BLACK_PIECE_IDS                # alias for main_l1

MARKER_TO_PIECE = {i: i for i in range(1, 11)}
PIECE_TO_MARKER = {v: k for k, v in MARKER_TO_PIECE.items()}


# =============================================================================
# 10. MOVE FORMAT
# =============================================================================

MOVE_REGEX = re.compile(r"^(\d+):([A-F][1-6])->([A-F][1-6])(?:=(\d+))?$")


# =============================================================================
# 11. HELPER FUNCTIONS
# =============================================================================

def cell_to_world(row: int, col: int) -> tuple:
    """Convert (row, col) to world (x, y) in mm. Row 0=rank 1, Col 0=file A."""
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Invalid indices (row={row}, col={col})")
    return (CELL_CENTERS_X[col], CELL_CENTERS_Y[row])


def cell_name_to_world(cell: str) -> tuple:
    """Convert cell name like 'D3' to world (x, y) in mm."""
    col = ord(cell[0].upper()) - ord("A")
    row = int(cell[1]) - 1
    return cell_to_world(row, col)


def world_to_cell(x: float, y: float) -> tuple:
    """Snap world coords to nearest (row, col)."""
    col = int(round((x - CELL_CENTERS_X[0]) / SQUARE_SIZE_MM))
    row = int(round((y - CELL_CENTERS_Y[0]) / SQUARE_SIZE_MM))
    return (max(0, min(5, row)), max(0, min(5, col)))


def is_white_piece(pid: int) -> bool:
    return pid in WHITE_PIECE_IDS


def is_black_piece(pid: int) -> bool:
    return pid in BLACK_PIECE_IDS
