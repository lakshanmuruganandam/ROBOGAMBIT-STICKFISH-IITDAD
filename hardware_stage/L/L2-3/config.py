"""
Configuration for Team L2 hardware stage.

This file centralizes all tunables for arm control, perception, and game loop.
"""

import os
import re
from typing import Tuple

import numpy as np

# Network and serial settings
ARM_IP = os.getenv("ROBO_ARM_IP", "192.168.4.1")
ARM_PORT = int(os.getenv("ROBO_ARM_PORT", "80"))
ARM_BASE_URL = f"http://{ARM_IP}:{ARM_PORT}"

SERIAL_PORT = os.getenv("ROBO_SERIAL_PORT", "/dev/ttyUSB0")
BAUD_RATE = int(os.getenv("ROBO_BAUD", "115200"))
SERIAL_TIMEOUT = float(os.getenv("ROBO_SERIAL_TIMEOUT", "1.0"))

CAMERA_IP = os.getenv("ROBO_CAMERA_IP", "192.168.4.6")
CAMERA_PORT = int(os.getenv("ROBO_CAMERA_PORT", "9994"))
CAMERA_CONNECT_TIMEOUT = float(os.getenv("ROBO_CAMERA_CONNECT_TIMEOUT", "8.0"))
CAMERA_RECV_TIMEOUT = float(os.getenv("ROBO_CAMERA_RECV_TIMEOUT", "8.0"))

# HTTP behavior: movement command is fire-and-forget; feedback is strict response
HTTP_CONNECT_TIMEOUT = float(os.getenv("ROBO_HTTP_CONNECT_TIMEOUT", "3.0"))
HTTP_MOVE_SEND_TIMEOUT = float(os.getenv("ROBO_HTTP_MOVE_TIMEOUT", "2.0"))
HTTP_FEEDBACK_TIMEOUT = float(os.getenv("ROBO_HTTP_FEEDBACK_TIMEOUT", "2.0"))
MAX_RETRIES = int(os.getenv("ROBO_MAX_RETRIES", "3"))
RETRY_DELAY_BASE = float(os.getenv("ROBO_RETRY_DELAY", "0.7"))

# Board geometry
BOARD_SIZE = 6
SQUARE_SIZE_MM = 60.0
FILES = "ABCDEF"
RANKS = "123456"
CELL_CENTERS_X = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
CELL_CENTERS_Y = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
CELL_THRESHOLD_MM = 60.0

CORNER_WORLD_COORDS = {
    21: (212.5, 212.5),
    22: (212.5, -212.5),
    23: (-212.5, -212.5),
    24: (-212.5, 212.5),
}
ARUCO_CORNER_IDS = set(CORNER_WORLD_COORDS.keys())
ARUCO_PIECE_IDS = set(range(1, 11))

# Camera intrinsics (replace with your calibrated matrix if needed)
CAMERA_MATRIX = np.array(
    [
        [1030.4890823364258, 0.0, 960.0],
        [0.0, 1030.489103794098, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

# Arm kinematics and timing
Z_SAFE = 180.0
Z_PICK = 10.0
Z_PLACE = 10.0
HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, Z_SAFE

SPEED_FAST = 800
SPEED_NORMAL = 500
SPEED_SLOW = 250

ARM_SETTLE_TIME = 0.35
GRIPPER_ACTIVATE_TIME = 0.30
ARM_ARRIVAL_TIMEOUT = 25.0
ARM_POLL_INTERVAL = 0.25
POSITION_TOLERANCE_MM = 10.0

# Perception stability
STABILITY_FRAMES = 4
STABILITY_TIMEOUT = 8.0
MOVE_TIMEOUT = 180.0
VERIFY_TIMEOUT = 4.0

# Captured-piece parking lanes
GRAVEYARD_WHITE = [(250.0, -150.0 + i * 60.0) for i in range(12)]
GRAVEYARD_BLACK = [(-250.0, -150.0 + i * 60.0) for i in range(12)]

# Piece IDs
EMPTY = 0
W_PAWN, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING = 1, 2, 3, 4, 5
B_PAWN, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING = 6, 7, 8, 9, 10

WHITE_PIECE_IDS = {W_PAWN, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING}
BLACK_PIECE_IDS = {B_PAWN, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING}

PIECE_NAMES = {
    0: ".",
    1: "wP",
    2: "wN",
    3: "wB",
    4: "wQ",
    5: "wK",
    6: "bP",
    7: "bN",
    8: "bB",
    9: "bQ",
    10: "bK",
}

MOVE_REGEX = re.compile(r"^(\d+):([A-F][1-6])->([A-F][1-6])(?:=(\d+))?$")


def rc_to_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def cell_to_rc(cell: str) -> Tuple[int, int]:
    col = ord(cell[0].upper()) - ord("A")
    row = int(cell[1]) - 1
    if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
        raise ValueError(f"Invalid cell: {cell}")
    return row, col


def rc_to_world(row: int, col: int) -> Tuple[float, float]:
    return CELL_CENTERS_X[col], CELL_CENTERS_Y[row]


def world_to_rc(x: float, y: float) -> Tuple[int, int]:
    col = int(round((x - CELL_CENTERS_X[0]) / SQUARE_SIZE_MM))
    row = int(round((y - CELL_CENTERS_Y[0]) / SQUARE_SIZE_MM))
    row = max(0, min(BOARD_SIZE - 1, row))
    col = max(0, min(BOARD_SIZE - 1, col))
    return row, col


def is_white_piece(piece_id: int) -> bool:
    return piece_id in WHITE_PIECE_IDS


def is_black_piece(piece_id: int) -> bool:
    return piece_id in BLACK_PIECE_IDS