"""
Main hardware-stage controller for Team L2.

Usage examples:
  python main.py --white
  python main.py --black
  python main.py --calibrate
"""

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import cv2
import numpy as np
import requests
import requests.adapters

import game
import perception as perception_module
from perception import PerceptionSystem


# Inlined configuration (merged from config.py) for standalone main.py execution.
ARM_IP = os.getenv("ROBO_ARM_IP", "192.168.4.1")
ARM_PORT = int(os.getenv("ROBO_ARM_PORT", "80"))
ARM_BASE_URL = f"http://{ARM_IP}:{ARM_PORT}"

SERIAL_PORT = os.getenv("ROBO_SERIAL_PORT", "/dev/ttyUSB0")
BAUD_RATE = int(os.getenv("ROBO_BAUD", "115200"))
SERIAL_TIMEOUT = float(os.getenv("ROBO_SERIAL_TIMEOUT", "1.0"))

HTTP_MOVE_SEND_TIMEOUT = float(os.getenv("ROBO_HTTP_MOVE_TIMEOUT", "2.0"))
HTTP_FEEDBACK_TIMEOUT = float(os.getenv("ROBO_HTTP_FEEDBACK_TIMEOUT", "2.0"))
MAX_RETRIES = int(os.getenv("ROBO_MAX_RETRIES", "3"))
RETRY_DELAY_BASE = float(os.getenv("ROBO_RETRY_DELAY", "0.7"))

BOARD_SIZE = 6
SQUARE_SIZE_MM = 60.0
TOP_LEFT_X_MM = 180.0
TOP_LEFT_Y_MM = 180.0
CELL_CENTERS_X = [TOP_LEFT_X_MM - (row * SQUARE_SIZE_MM + SQUARE_SIZE_MM / 2.0) for row in range(BOARD_SIZE)]
CELL_CENTERS_Y = [TOP_LEFT_Y_MM - (col * SQUARE_SIZE_MM + SQUARE_SIZE_MM / 2.0) for col in range(BOARD_SIZE)]

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

GRAVEYARD_WHITE = [(250.0, -150.0 + i * 60.0) for i in range(12)]
GRAVEYARD_BLACK = [(-250.0, -150.0 + i * 60.0) for i in range(12)]

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
    return CELL_CENTERS_X[row], CELL_CENTERS_Y[col]


def transform_to_robot(world_x: float, world_y: float, perception: PerceptionSystem) -> Tuple[float, float]:
    """
    Transform world coordinates (board frame) to robot frame via homography.
    Uses the H_WORLD_TO_ROBOT matrix from perception module.
    
    Args:
        world_x: X coordinate in world/board frame (mm)
        world_y: Y coordinate in world/board frame (mm)
        perception: PerceptionSystem instance with H_WORLD_TO_ROBOT matrix
    
    Returns:
        (rx, ry): Coordinates in robot frame (mm)
    """
    if perception.H_WORLD_TO_ROBOT is None:
        # Homography not yet computed; fall back to world coords
        log.warning("transform_to_robot: H_WORLD_TO_ROBOT not yet available, using world coords as fallback")
        return world_x, world_y
    
    point = np.array([[[world_x, world_y]]], dtype=np.float32)
    try:
        transformed = cv2.perspectiveTransform(point, perception.H_WORLD_TO_ROBOT)
        rx, ry = transformed[0][0]
        return float(rx), float(ry)
    except Exception as e:
        log.warning("transform_to_robot failed: %s, using world coords as fallback", e)
        return world_x, world_y


def find_nearest_piece(target_piece_id: int, start_row: int, start_col: int, 
                       all_poses: Dict[int, list]) -> Tuple[float, float]:
    """
    Find the actual detected position of a piece, or fall back to grid position.
    
    Args:
        target_piece_id: Piece ID we're trying to move
        start_row: Expected grid row
        start_col: Expected grid column
        all_poses: Dict from perception.get_piece_poses() of {piece_id: [(x1,y1), (x2,y2), ...]}
    
    Returns:
        (x, y): Detected position in world coords, or grid fallback
    """
    expected_x, expected_y = rc_to_world(start_row, start_col)
    
    if target_piece_id not in all_poses or not all_poses[target_piece_id]:
        log.debug("find_nearest_piece: piece %d not detected, using grid fallback (%s,%s)",
                  target_piece_id, start_row, start_col)
        return expected_x, expected_y
    
    # Find the detected coordinate closest to expected square center
    detected_poses = all_poses[target_piece_id]
    best_pose = min(detected_poses, 
                    key=lambda p: math.hypot(p[0] - expected_x, p[1] - expected_y))
    
    distance_mm = math.hypot(best_pose[0] - expected_x, best_pose[1] - expected_y)
    log.debug("find_nearest_piece: piece %d found %.1f mm from grid center", target_piece_id, distance_mm)
    
    return best_pose[0], best_pose[1]

try:
    import serial  # type: ignore
except ImportError:
    serial = None


log = logging.getLogger("l2.main")
CHECKPOINT_PATH = Path(__file__).resolve().parent / "resume_checkpoint.json"
CHECKPOINT_VERSION = 1


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> Path:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    default_dir = Path(__file__).resolve().parent / "logs"
    default_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = Path(log_file) if log_file else (default_dir / "l2_main.log")
    file_handler = RotatingFileHandler(logfile_path, maxBytes=3_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    log.info("logging initialized: level=%s, file=%s", logging.getLevelName(level), logfile_path)
    return logfile_path


@dataclass
class ParsedMove:
    piece: int
    from_cell: Tuple[int, int]
    to_cell: Tuple[int, int]
    promotion: Optional[int]


@dataclass
class RuntimeStats:
    move_count: int = 0
    recoveries: int = 0
    invalid_moves: int = 0
    execution_retries: int = 0
    fallback_moves: int = 0
    opponent_rejects: int = 0
    camera_recovers: int = 0
    verify_failures: int = 0
    total_think_time_s: float = 0.0
    total_loop_time_s: float = 0.0


@dataclass
class PromotionTracker:
    promoted: Dict[Tuple[int, int], int]

    def __init__(self) -> None:
        self.promoted = {}

    def mark(self, row: int, col: int, piece_id: int) -> None:
        self.promoted[(row, col)] = piece_id

    def clear_at(self, row: int, col: int) -> None:
        self.promoted.pop((row, col), None)

    def apply(self, board: np.ndarray) -> np.ndarray:
        out = board.copy()
        for (r, c), pid in self.promoted.items():
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if int(out[r, c]) != EMPTY:
                    out[r, c] = pid
        return out


def _stats_to_dict(stats: RuntimeStats) -> Dict[str, Any]:
    return {
        "move_count": stats.move_count,
        "recoveries": stats.recoveries,
        "invalid_moves": stats.invalid_moves,
        "execution_retries": stats.execution_retries,
        "fallback_moves": stats.fallback_moves,
        "opponent_rejects": stats.opponent_rejects,
        "camera_recovers": stats.camera_recovers,
        "verify_failures": stats.verify_failures,
        "total_think_time_s": stats.total_think_time_s,
        "total_loop_time_s": stats.total_loop_time_s,
    }


def _stats_from_dict(raw: Dict[str, Any]) -> RuntimeStats:
    return RuntimeStats(
        move_count=int(raw.get("move_count", 0)),
        recoveries=int(raw.get("recoveries", 0)),
        invalid_moves=int(raw.get("invalid_moves", 0)),
        execution_retries=int(raw.get("execution_retries", 0)),
        fallback_moves=int(raw.get("fallback_moves", 0)),
        opponent_rejects=int(raw.get("opponent_rejects", 0)),
        camera_recovers=int(raw.get("camera_recovers", 0)),
        verify_failures=int(raw.get("verify_failures", 0)),
        total_think_time_s=float(raw.get("total_think_time_s", 0.0)),
        total_loop_time_s=float(raw.get("total_loop_time_s", 0.0)),
    )


def _checkpoint_write(payload: Dict[str, Any]) -> None:
    payload["version"] = CHECKPOINT_VERSION
    payload["saved_at"] = time.time()
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def _checkpoint_load() -> Optional[Dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        raw = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("failed to load checkpoint: %s", exc)
        return None
    if raw.get("version") != CHECKPOINT_VERSION:
        log.warning("checkpoint version mismatch, ignoring old checkpoint")
        return None
    return raw


def _checkpoint_clear() -> None:
    try:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
    except Exception as exc:
        log.warning("failed to clear checkpoint: %s", exc)


class ArmController:
    def __init__(self) -> None:
        self.base_url = ARM_BASE_URL
        self.serial_conn = None
        self.x = HOME_X
        self.y = HOME_Y
        self.z = HOME_Z
        self.command_counter = 0
        self.serial_fallback_count = 0
        self.feedback_failures = 0

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
        self._session.mount("http://", adapter)

        self._servo_enabled = os.getenv("ROBO_USE_SERVO_GRIPPER", "0") == "1"
        self._servo_id = int(os.getenv("ROBO_GRIPPER_SERVO_ID", "5"))
        self._servo_open_angle = int(os.getenv("ROBO_GRIPPER_OPEN_ANGLE", "90"))
        self._servo_close_angle = int(os.getenv("ROBO_GRIPPER_CLOSE_ANGLE", "20"))
        self._servo_speed = int(os.getenv("ROBO_GRIPPER_SERVO_SPEED", "250"))

        if serial is not None:
            try:
                self.serial_conn = serial.Serial(
                    SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT, dsrdtr=None,
                )
                self.serial_conn.setRTS(False)
                self.serial_conn.setDTR(False)
                log.info("serial opened: %s @ %d (RTS/DTR disabled)", SERIAL_PORT, BAUD_RATE)
            except Exception as exc:
                log.warning("serial unavailable: %s", exc)
        else:
            log.warning("pyserial not installed, serial fallback disabled")

    def close(self) -> None:
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self._session.close()

    def check_connection(self) -> bool:
        payload = {"T": 105}
        for attempt in range(1, MAX_RETRIES + 1):
            if self._send_http(payload, timeout=HTTP_FEEDBACK_TIMEOUT, accept_timeout=False):
                return True
            time.sleep(RETRY_DELAY_BASE * attempt)
        return False

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _send_http(self, payload: Dict, timeout: float, accept_timeout: bool) -> bool:
        self.command_counter += 1
        url = f"{self.base_url}/js?json={quote(json.dumps(payload))}"
        t0 = time.monotonic()
        try:
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
            log.debug("http cmd #%d ok in %.3fs: %s", self.command_counter, time.monotonic() - t0, payload)
            return True
        except requests.exceptions.ReadTimeout:
            if accept_timeout:
                log.debug("HTTP read timeout treated as accepted send for payload: %s", payload)
                return True
            log.warning("HTTP read timeout on payload: %s", payload)
            return False
        except requests.exceptions.ConnectTimeout:
            log.warning("HTTP connect timeout on payload: %s", payload)
            return False
        except requests.exceptions.ConnectionError as exc:
            # The arm's firmware resets the TCP connection on move commands (T:104)
            # after accepting them.  A ConnectionResetError therefore means the
            # command was received and is being executed — treat it as success
            # when we are in fire-and-forget mode (accept_timeout=True).
            if accept_timeout:
                log.debug(
                    "HTTP connection reset treated as accepted send for payload: %s (detail: %s)",
                    payload,
                    exc,
                )
                return True
            log.warning("HTTP connection error for payload %s: %s", payload, exc)
            return False
        except requests.RequestException as exc:
            log.warning("HTTP error for payload %s: %s", payload, exc)
            return False
        except Exception as exc:
            log.exception("HTTP unexpected error for payload %s: %s", payload, exc)
            return False

    def _send_serial_line(self, text: str) -> bool:
        if self.serial_conn is None:
            log.debug("serial send skipped (no serial): %s", text)
            return False
        try:
            self.serial_conn.write(text.encode("utf-8"))
            self.serial_conn.flush()
            log.debug("serial send ok: %s", text)
            return True
        except Exception as exc:
            log.exception("serial write failed for %s: %s", text, exc)
            return False

    def _query_position(self) -> Optional[Tuple[float, float, float]]:
        payload = {"T": 105}
        try:
            url = f"{self.base_url}/js?json={quote(json.dumps(payload))}"
            response = self._session.get(url, timeout=HTTP_FEEDBACK_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            x = float(data.get("x", self.x))
            y = float(data.get("y", self.y))
            z = float(data.get("z", self.z))
            log.debug("feedback position: x=%.2f y=%.2f z=%.2f", x, y, z)
            return x, y, z
        except Exception as exc:
            self.feedback_failures += 1
            log.debug("feedback query failed (%d): %s", self.feedback_failures, exc)
            return None

    def move_xyz(self, x: float, y: float, z: float, speed: int = SPEED_NORMAL) -> bool:
        payload = {"T": 104, "x": float(x), "y": float(y), "z": float(z), "t": int(speed)}
        log.info("arm_move start -> x=%.1f y=%.1f z=%.1f speed=%d", x, y, z, speed)

        sent = False
        for attempt in range(1, MAX_RETRIES + 1):
            log.debug("move send attempt %d/%d: %s", attempt, MAX_RETRIES, payload)
            sent = self._send_http(payload, timeout=HTTP_MOVE_SEND_TIMEOUT, accept_timeout=True)
            if sent:
                break
            time.sleep(RETRY_DELAY_BASE * attempt)

        if not sent:
            # Fallback path for cases where WiFi command channel is unreliable.
            self.serial_fallback_count += 1
            log.warning("HTTP send failed for move, trying serial fallback #%d", self.serial_fallback_count)
            sent = self._send_serial_line(json.dumps(payload))

        if not sent:
            log.error("move command send failed: %s", payload)
            return False

        target = (x, y, z)
        dist = self._distance((self.x, self.y, self.z), target)
        estimated = max(0.8, dist / max(1.0, float(speed)) * 7.5)
        effective_timeout = min(ARM_ARRIVAL_TIMEOUT, max(3.0, estimated * 2.5))
        deadline = time.monotonic() + effective_timeout
        polls = 0
        last_pos = None
        while time.monotonic() < deadline:
            polls += 1
            pos = self._query_position()
            if pos is not None:
                last_pos = pos
                if self._distance(pos, target) <= POSITION_TOLERANCE_MM:
                    self.x, self.y, self.z = x, y, z
                    log.info("arm_move arrived in %d polls", polls)
                    return True
            time.sleep(ARM_POLL_INTERVAL)

        # We do not hard-fail here because some firmware builds do not report feedback reliably.
        self.x, self.y, self.z = x, y, z
        log.warning(
            "arrival poll timeout for target (%.1f, %.1f, %.1f), polls=%d, last_feedback=%s",
            x,
            y,
            z,
            polls,
            last_pos,
        )
        return True

    def linear_move_to(self, target_x: float, target_y: float, target_z: float,
                      step_size_mm: float = 5.0, step_delay_s: float = 0.05,
                      kp_correction: float = 0.8) -> bool:
        """
        Closed-loop linear interpolation with proportional feedback correction.
        
        This method provides stable, accurate motion by:
        1. Reading the actual arm position before starting
        2. Computing waypoints along the straight line to target
        3. For each waypoint, reading feedback and blending: cmd = ideal + Kp * (ideal - actual)
        4. This proportional correction minimizes drift accumulated from mechanical play
        
        Args:
            target_x, target_y, target_z: Target coordinates (mm)
            step_size_mm: Distance between interpolation waypoints (mm)
            step_delay_s: Delay between sending waypoints (s) - tune this for tight tracking
            kp_correction: Proportional gain (0.0=open-loop, 1.0=full correction, 0.8=default)
        
        Returns:
            True if motion completed successfully
        """
        # Read actual start position (real, not cached)
        start_pos = self._query_position()
        if start_pos is None:
            log.debug("[LIN] TESTING/No feedback: simulating straight move.")
            start_pos = (self.x, self.y, self.z)
        
        sx, sy, sz = start_pos
        dx = target_x - sx
        dy = target_y - sy
        dz = target_z - sz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist < 0.5:
            log.debug("[LIN] Already at target (%.1f, %.1f, %.1f)", target_x, target_y, target_z)
            self.x, self.y, self.z = target_x, target_y, target_z
            return True
        
        n_steps = max(1, math.ceil(dist / step_size_mm))
        log.info("[LIN] (%.1f,%.1f,%.1f) → (%.1f,%.1f,%.1f) | %.1f mm / %d steps",
                 sx, sy, sz, target_x, target_y, target_z, dist, n_steps)
        
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            
            # Ideal waypoint along straight line
            ideal_x = sx + alpha * dx
            ideal_y = sy + alpha * dy
            ideal_z = sz + alpha * dz
            
            # Read actual position
            actual_pos = self._query_position()
            if actual_pos is None:
                actual_x, actual_y, actual_z = ideal_x, ideal_y, ideal_z
            else:
                actual_x, actual_y, actual_z = actual_pos
            
            # Proportional correction: cmd = ideal + Kp * (ideal - actual)
            corrected_x = ideal_x + kp_correction * (ideal_x - actual_x)
            corrected_y = ideal_y + kp_correction * (ideal_y - actual_y)
            corrected_z = ideal_z + kp_correction * (ideal_z - actual_z)
            
            # Send corrected waypoint
            payload = {
                "T": 104,
                "x": float(corrected_x),
                "y": float(corrected_y),
                "z": float(corrected_z),
                "t": SPEED_NORMAL
            }
            self._send_http(payload, timeout=HTTP_MOVE_SEND_TIMEOUT, accept_timeout=True)
            
            # Log tracking error
            err = math.sqrt((ideal_x-actual_x)**2 + (ideal_y-actual_y)**2 + (ideal_z-actual_z)**2)
            if err > 15.0:
                log.warning("[LIN] step %d/%d: large tracking error %.1f mm", i, n_steps, err)
            else:
                log.debug("[LIN] step %d/%d: ideal=(%.1f,%.1f,%.1f) actual=(%.1f,%.1f,%.1f) err=%.1f mm",
                         i, n_steps, ideal_x, ideal_y, ideal_z, actual_x, actual_y, actual_z, err)
            
            time.sleep(step_delay_s)
        
        self.x, self.y, self.z = target_x, target_y, target_z
        log.debug("[LIN] Motion complete.")
        return True

    def arm_home(self) -> bool:
        return self.move_xyz(HOME_X, HOME_Y, HOME_Z, SPEED_FAST)

    def magnet_on(self) -> bool:
        if self._servo_enabled:
            self._send_http(
                {"T": 11, "id": self._servo_id, "angle": self._servo_close_angle, "t": self._servo_speed},
                timeout=HTTP_FEEDBACK_TIMEOUT,
                accept_timeout=True,
            )
        ok = self._send_serial_line("1")
        time.sleep(GRIPPER_ACTIVATE_TIME)
        if self.serial_conn is None:
            return True
        log.debug("magnet_on=%s", ok)
        return ok

    def magnet_off(self) -> bool:
        if self._servo_enabled:
            self._send_http(
                {"T": 11, "id": self._servo_id, "angle": self._servo_open_angle, "t": self._servo_speed},
                timeout=HTTP_FEEDBACK_TIMEOUT,
                accept_timeout=True,
            )
        ok = self._send_serial_line("0")
        time.sleep(GRIPPER_ACTIVATE_TIME)
        if self.serial_conn is None:
            return True
        log.debug("magnet_off=%s", ok)
        return ok

    def pick_piece(self, x: float, y: float, speed: int) -> bool:
        return (
            self.move_xyz(x, y, HOME_Z, speed)
            and self.move_xyz(x, y, Z_PICK, speed)
            and self.magnet_on()
            and self.move_xyz(x, y, HOME_Z, speed)
        )

    def place_piece(self, x: float, y: float, speed: int) -> bool:
        ok = (
            self.move_xyz(x, y, HOME_Z, speed)
            and self.move_xyz(x, y, Z_PLACE, speed)
            and self.magnet_off()
            and self.move_xyz(x, y, HOME_Z, speed)
        )
        time.sleep(ARM_SETTLE_TIME)
        return ok


def parse_move(move_str: str) -> Optional[ParsedMove]:
    if not move_str:
        log.error("parse_move got empty move string")
        return None
    match = MOVE_REGEX.match(move_str.strip())
    if not match:
        log.error("parse_move failed for move string: %s", move_str)
        return None

    piece = int(match.group(1))
    src = cell_to_rc(match.group(2))
    dst = cell_to_rc(match.group(3))
    promotion = int(match.group(4)) if match.group(4) else None
    return ParsedMove(piece=piece, from_cell=src, to_cell=dst, promotion=promotion)


def _side_piece_ids(play_white: bool):
    return WHITE_PIECE_IDS if play_white else BLACK_PIECE_IDS


def _is_white_piece(piece_id: int) -> bool:
    return piece_id in WHITE_PIECE_IDS


def _is_black_piece(piece_id: int) -> bool:
    return piece_id in BLACK_PIECE_IDS


def _is_board_plausible(board: np.ndarray) -> bool:
    # A plausible board should have at most one white king and one black king.
    white_king_count = int(np.count_nonzero(board == 5))
    black_king_count = int(np.count_nonzero(board == 10))
    return white_king_count <= 1 and black_king_count <= 1


def validate_move_for_side(board: np.ndarray, move: ParsedMove, play_white: bool) -> Tuple[bool, str]:
    fr, fc = move.from_cell
    tr, tc = move.to_cell

    if not (0 <= fr < BOARD_SIZE and 0 <= fc < BOARD_SIZE and 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE):
        return False, "cell out of board"

    if (fr, fc) == (tr, tc):
        return False, "source and target are identical"

    own_ids = _side_piece_ids(play_white)
    src = int(board[fr, fc])
    dst = int(board[tr, tc])

    if src == EMPTY:
        return False, "source square empty"
    if src not in own_ids:
        return False, f"source piece {src} is not ours"
    if dst in own_ids:
        return False, f"target occupied by our piece {dst}"

    if move.promotion is not None and move.promotion not in own_ids:
        return False, f"promotion piece {move.promotion} not valid for side"

    return True, "ok"


def _board_change_count(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(before != after))


def _board_delta_to_expected(observed: np.ndarray, expected: np.ndarray) -> int:
    return int(np.count_nonzero(observed != expected))


def accept_opponent_board_change(before: np.ndarray, after: np.ndarray) -> Tuple[bool, str]:
    if after is None:
        return False, "missing board"
    if not _is_board_plausible(after):
        return False, "implausible board state"

    delta = _board_change_count(before, after)
    if delta == 0:
        return False, "no board change"
    if delta > 8:
        return False, f"too many changed cells: {delta}"

    return True, "ok"


def _recover_camera_stream(perception: PerceptionSystem, stats: RuntimeStats) -> bool:
    stats.recoveries += 1
    stats.camera_recovers += 1
    perception.close()
    return perception.start_background()


def _reconcile_post_move_board(
    perception: PerceptionSystem,
    expected: np.ndarray,
    fallback_board: np.ndarray,
    stats: RuntimeStats,
    verify_retries: int = 2,
) -> np.ndarray:
    if perception.verify_board(expected):
        return expected

    stats.verify_failures += 1
    best = fallback_board
    best_delta = _board_delta_to_expected(best, expected)

    for _ in range(max(1, verify_retries)):
        latest = perception.wait_for_stable_board(timeout=4.0)
        if latest is None or not _is_board_plausible(latest):
            continue
        delta = _board_delta_to_expected(latest, expected)
        if delta < best_delta:
            best = latest
            best_delta = delta
        if delta == 0:
            break

    if best_delta > 3:
        log.warning("board verify mismatch (delta=%d), using expected state", best_delta)
        stats.recoveries += 1
        return expected

    if best_delta > 0:
        stats.recoveries += 1
    log.warning("board verification mismatch resolved with delta=%d", best_delta)
    return best


def apply_move(board: np.ndarray, move: ParsedMove) -> Tuple[np.ndarray, int]:
    updated = board.copy()
    fr, fc = move.from_cell
    tr, tc = move.to_cell
    captured = int(updated[tr, tc])
    updated[fr, fc] = EMPTY
    updated[tr, tc] = move.promotion if move.promotion is not None else move.piece
    return updated, captured


def board_to_text(board: np.ndarray) -> str:
    rows = []
    rows.append("    A   B   C   D   E   F")
    for r in range(BOARD_SIZE - 1, -1, -1):
        chunks = []
        for c in range(BOARD_SIZE):
            pid = int(board[r, c])
            chunks.append(f"{PIECE_NAMES.get(pid, '..'):>2}")
        rows.append(f"{r + 1} | " + " ".join(chunks))
    return "\n".join(rows)


def execute_move(
    arm: ArmController,
    board: np.ndarray,
    move: ParsedMove,
    white_grave_idx: int,
    black_grave_idx: int,
    perception: PerceptionSystem,
    speed: int,
) -> Tuple[np.ndarray, int, int, bool]:
    """
    Execute a move with advanced piece detection and closed-loop motion.
    
    This function combines detected piece positions (from perception.get_piece_poses)
    with closed-loop feedback motion (linear_move_to) for robust execution.
    
    Args:
        arm: ArmController instance
        board: Current board state
        move: ParsedMove to execute
        white_grave_idx, black_grave_idx: Graveyard indices
        perception: PerceptionSystem with detected piece poses and homography
        speed: Motion speed parameter
    
    Returns:
        (new_board, white_grave_idx, black_grave_idx, success)
    """
    fr, fc = move.from_cell
    tr, tc = move.to_cell
    if (fr, fc) == (tr, tc):
        log.error("invalid no-op move: %s -> %s", rc_to_label(fr, fc), rc_to_label(tr, tc))
        return board, white_grave_idx, black_grave_idx, False

    source_piece = int(board[fr, fc])
    if source_piece == EMPTY:
        log.error("invalid move source is empty at %s", rc_to_label(fr, fc))
        return board, white_grave_idx, black_grave_idx, False

    effective_piece = move.piece
    if source_piece != move.piece:
        log.warning(
            "engine piece mismatch at %s: move piece=%d board piece=%d; using board piece",
            rc_to_label(fr, fc),
            move.piece,
            source_piece,
        )
        effective_piece = source_piece

    destination_piece = int(board[tr, tc])
    if destination_piece != EMPTY:
        same_side_capture = (
            (_is_white_piece(effective_piece) and _is_white_piece(destination_piece))
            or (_is_black_piece(effective_piece) and _is_black_piece(destination_piece))
        )
        if same_side_capture:
            log.error(
                "invalid self-capture: mover=%d target=%d at %s",
                effective_piece,
                destination_piece,
                rc_to_label(tr, tc),
            )
            return board, white_grave_idx, black_grave_idx, False

    log.info(
        "execute_move piece=%d from=%s to=%s promotion=%s",
        effective_piece,
        rc_to_label(*move.from_cell),
        rc_to_label(*move.to_cell),
        move.promotion,
    )
    effective_move = ParsedMove(
        piece=effective_piece,
        from_cell=move.from_cell,
        to_cell=move.to_cell,
        promotion=move.promotion,
    )
    expected, captured = apply_move(board, effective_move)

    # Get detected piece poses for accurate positioning
    all_poses = perception.get_piece_poses()
    
    # Find actual piece positions (detected or fallback to grid)
    from_world_x, from_world_y = find_nearest_piece(effective_piece, fr, fc, all_poses)
    from_robot_x, from_robot_y = transform_to_robot(from_world_x, from_world_y, perception)
    
    to_world_x, to_world_y = rc_to_world(tr, tc)
    to_robot_x, to_robot_y = transform_to_robot(to_world_x, to_world_y, perception)

    # Handle captures with detected position for captured piece
    if captured != EMPTY:
        log.info("capture detected at %s: piece=%d", rc_to_label(*move.to_cell), captured)
        captured_is_white = captured in WHITE_PIECE_IDS
        
        # Find the actual position of the piece to capture
        captured_world_x, captured_world_y = find_nearest_piece(captured, tr, tc, all_poses)
        captured_robot_x, captured_robot_y = transform_to_robot(captured_world_x, captured_world_y, perception)
        
        if captured_is_white:
            slot_idx = min(white_grave_idx, len(GRAVEYARD_WHITE) - 1)
            slot = GRAVEYARD_WHITE[slot_idx]
        else:
            slot_idx = min(black_grave_idx, len(GRAVEYARD_BLACK) - 1)
            slot = GRAVEYARD_BLACK[slot_idx]

        # Pick captured piece with closed-loop motion
        if not arm.linear_move_to(captured_robot_x, captured_robot_y, Z_SAFE):
            log.error("failed while moving to captured piece")
            return board, white_grave_idx, black_grave_idx, False
        if not arm.linear_move_to(captured_robot_x, captured_robot_y, Z_PICK):
            log.error("failed while lowering to captured piece")
            return board, white_grave_idx, black_grave_idx, False
        if not arm.magnet_on():
            log.error("failed while energizing magnet for capture")
            return board, white_grave_idx, black_grave_idx, False
        time.sleep(GRIPPER_ACTIVATE_TIME)
        if not arm.linear_move_to(captured_robot_x, captured_robot_y, Z_SAFE):
            log.error("failed while lifting captured piece")
            return board, white_grave_idx, black_grave_idx, False
        
        # Place in graveyard
        if not arm.linear_move_to(slot[0], slot[1], Z_SAFE):
            log.error("failed while moving to graveyard")
            return board, white_grave_idx, black_grave_idx, False
        if not arm.linear_move_to(slot[0], slot[1], Z_PLACE):
            log.error("failed while lowering to graveyard")
            return board, white_grave_idx, black_grave_idx, False
        if not arm.magnet_off():
            log.error("failed while deenergizing magnet at graveyard")
            return board, white_grave_idx, black_grave_idx, False
        time.sleep(GRIPPER_ACTIVATE_TIME)
        if not arm.linear_move_to(slot[0], slot[1], Z_SAFE):
            log.error("failed while lifting from graveyard")
            return board, white_grave_idx, black_grave_idx, False
        
        if captured_is_white:
            white_grave_idx += 1
        else:
            black_grave_idx += 1

    # Move piece from source to destination with closed-loop motion and detected position
    if not arm.linear_move_to(from_robot_x, from_robot_y, Z_SAFE):
        log.error("failed while moving to piece")
        return board, white_grave_idx, black_grave_idx, False
    if not arm.linear_move_to(from_robot_x, from_robot_y, Z_PICK):
        log.error("failed while lowering to piece")
        return board, white_grave_idx, black_grave_idx, False
    if not arm.magnet_on():
        log.error("failed while energizing magnet")
        return board, white_grave_idx, black_grave_idx, False
    time.sleep(GRIPPER_ACTIVATE_TIME)
    if not arm.linear_move_to(from_robot_x, from_robot_y, Z_SAFE):
        log.error("failed while lifting piece")
        return board, white_grave_idx, black_grave_idx, False

    # Place at destination
    if not arm.linear_move_to(to_robot_x, to_robot_y, Z_SAFE):
        log.error("failed while moving to destination")
        return board, white_grave_idx, black_grave_idx, False
    if not arm.linear_move_to(to_robot_x, to_robot_y, Z_PLACE):
        log.error("failed while lowering to destination")
        return board, white_grave_idx, black_grave_idx, False
    if not arm.magnet_off():
        log.error("failed while deenergizing magnet at destination")
        return board, white_grave_idx, black_grave_idx, False
    time.sleep(GRIPPER_ACTIVATE_TIME)
    if not arm.linear_move_to(to_robot_x, to_robot_y, Z_SAFE):
        log.error("failed while lifting from destination")
        return board, white_grave_idx, black_grave_idx, False
    
    # Return home to rest position
    if not arm.arm_home():
        log.warning("failed while homing arm after move")
    
    return expected, white_grave_idx, black_grave_idx, True


def calibration_wizard(arm: ArmController) -> None:
    checks = [
        ("home", (HOME_X, HOME_Y, HOME_Z)),
        ("A1", (*rc_to_world(0, 0), HOME_Z)),
        ("F1", (*rc_to_world(0, 5), HOME_Z)),
        ("A6", (*rc_to_world(5, 0), HOME_Z)),
        ("F6", (*rc_to_world(5, 5), HOME_Z)),
    ]
    for label, (x, y, z) in checks:
        input(f"Press Enter to move to {label} ({x:.1f}, {y:.1f}, {z:.1f})...")
        ok = arm.move_xyz(x, y, z, SPEED_NORMAL)
        print(f"{label}: {'OK' if ok else 'FAILED'}")


def choose_move_speed(move_count: int, think_time_s: float) -> int:
    # Openings prioritize placement precision; speed up when game gets tactical.
    if move_count < 3:
        return SPEED_SLOW
    if move_count >= 18 or think_time_s > 8.0:
        return SPEED_FAST
    return SPEED_NORMAL


def request_engine_move(board: np.ndarray, play_white: bool) -> Tuple[Optional[str], bool]:
    # Keep game.py untouched: sanitize board here before engine calls.
    safe_board = np.array(board, dtype=int)
    if safe_board.shape != (6, 6):
        fixed = np.zeros((6, 6), dtype=int)
        rows = min(6, safe_board.shape[0]) if safe_board.ndim >= 1 else 0
        cols = min(6, safe_board.shape[1]) if safe_board.ndim >= 2 else 0
        if rows > 0 and cols > 0:
            fixed[:rows, :cols] = safe_board[:rows, :cols]
        safe_board = fixed
    safe_board = np.clip(safe_board, 0, 10)

    try:
        best = game.get_best_move(safe_board, play_white)
        if best:
            parsed = parse_move(best)
            if parsed is not None:
                ok, reason = validate_move_for_side(safe_board, parsed, play_white)
                if ok:
                    return best, False
                log.warning("get_best_move rejected: %s | %s", best, reason)
        log.warning("get_best_move returned invalid move: %s", best)
    except Exception as exc:
        log.exception("get_best_move raised: %s", exc)

    turn_value = 1 if play_white else 0
    try:
        fallback = game.get_move(safe_board, turn_value)
        if fallback:
            parsed = parse_move(fallback)
            if parsed is not None:
                ok, reason = validate_move_for_side(safe_board, parsed, play_white)
                if ok:
                    log.warning("using get_move fallback")
                    return fallback, True
                log.warning("get_move fallback rejected: %s | %s", fallback, reason)
    except Exception as exc:
        log.exception("get_move fallback raised: %s", exc)

    return None, False


def _is_game_over(board: np.ndarray) -> bool:
    has_wk = bool(np.any(board == 5))
    has_bk = bool(np.any(board == 10))
    if not has_wk:
        log.info("game over: white king missing")
        return True
    if not has_bk:
        log.info("game over: black king missing")
        return True
    return False


def _wait_and_accept_opponent_board(
    perception: PerceptionSystem,
    board: np.ndarray,
    manual_mode: bool,
    stats: RuntimeStats,
    promotions: PromotionTracker,
) -> Optional[np.ndarray]:
    if manual_mode:
        input("Press Enter after opponent move is complete...")
        changed = perception.wait_for_stable_board(timeout=30.0)
    else:
        changed = perception.wait_for_board_change(board)

    if changed is None:
        log.warning("timeout waiting for opponent move, attempting camera recovery")
        if _recover_camera_stream(perception, stats):
            changed = perception.wait_for_board_change(board, timeout=45.0)
    if changed is None:
        return None

    changed = promotions.apply(changed)
    accept, reason = accept_opponent_board_change(board, changed)
    if accept:
        return changed

    stats.opponent_rejects += 1
    log.warning("rejected opponent board change: %s", reason)
    changed_retry = perception.wait_for_stable_board(timeout=4.0)
    if changed_retry is None:
        return None

    changed_retry = promotions.apply(changed_retry)
    accept, reason = accept_opponent_board_change(board, changed_retry)
    if not accept:
        stats.opponent_rejects += 1
        log.warning("rejected retry opponent board change: %s", reason)
        return None
    return changed_retry


def run_game(play_white: bool, manual_mode: bool = False, max_moves: int = 0, resume: bool = True) -> int:
    log.info("run_game start: play_white=%s, arm_ip=%s, serial_port=%s", play_white, ARM_IP, SERIAL_PORT)
    arm = ArmController()
    perception = PerceptionSystem()
    stats = RuntimeStats()
    promotions = PromotionTracker()
    clear_checkpoint_on_exit = False

    loaded_checkpoint = _checkpoint_load() if resume else None

    def save_checkpoint(board_state: np.ndarray, white_idx: int, black_idx: int, next_action: str) -> None:
        payload = {
            "play_white": play_white,
            "manual_mode": manual_mode,
            "max_moves": max_moves,
            "next_action": next_action,
            "board": board_state.tolist(),
            "white_grave_idx": int(white_idx),
            "black_grave_idx": int(black_idx),
            "stats": _stats_to_dict(stats),
            "promotions": [
                {"r": int(r), "c": int(c), "piece": int(pid)}
                for (r, c), pid in promotions.promoted.items()
            ],
        }
        _checkpoint_write(payload)

    if not arm.check_connection():
        log.error("arm preflight failed")
        arm.close()
        return 2

    if not perception.start_background():
        log.error("camera connection failed")
        arm.close()
        return 2

    board = None
    white_grave_idx = 0
    black_grave_idx = 0
    next_action = "our_turn" if play_white else "wait_opponent"

    if loaded_checkpoint is not None:
        if bool(loaded_checkpoint.get("play_white", play_white)) != play_white:
            log.warning("checkpoint side mismatch; ignoring checkpoint")
            loaded_checkpoint = None
        elif bool(loaded_checkpoint.get("manual_mode", manual_mode)) != manual_mode:
            log.warning("checkpoint mode mismatch; ignoring checkpoint")
            loaded_checkpoint = None

    if loaded_checkpoint is not None:
        try:
            cp_board = np.array(loaded_checkpoint.get("board", []), dtype=int)
            if cp_board.shape != (BOARD_SIZE, BOARD_SIZE):
                raise ValueError(f"bad checkpoint board shape: {cp_board.shape}")
            if not _is_board_plausible(cp_board):
                raise ValueError("checkpoint board not plausible")

            white_grave_idx = int(loaded_checkpoint.get("white_grave_idx", 0))
            black_grave_idx = int(loaded_checkpoint.get("black_grave_idx", 0))
            next_action = str(loaded_checkpoint.get("next_action", next_action))
            stats = _stats_from_dict(loaded_checkpoint.get("stats", {}))
            promotions.promoted.clear()
            for row in loaded_checkpoint.get("promotions", []):
                promotions.mark(int(row["r"]), int(row["c"]), int(row["piece"]))

            live_board = perception.wait_for_stable_board(timeout=8.0)
            if live_board is not None and _is_board_plausible(live_board):
                live_board = promotions.apply(live_board)
                delta = _board_change_count(cp_board, live_board)
                if delta <= 6:
                    board = live_board
                    log.info("resumed from checkpoint with live board sync (delta=%d)", delta)
                else:
                    board = cp_board
                    log.warning("live board differs too much from checkpoint (delta=%d), using checkpoint board", delta)
            else:
                board = cp_board
                log.info("resumed from checkpoint without live sync")
        except Exception as exc:
            log.warning("failed to restore checkpoint, starting fresh: %s", exc)
            board = None

    if board is None:
        board = perception.wait_for_stable_board()
        if board is None:
            log.error("could not read stable initial board")
            perception.close()
            arm.close()
            return 2

    if not _is_board_plausible(board):
        log.warning("initial board not plausible, retrying stabilization")
        retry_board = perception.wait_for_stable_board(timeout=10.0)
        if retry_board is not None and _is_board_plausible(retry_board):
            board = retry_board
        else:
            log.error("initial board remained implausible")
            perception.close()
            arm.close()
            return 2

    board = promotions.apply(board)

    log.info("active board:\n%s", board_to_text(board))
    log.info("checkpoint file: %s", CHECKPOINT_PATH)
    save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)

    try:
        while True:
            if max_moves > 0 and stats.move_count >= max_moves:
                log.info("reached max_moves=%d, stopping cleanly", max_moves)
                clear_checkpoint_on_exit = True
                break

            loop_t0 = time.monotonic()
            if next_action == "wait_opponent":
                changed = _wait_and_accept_opponent_board(perception, board, manual_mode, stats, promotions)
                if changed is None:
                    log.error("timeout waiting for opponent move")
                    break

                board = changed
                if _is_game_over(board):
                    clear_checkpoint_on_exit = True
                    break
                next_action = "our_turn"
                save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)
                continue

            my_board = board

            think_t0 = time.monotonic()
            move_str, used_fallback = request_engine_move(my_board, play_white)
            think_elapsed = time.monotonic() - think_t0
            log.info("engine think time: %.3fs", think_elapsed)
            stats.total_think_time_s += think_elapsed
            if not move_str:
                stats.invalid_moves += 1
                log.error("engine failed to provide a valid move")
                break
            if used_fallback:
                stats.fallback_moves += 1

            parsed = parse_move(move_str)
            if parsed is None:
                stats.invalid_moves += 1
                log.error("engine returned invalid move: %s", move_str)
                break

            valid, reason = validate_move_for_side(board, parsed, play_white)
            if not valid:
                stats.invalid_moves += 1
                log.error("engine move rejected: %s | %s", move_str, reason)
                break

            speed = choose_move_speed(stats.move_count, think_elapsed)
            log.info("engine move %d: %s | speed=%d", stats.move_count + 1, move_str, speed)
            expected, white_grave_idx, black_grave_idx, ok = execute_move(
                arm,
                board,
                parsed,
                white_grave_idx,
                black_grave_idx,
                perception,
                speed=speed,
            )
            if not ok:
                stats.execution_retries += 1
                log.warning("arm execution failed, homing and retrying once: %s", move_str)
                arm.arm_home()
                expected, white_grave_idx, black_grave_idx, ok = execute_move(
                    arm,
                    board,
                    parsed,
                    white_grave_idx,
                    black_grave_idx,
                    perception,
                    speed=SPEED_FAST,
                )
                if not ok:
                    log.error("arm execution failed for move after retry: %s", move_str)
                    break

            board = _reconcile_post_move_board(
                perception=perception,
                expected=expected,
                fallback_board=board,
                stats=stats,
                verify_retries=2,
            )

            src_rc = parsed.from_cell
            dst_rc = parsed.to_cell
            carried_promo = promotions.promoted.get(src_rc)
            promotions.clear_at(src_rc[0], src_rc[1])
            if parsed.promotion is not None:
                promotions.mark(dst_rc[0], dst_rc[1], parsed.promotion)
            elif carried_promo is not None:
                # Move existing promoted-piece identity with the piece.
                promotions.mark(dst_rc[0], dst_rc[1], int(carried_promo))
            board = promotions.apply(board)

            log.info("board after move:\n%s", board_to_text(board))
            stats.move_count += 1
            loop_elapsed = time.monotonic() - loop_t0
            stats.total_loop_time_s += loop_elapsed
            log.info("loop iteration time: %.3fs", loop_elapsed)
            if _is_game_over(board):
                clear_checkpoint_on_exit = True
                break
            next_action = "wait_opponent"
            save_checkpoint(board, white_grave_idx, black_grave_idx, next_action)

    except KeyboardInterrupt:
        log.info("stopped by user")
    except Exception as exc:
        log.exception("unhandled exception in run_game: %s", exc)
        return 3
    finally:
        arm.arm_home()
        perception.close()
        arm.close()
        if clear_checkpoint_on_exit:
            _checkpoint_clear()
            log.info("checkpoint cleared after clean stop")
        log.info(
            "run_game cleanup done | commands=%d serial_fallbacks=%d feedback_failures=%d moves=%d recoveries=%d invalid_moves=%d retries=%d fallback_moves=%d opponent_rejects=%d camera_recovers=%d verify_failures=%d avg_think=%.3fs avg_loop=%.3fs",
            arm.command_counter,
            arm.serial_fallback_count,
            arm.feedback_failures,
            stats.move_count,
            stats.recoveries,
            stats.invalid_moves,
            stats.execution_retries,
            stats.fallback_moves,
            stats.opponent_rejects,
            stats.camera_recovers,
            stats.verify_failures,
            stats.total_think_time_s / max(1, stats.move_count),
            stats.total_loop_time_s / max(1, stats.move_count),
        )

    return 0


def run_preflight(check_frame: bool = False) -> int:
    """Run a startup connectivity preflight and exit.

    Returns 0 on success, non-zero when any mandatory check fails.
    """
    arm = ArmController()
    perception = PerceptionSystem()
    arm_ok = False
    cam_ok = False
    frame_ok: Optional[bool] = None
    try:
        arm_ok = arm.check_connection()
        cam_ok = perception.connect_camera()
        if check_frame and cam_ok:
            frame_ok = perception.recv_frame() is not None

        log.info(
            "preflight result | arm_ok=%s cam_ok=%s frame_ok=%s arm=%s camera_candidates=%s active_camera=%s",
            arm_ok,
            cam_ok,
            frame_ok,
            arm.base_url,
            getattr(perception, "_camera_candidates", None),
            getattr(perception, "_active_camera", None),
        )
        print(
            "PREFLIGHT",
            f"arm_ok={arm_ok}",
            f"cam_ok={cam_ok}",
            f"frame_ok={frame_ok}",
            f"arm={arm.base_url}",
            f"active_camera={getattr(perception, '_active_camera', None)}",
        )
        if check_frame:
            return 0 if (arm_ok and cam_ok and frame_ok) else 2
        return 0 if (arm_ok and cam_ok) else 2
    finally:
        perception.close()
        arm.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="RoboGambit Team L2 controller")
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--white", action="store_true", help="play as white")
    side.add_argument("--black", action="store_true", help="play as black")
    parser.add_argument("--arm-ip", type=str, default="", help="override arm controller IP")
    parser.add_argument("--arm-port", type=int, default=0, help="override arm controller port")
    parser.add_argument("--camera-ip", type=str, default="", help="override camera stream IP")
    parser.add_argument("--camera-port", type=int, default=0, help="override camera stream port")
    parser.add_argument(
        "--camera-candidates",
        type=str,
        default="",
        help="comma separated camera endpoints host:port,host:port",
    )
    parser.add_argument("--calibrate", action="store_true", help="run calibration wizard")
    parser.add_argument("--preflight", action="store_true", help="run arm+camera connectivity preflight and exit")
    parser.add_argument("--preflight-frame", action="store_true", help="when used with --preflight, also require one camera frame")
    parser.add_argument("--manual", action="store_true", help="manual opponent move trigger")
    parser.add_argument("--max-moves", type=int, default=0, help="stop after N of our own moves (0=unlimited)")
    parser.add_argument("--fresh", action="store_true", help="ignore and clear saved checkpoint before starting")
    parser.add_argument("--no-resume", action="store_true", help="run without loading checkpoint")
    parser.add_argument("--debug", action="store_true", help="enable verbose debug logging")
    parser.add_argument("--log-file", type=str, default="", help="custom path for log file")
    args = parser.parse_args()

    setup_logging(debug=args.debug, log_file=args.log_file or None)

    global ARM_IP, ARM_PORT, ARM_BASE_URL
    if args.arm_ip:
        ARM_IP = args.arm_ip.strip()
    if args.arm_port > 0:
        ARM_PORT = int(args.arm_port)
    ARM_BASE_URL = f"http://{ARM_IP}:{ARM_PORT}"

    if args.camera_ip:
        perception_module.CAMERA_IP = args.camera_ip.strip()
    if args.camera_port > 0:
        perception_module.CAMERA_PORT = int(args.camera_port)
    if args.camera_candidates:
        perception_module.CANDIDATE_CAMERA_ENDPOINTS = args.camera_candidates.strip()

    log.info(
        "main args: white=%s black=%s calibrate=%s manual=%s max_moves=%d fresh=%s no_resume=%s debug=%s arm=%s:%d camera=%s:%d candidates=%s",
        args.white,
        args.black,
        args.calibrate,
        args.manual,
        args.max_moves,
        args.fresh,
        args.no_resume,
        args.debug,
        ARM_IP,
        ARM_PORT,
        perception_module.CAMERA_IP,
        perception_module.CAMERA_PORT,
        perception_module.CANDIDATE_CAMERA_ENDPOINTS,
    )

    if args.fresh:
        _checkpoint_clear()
        log.info("cleared checkpoint due to --fresh")

    if args.preflight:
        return run_preflight(check_frame=args.preflight_frame)

    if not args.white and not args.black:
        args.white = True

    arm = ArmController()
    try:
        if args.calibrate:
            calibration_wizard(arm)
            return 0
    finally:
        if args.calibrate:
            arm.close()

    def _sigint_handler(sig_num, frame):
        log.warning("SIGINT received, stopping now")
        raise KeyboardInterrupt()

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    resume_enabled = (not args.no_resume) and (not args.fresh)
    try:
        return run_game(
        play_white=args.white,
        manual_mode=args.manual,
        max_moves=max(0, args.max_moves),
        resume=resume_enabled,
        )
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
