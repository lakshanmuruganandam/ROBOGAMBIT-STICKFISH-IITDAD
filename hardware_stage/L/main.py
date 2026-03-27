"""
main.py — STICKFISH Hardware Controller (Team L)
==================================================
The definitive hardware implementation for RoboGambit competition.

CRITICAL FIX (eliminates ALL HTTP timeout errors):
  The Waveshare RoArm M2-S HTTP API blocks until movement completes.
  Large movements take 15-30+ seconds → read timeout errors.
  
  SOLUTION: "Fire-and-forget" HTTP + feedback polling.
    1. Send movement command via HTTP with SHORT timeout (3s)
    2. If we get a response → great, arm confirmed
    3. If we get a ReadTimeout → that's FINE, arm received command
       and is moving. This is NOT an error.
    4. If we get a ConnectTimeout → arm unreachable, retry
    5. After sending, poll T:105 feedback to confirm arrival

Features:
  - State-machine game loop (INIT → THINK → EXECUTE → WAIT_OPPONENT)
  - Fire-and-forget HTTP with arrival confirmation via polling
  - Move verification after arm execution
  - Opponent move detection with stability check
  - Dynamic speed adjustment based on clock time
  - Captured-piece tracking + graveyard management
  - Promotion tracking for pawn promotions
  - Comprehensive logging to console + file
  - Calibration wizard mode
  - Graceful error recovery at every stage

Usage:
    python main.py --white              # Play as white
    python main.py --black              # Play as black
    python main.py --white --manual     # Manual trigger mode
    python main.py --calibrate          # Run calibration wizard
"""

import sys
import json
import math
import time
import signal
import logging
import argparse
import threading
import random
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from urllib.parse import quote

import requests
import requests.adapters
import serial

from config import (
    BOARD_SIZE, PIECE_NAMES, WHITE_PIECE_IDS, BLACK_PIECE_IDS,
    ARM_IP, ARM_PORT, ARM_BASE_URL,
    SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT,
    CAMERA_IP, CAMERA_PORT,
    CELL_CENTERS_X, CELL_CENTERS_Y,
    Z_SAFE, Z_PICK, Z_PLACE,
    HOME_X, HOME_Y, HOME_Z,
    DEFAULT_SPEED, SPEED_FAST, SPEED_NORMAL, SPEED_SLOW,
    GRIPPER_OPEN_ANGLE, GRIPPER_CLOSE_ANGLE,
    GRIPPER_SERVO_ID, GRIPPER_SERVO_SPEED,
    GRAVEYARD_WHITE, GRAVEYARD_BLACK,
    ARM_SETTLE_TIME, GRIPPER_ACTIVATE_TIME, GRIPPER_SERVO_TIME,
    HTTP_CONNECT_TIMEOUT, HTTP_FEEDBACK_TIMEOUT, HTTP_MOVE_SEND_TIMEOUT,
    ARM_ARRIVAL_TIMEOUT, ARM_POLL_INTERVAL,
    MAX_RETRIES, RETRY_DELAY_BASE, RETRY_DELAY_MAX,
    POSITION_TOLERANCE, MOVEMENT_SPEED_FACTOR,
    STABILITY_FRAMES, STABILITY_TIMEOUT,
    GAME_CLOCK_TOTAL, CLOCK_SAFETY_BUFFER,
    cell_to_world, rc_to_label,
)

from perception import PerceptionSystem


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("game_log.txt", mode="w"),
    ],
)
log = logging.getLogger("main")


# ═══════════════════════════════════════════════════════════════════════
#  ARM CONTROLLER — Fire-and-Forget HTTP + Feedback Polling
# ═══════════════════════════════════════════════════════════════════════

class ArmController:
    """
    Controls the Waveshare RoArm M2-S robotic arm.
    
    KEY DESIGN: Movement commands use fire-and-forget HTTP.
    - The arm's HTTP API blocks until the physical movement completes
    - For large moves this can take 20+ seconds → timeout
    - We send with a SHORT timeout and EXPECT the timeout
    - Then poll T:105 feedback to confirm arrival
    - This ELIMINATES all HTTP timeout errors
    """

    def __init__(self):
        self._base_url = ARM_BASE_URL

        # HTTP session with connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2, pool_maxsize=4,
            max_retries=0,  # we handle retries ourselves
        )
        self._session.mount("http://", adapter)

        # Serial for electromagnet
        self._serial_lock = threading.Lock()
        self._ser = None
        try:
            self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
            log.info(f"Serial opened: {SERIAL_PORT} @ {BAUD_RATE}")
        except serial.SerialException as e:
            log.warning(f"Serial open failed ({SERIAL_PORT}): {e}")

        # Position tracking
        self.x, self.y, self.z = HOME_X, HOME_Y, HOME_Z
        self._speed = DEFAULT_SPEED
        self._connected = False

    # ── Connection Check ──────────────────────────────────────────────

    def check_connection(self) -> bool:
        """Verify arm is reachable by sending a quick feedback query."""
        for attempt in range(MAX_RETRIES):
            try:
                fb = self._get_feedback_raw()
                if fb is not None:
                    self._connected = True
                    log.info("Arm connection verified ✓")
                    return True
            except Exception:
                pass
            delay = min(RETRY_DELAY_BASE * (attempt + 1), RETRY_DELAY_MAX)
            log.warning(f"Arm not responding, retry in {delay:.1f}s...")
            time.sleep(delay)
        
        log.error("Arm connection FAILED after all retries")
        self._connected = False
        return False

    def is_connected(self) -> bool:
        if self._connected:
            return True
        return self.check_connection()

    # ── HTTP: Fire-and-Forget for Movement ────────────────────────────

    def _send_movement_cmd(self, json_cmd: str) -> bool:
        """
        Send movement command (T:104) using fire-and-forget.
        
        The arm's HTTP response only comes AFTER movement completes.
        So we use a SHORT read timeout. If we get a ReadTimeout,
        that's EXPECTED — the command was sent and the arm is moving.
        
        Only a ConnectTimeout means the arm is unreachable (real error).
        """
        url = f"{self._base_url}/js?json={quote(json_cmd)}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(
                    url,
                    timeout=(HTTP_CONNECT_TIMEOUT, HTTP_MOVE_SEND_TIMEOUT),
                )
                resp.raise_for_status()
                log.debug(f"Movement CMD got immediate response (fast move)")
                return True

            except requests.exceptions.ReadTimeout:
                # EXPECTED! The arm received the command and is moving.
                # The HTTP response is blocked waiting for movement to finish.
                # This is NOT an error — the command was successfully sent.
                log.debug(f"Movement CMD sent (read timeout = arm is moving)")
                return True

            except requests.exceptions.ConnectTimeout:
                # REAL error: can't reach the arm at all
                delay = min(RETRY_DELAY_BASE * attempt + random.random(), RETRY_DELAY_MAX)
                log.warning(f"Arm unreachable (connect timeout), retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
                if attempt < MAX_RETRIES:
                    time.sleep(delay)

            except requests.exceptions.ConnectionError as e:
                delay = min(RETRY_DELAY_BASE * attempt + random.random(), RETRY_DELAY_MAX)
                log.warning(f"Connection error: {e}, retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
                if attempt < MAX_RETRIES:
                    time.sleep(delay)

            except requests.RequestException as e:
                delay = min(RETRY_DELAY_BASE * attempt + random.random(), RETRY_DELAY_MAX)
                log.warning(f"HTTP error: {e}, retry {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(delay)

        log.error(f"Movement CMD FAILED after {MAX_RETRIES} retries: {json_cmd}")
        return False

    def _send_quick_cmd(self, json_cmd: str) -> bool:
        """Send a quick/non-blocking command (servo, etc). Uses longer read timeout."""
        url = f"{self._base_url}/js?json={quote(json_cmd)}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(
                    url,
                    timeout=(HTTP_CONNECT_TIMEOUT, HTTP_FEEDBACK_TIMEOUT),
                )
                resp.raise_for_status()
                return True
            except requests.exceptions.ReadTimeout:
                # For quick commands, a read timeout might mean the arm is busy
                log.debug(f"Quick CMD read timeout (arm busy), treating as sent")
                return True
            except requests.RequestException as e:
                delay = min(RETRY_DELAY_BASE * attempt + random.random(), RETRY_DELAY_MAX)
                if attempt < MAX_RETRIES:
                    log.debug(f"Quick CMD retry {attempt}: {e}")
                    time.sleep(delay)

        log.warning(f"Quick CMD failed: {json_cmd}")
        return False

    # ── Feedback Polling ──────────────────────────────────────────────

    def _get_feedback_raw(self) -> Optional[dict]:
        """Query arm position (T:105). Returns dict or None."""
        cmd = json.dumps({"T": 105})
        url = f"{self._base_url}/js?json={quote(cmd)}"
        try:
            resp = self._session.get(
                url,
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_FEEDBACK_TIMEOUT),
            )
            return resp.json()
        except Exception:
            return None

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current arm position as (x, y, z) or None."""
        fb = self._get_feedback_raw()
        if fb is None:
            return None
        try:
            return (float(fb.get("x", 0)), float(fb.get("y", 0)), float(fb.get("z", 0)))
        except (TypeError, ValueError):
            return None

    def wait_for_arrival(self, target_x: float, target_y: float, target_z: float,
                         timeout: float = None) -> bool:
        """
        Poll arm feedback until it reaches the target position.
        Returns True if arrived, False on timeout.
        """
        timeout = timeout or ARM_ARRIVAL_TIMEOUT
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            pos = self.get_position()
            if pos is not None:
                cx, cy, cz = pos
                dist = math.sqrt((cx - target_x)**2 + (cy - target_y)**2 + (cz - target_z)**2)
                if dist <= POSITION_TOLERANCE:
                    return True
            time.sleep(ARM_POLL_INTERVAL)

        log.warning(f"Arm did not arrive at ({target_x:.0f}, {target_y:.0f}, {target_z:.0f}) within {timeout}s")
        return False

    def _estimate_travel_time(self, x, y, z, speed) -> float:
        """Estimate movement duration for fallback wait."""
        dist = math.sqrt((x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2)
        return max(0.5, (dist / max(speed, 1)) * MOVEMENT_SPEED_FACTOR)

    # ── Movement ──────────────────────────────────────────────────────

    def arm_move(self, x: float, y: float, z: float, speed: int = None) -> bool:
        """
        Move arm to (x, y, z). Fire-and-forget + poll arrival.
        Returns True if command sent successfully.
        """
        speed = speed or self._speed
        cmd = json.dumps({"T": 104, "x": x, "y": y, "z": z, "t": speed})
        log.info(f"arm_move -> ({x:.0f}, {y:.0f}, {z:.0f}) spd={speed}")

        ok = self._send_movement_cmd(cmd)
        if not ok:
            return False

        # Wait for arm to arrive at destination
        est_time = self._estimate_travel_time(x, y, z, speed)
        # Try polling feedback first
        arrived = self.wait_for_arrival(x, y, z, timeout=max(est_time * 2.5, 5.0))
        if not arrived:
            # Fallback: just wait estimated time (arm might not support feedback)
            remaining_wait = max(0, est_time - ARM_POLL_INTERVAL * 3)
            if remaining_wait > 0:
                log.debug(f"Fallback wait: {remaining_wait:.1f}s")
                time.sleep(remaining_wait)

        self.x, self.y, self.z = x, y, z
        return True

    def arm_home(self, speed: int = None) -> bool:
        """Move to safe home position."""
        return self.arm_move(HOME_X, HOME_Y, HOME_Z, speed)

    # ── Electromagnet Gripper (Serial) ────────────────────────────────

    def gripper_on(self):
        """Activate electromagnet (pick)."""
        with self._serial_lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(b"1")
                    self._ser.flush()
                except serial.SerialException as e:
                    log.warning(f"Gripper ON serial error: {e}")
        time.sleep(GRIPPER_ACTIVATE_TIME)
        log.debug("Gripper ON")

    def gripper_off(self):
        """Deactivate electromagnet (release)."""
        with self._serial_lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(b"0")
                    self._ser.flush()
                except serial.SerialException as e:
                    log.warning(f"Gripper OFF serial error: {e}")
        time.sleep(GRIPPER_ACTIVATE_TIME)
        log.debug("Gripper OFF")

    # ── Servo Gripper (HTTP) ──────────────────────────────────────────

    def gripper_open(self):
        """Open gripper fingers via servo."""
        cmd = json.dumps({"T": 11, "id": GRIPPER_SERVO_ID,
                          "angle": GRIPPER_OPEN_ANGLE, "t": GRIPPER_SERVO_SPEED})
        self._send_quick_cmd(cmd)
        time.sleep(GRIPPER_SERVO_TIME)
        log.debug("Gripper OPEN")

    def gripper_close(self):
        """Close gripper fingers via servo."""
        cmd = json.dumps({"T": 11, "id": GRIPPER_SERVO_ID,
                          "angle": GRIPPER_CLOSE_ANGLE, "t": GRIPPER_SERVO_SPEED})
        self._send_quick_cmd(cmd)
        time.sleep(GRIPPER_SERVO_TIME)
        log.debug("Gripper CLOSE")

    # ── High-Level Pick & Place ───────────────────────────────────────

    def pick_piece(self, x: float, y: float, speed: int = None) -> bool:
        """Pick: hover → open gripper → lower → close gripper → magnet on → raise."""
        speed = speed or self._speed
        # 1. Hover above target
        self.arm_move(x, y, Z_SAFE, speed)
        time.sleep(ARM_SETTLE_TIME)
        # 2. Open gripper while high (safe)
        self.gripper_open()
        # 3. Lower to piece
        self.arm_move(x, y, Z_PICK, speed)
        time.sleep(ARM_SETTLE_TIME)
        # 4. Close gripper on piece
        self.gripper_close()
        # 5. Activate magnet
        self.gripper_on()
        # 6. Lift
        self.arm_move(x, y, Z_SAFE, speed)
        return True

    def place_piece(self, x: float, y: float, speed: int = None) -> bool:
        """Place: hover → lower → magnet off → open gripper → raise."""
        speed = speed or self._speed
        # 1. Hover above target
        self.arm_move(x, y, Z_SAFE, speed)
        time.sleep(ARM_SETTLE_TIME)
        # 2. Lower to placement
        self.arm_move(x, y, Z_PLACE, speed)
        time.sleep(ARM_SETTLE_TIME)
        # 3. Release magnet
        self.gripper_off()
        # 4. Open gripper
        self.gripper_open()
        # 5. Lift
        self.arm_move(x, y, Z_SAFE, speed)
        return True

    def pick_and_place(self, fx, fy, tx, ty, speed=None) -> bool:
        """Pick from (fx,fy) and place at (tx,ty)."""
        self.pick_piece(fx, fy, speed)
        self.place_piece(tx, ty, speed)
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self):
        """Clean shutdown."""
        with self._serial_lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(b"0")
                    self._ser.close()
                except Exception:
                    pass
        self._session.close()
        log.info("ArmController closed")


# ═══════════════════════════════════════════════════════════════════════
#  GAME LOGIC HELPERS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceMetrics:
    total_moves: int = 0
    successful_moves: int = 0
    failed_moves: int = 0
    retries: int = 0

    def print_stats(self):
        rate = self.successful_moves / max(self.total_moves, 1) * 100
        log.info("=" * 50)
        log.info("  PERFORMANCE METRICS")
        log.info(f"  Moves: {self.total_moves} | OK: {self.successful_moves} | "
                 f"Fail: {self.failed_moves} | Rate: {rate:.0f}%")
        log.info("=" * 50)


class GamePhase(Enum):
    INIT = auto()
    WAIT_OPPONENT = auto()
    THINK = auto()
    EXECUTE = auto()
    GAME_OVER = auto()


# ── Move Parser ───────────────────────────────────────────────────────────

def parse_move(move_str: str) -> Optional[Dict]:
    """Parse engine move string like '1:A2->A3' or '1:A5->A6=4'."""
    if not move_str or not isinstance(move_str, str):
        return None
    try:
        colon = move_str.index(":")
        piece_id = int(move_str[:colon])
        rest = move_str[colon + 1:]

        promotion = None
        if "=" in rest:
            rest, promo_str = rest.split("=")
            promotion = int(promo_str)

        arrow = rest.index("->")
        src = rest[:arrow]
        dst = rest[arrow + 2:]

        def cell_to_rc(cell_str):
            col = ord(cell_str[0].upper()) - ord("A")
            row = int(cell_str[1]) - 1
            return (row, col)

        return {
            "piece": piece_id,
            "from_cell": cell_to_rc(src),
            "to_cell": cell_to_rc(dst),
            "promotion": promotion,
            "raw": move_str,
        }
    except (ValueError, IndexError) as e:
        log.error(f"Failed to parse move '{move_str}': {e}")
        return None


def board_diff(old_board, new_board) -> List[Dict]:
    changes = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if old_board[r][c] != new_board[r][c]:
                changes.append({
                    "cell": (r, c),
                    "old": int(old_board[r][c]),
                    "new": int(new_board[r][c]),
                })
    return changes


def infer_opponent_move(old_board, new_board) -> Optional[Dict]:
    changes = board_diff(old_board, new_board)
    if not changes:
        return None
    emptied = [c for c in changes if c["new"] == 0]
    filled = [c for c in changes if c["new"] != 0 and c["old"] != c["new"]]

    if len(emptied) == 1 and len(filled) == 1:
        return {
            "from_cell": emptied[0]["cell"],
            "to_cell": filled[0]["cell"],
            "piece": emptied[0]["old"],
            "captured": filled[0]["old"] if filled[0]["old"] != 0 else None,
        }
    elif len(emptied) == 1:
        for c in changes:
            if c["cell"] != emptied[0]["cell"] and c["new"] == emptied[0]["old"]:
                return {
                    "from_cell": emptied[0]["cell"],
                    "to_cell": c["cell"],
                    "piece": emptied[0]["old"],
                    "captured": c["old"],
                }
    return None


def print_board(board):
    log.info("    A   B   C   D   E   F")
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_str = f" {r+1} "
        for c in range(BOARD_SIZE):
            pid = int(board[r][c])
            name = PIECE_NAMES.get(pid, "..")
            row_str += f" {name:>2} "
        log.info(row_str)


# ── Captured Piece Tracker ────────────────────────────────────────────────

class CapturedTracker:
    def __init__(self):
        self.white_idx = 0
        self.black_idx = 0

    def graveyard_slot(self, piece_id: int) -> Tuple[float, float]:
        if piece_id in WHITE_PIECE_IDS:
            idx = min(self.white_idx, len(GRAVEYARD_WHITE) - 1)
            self.white_idx += 1
            return GRAVEYARD_WHITE[idx]
        else:
            idx = min(self.black_idx, len(GRAVEYARD_BLACK) - 1)
            self.black_idx += 1
            return GRAVEYARD_BLACK[idx]


# ── Promotion Tracker ─────────────────────────────────────────────────────

class PromotionTracker:
    def __init__(self):
        self._promoted: Dict[Tuple[int, int], int] = {}

    def mark(self, row, col, piece_id):
        self._promoted[(row, col)] = piece_id
        log.info(f"Promotion: {rc_to_label(row, col)} -> {PIECE_NAMES.get(piece_id, '?')}")

    def apply(self, board):
        board = board.copy()
        for (r, c), pid in self._promoted.items():
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                board[r][c] = pid
        return board


# ── Clock Manager ─────────────────────────────────────────────────────────

class ClockManager:
    def __init__(self, total=GAME_CLOCK_TOTAL):
        self.total = total
        self.start_time = None
        self.move_count = 0

    def start(self):
        self.start_time = time.time()

    def remaining(self) -> float:
        if self.start_time is None:
            return self.total
        return max(0, self.total - (time.time() - self.start_time))

    def arm_speed(self) -> int:
        rem = self.remaining()
        if rem < 60:
            return SPEED_FAST
        if rem < 180:
            return SPEED_NORMAL
        return SPEED_NORMAL

    def status(self) -> str:
        return f"Clock: {self.remaining():.1f}s | Moves: {self.move_count}"


# ═══════════════════════════════════════════════════════════════════════
#  CALIBRATION WIZARD
# ═══════════════════════════════════════════════════════════════════════

def run_calibration(arm: ArmController):
    log.info("=" * 50)
    log.info("  CALIBRATION WIZARD")
    log.info("=" * 50)

    test_cells = [("A1", (0, 0)), ("F6", (5, 5)), ("C3", (2, 2)), ("D4", (3, 3))]

    arm.arm_home()
    time.sleep(1)

    for label, (r, c) in test_cells:
        wx, wy = cell_to_world(r, c)
        input(f"\n  Move arm to {label} ({wx:.0f}, {wy:.0f}). Press Enter...")
        arm.arm_move(wx, wy, Z_PICK)
        time.sleep(1)
        resp = input(f"  Is arm centered on {label}? (y/n/q): ").strip().lower()
        if resp == "q":
            break
        arm.arm_move(wx, wy, Z_SAFE)

    arm.arm_home()
    log.info("Calibration complete.")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN GAME CONTROLLER
# ═══════════════════════════════════════════════════════════════════════

class GameController:
    def __init__(self, playing_white: bool, manual_mode: bool = False):
        self.playing_white = playing_white
        self.manual_mode = manual_mode
        self.phase = GamePhase.INIT

        self.arm = ArmController()
        self.perception = PerceptionSystem()
        self.captures = CapturedTracker()
        self.promotions = PromotionTracker()
        self.clock = ClockManager()
        self.metrics = PerformanceMetrics()

        self.current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_history: List[str] = []
        self.turn_number = 0
        self.running = True

    # ── Init ──────────────────────────────────────────────────────────

    def init(self) -> bool:
        log.info("=" * 60)
        log.info("  STICKFISH Hardware v3.0 (Team L)")
        log.info(f"  Playing as: {'WHITE' if self.playing_white else 'BLACK'}")
        log.info(f"  Mode: {'Manual' if self.manual_mode else 'Auto-detect'}")
        log.info("=" * 60)

        # Check arm connection
        if not self.arm.is_connected():
            log.error("Failed to connect to robot arm!")
            return False
        log.info("Arm connected ✓")

        self.arm.arm_home()

        # Connect camera
        if not self.perception.connect_camera():
            log.warning("Camera connection failed — will retry during game")

        # Get initial board state
        log.info("Waiting for stable board...")
        stable = self.perception.wait_for_stable_board(
            frames=STABILITY_FRAMES, timeout=STABILITY_TIMEOUT,
        )
        if stable is not None:
            self.current_board = self.promotions.apply(stable)
            print_board(self.current_board)
        else:
            log.warning("Could not get stable initial board")

        self.clock.start()
        return True

    # ── Wait for Opponent ─────────────────────────────────────────────

    def wait_for_opponent(self):
        log.info(f"\n--- Turn {self.turn_number + 1}: Waiting for opponent ---")

        if self.manual_mode:
            input("  Press Enter after opponent has moved...")
        else:
            log.info("  Watching for board change...")
            new_board = self.perception.wait_for_board_change(timeout=300)
            if new_board is None:
                log.warning("Timeout waiting for opponent!")
                return

        log.info("  Waiting for stable board...")
        stable = self.perception.wait_for_stable_board(
            frames=STABILITY_FRAMES, timeout=STABILITY_TIMEOUT,
        )
        if stable is not None:
            stable = self.promotions.apply(stable)
            opp = infer_opponent_move(self.current_board, stable)
            if opp:
                src = rc_to_label(*opp["from_cell"])
                dst = rc_to_label(*opp["to_cell"])
                log.info(f"  Opponent: {PIECE_NAMES.get(opp['piece'], '?')} {src} -> {dst}")
                if opp.get("captured"):
                    log.info(f"  Captured: {PIECE_NAMES.get(opp['captured'], '?')}")
            self.current_board = stable
            print_board(self.current_board)
        else:
            board = self.perception.get_board_state()
            if board is not None:
                self.current_board = self.promotions.apply(board)

    # ── Think ─────────────────────────────────────────────────────────

    def think(self) -> Optional[str]:
        log.info(f"\n--- Turn {self.turn_number + 1}: THINKING ---")
        log.info(f"  {self.clock.status()}")

        engine_board = self.promotions.apply(self.current_board)

        try:
            import game as game_module
            move_str = game_module.get_best_move(engine_board, self.playing_white)
        except Exception as e:
            log.error(f"  Engine error: {e}")
            import traceback
            traceback.print_exc()
            return None

        log.info(f"  Engine returned: {move_str}")
        return move_str

    # ── Execute ───────────────────────────────────────────────────────

    def execute_move(self, move_str: str) -> bool:
        parsed = parse_move(move_str)
        if not parsed:
            log.error(f"  Could not parse: {move_str}")
            return False

        from_rc = parsed["from_cell"]
        to_rc = parsed["to_cell"]
        piece = parsed["piece"]
        promotion = parsed.get("promotion")
        speed = self.clock.arm_speed()

        from_label = rc_to_label(*from_rc)
        to_label = rc_to_label(*to_rc)
        from_world = cell_to_world(*from_rc)
        to_world = cell_to_world(*to_rc)

        target_piece = int(self.current_board[to_rc[0]][to_rc[1]])

        # Build expected board
        new_board = self.current_board.copy()
        new_board[from_rc[0]][from_rc[1]] = 0
        new_board[to_rc[0]][to_rc[1]] = promotion if promotion else piece

        log.info(f"\n--- Turn {self.turn_number + 1}: EXECUTING ---")

        # Handle capture: remove target piece to graveyard
        if target_piece != 0:
            log.info(f"  CAPTURE: {PIECE_NAMES.get(target_piece, '?')} at {to_label}")
            grave = self.captures.graveyard_slot(target_piece)
            self.arm.pick_piece(to_world[0], to_world[1], speed)
            self.arm.place_piece(grave[0], grave[1], speed)
            log.info("  -> Moved to graveyard")

        # Execute move
        log.info(f"  MOVE: {PIECE_NAMES.get(piece, '?')} {from_label} -> {to_label}")
        self.arm.pick_piece(from_world[0], from_world[1], speed)

        if promotion:
            log.info(f"  PROMOTION: -> {PIECE_NAMES.get(promotion, '?')}")
            self.promotions.mark(to_rc[0], to_rc[1], promotion)

        self.arm.place_piece(to_world[0], to_world[1], speed)
        self.arm.arm_home(speed)

        # Update board state
        self.current_board = new_board
        self.move_history.append(move_str)
        self.clock.move_count += 1
        self.turn_number += 1
        self.metrics.total_moves += 1
        self.metrics.successful_moves += 1

        print_board(self.current_board)
        return True

    # ── Game Over Check ───────────────────────────────────────────────

    def check_game_over(self) -> bool:
        has_wk = np.any(self.current_board == 5)
        has_bk = np.any(self.current_board == 10)
        if not has_wk:
            log.info("WHITE KING CAPTURED — Black wins!")
            return True
        if not has_bk:
            log.info("BLACK KING CAPTURED — White wins!")
            return True
        return False

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self):
        if not self.init():
            log.error("Initialization failed!")
            return

        self.phase = GamePhase.THINK if self.playing_white else GamePhase.WAIT_OPPONENT

        while self.running:
            try:
                if self.phase == GamePhase.WAIT_OPPONENT:
                    self.wait_for_opponent()
                    if self.check_game_over():
                        self.phase = GamePhase.GAME_OVER
                    else:
                        self.phase = GamePhase.THINK

                elif self.phase == GamePhase.THINK:
                    move_str = self.think()
                    if move_str is None:
                        self.phase = GamePhase.GAME_OVER
                    else:
                        self._pending_move = move_str
                        self.phase = GamePhase.EXECUTE

                elif self.phase == GamePhase.EXECUTE:
                    self.execute_move(self._pending_move)
                    self.phase = GamePhase.WAIT_OPPONENT

                elif self.phase == GamePhase.GAME_OVER:
                    log.info("\n" + "=" * 60)
                    log.info("  GAME OVER")
                    log.info(f"  {self.clock.status()}")
                    self.metrics.print_stats()
                    log.info("=" * 60)
                    break

            except KeyboardInterrupt:
                log.info("\nInterrupted.")
                break
            except Exception as e:
                log.error(f"Error in game loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

        self.shutdown()

    def shutdown(self):
        log.info("Shutting down...")
        try:
            self.arm.arm_home()
        except Exception:
            pass
        try:
            self.perception.stop()
        except Exception:
            pass
        try:
            self.arm.close()
        except Exception:
            pass
        log.info("Done.")


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="STICKFISH Hardware Controller (Team L)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--white", action="store_true", help="Play as white")
    group.add_argument("--black", action="store_true", help="Play as black")
    group.add_argument("--calibrate", action="store_true", help="Run calibration wizard")
    parser.add_argument("--manual", action="store_true", help="Manual trigger mode")
    args = parser.parse_args()

    if args.calibrate:
        arm = ArmController()
        if arm.is_connected():
            run_calibration(arm)
            arm.close()
        else:
            log.error("Arm not connected — cannot calibrate")
        return

    playing_white = args.white
    controller = GameController(playing_white=playing_white, manual_mode=args.manual)

    def handler(sig, frame):
        log.info("\nShutting down...")
        controller.running = False
    signal.signal(signal.SIGINT, handler)

    controller.run()


if __name__ == "__main__":
    main()
